# =============================================================================
# Ablation Study: Spiking Disentangled DBformer (SDBformer)
# Dataset: DTU (2-second overlapping windows, 18 subjects, LOSO-CV)
# Features Supported: 
#   - Domain Adversarial Training (GRL)
#   - Subject-Specific Domain Classification
#   - Orthogonal Constraint
#   - Spatial/Temporal Branch Fusion Ablation
# =============================================================================

import argparse
import math
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.nn.init import trunc_normal_
import matplotlib.pyplot as plt

# =============================================================================
# 1. Utilities & Loss Functions
# =============================================================================
class AvgMeter:
    """Computes and stores the average and current value for metrics."""
    def __init__(self, name="Metric"): 
        self.name = name
        self.reset()
        
    def reset(self): 
        self.avg, self.sum, self.count = [0] * 3
        
    def update(self, val, count=1): 
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count
        
    def __repr__(self): 
        return f"{self.name}: {self.avg:.4f}"

def get_lr(optimizer):
    """Retrieves the current learning rate from the optimizer."""
    for param_group in optimizer.param_groups: 
        return param_group["lr"]

def calc_orthogonal_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Computes the orthogonal loss between two latent representations.
    Forces the task and subject features to be structurally independent.
    """
    z1_norm = F.normalize(z1, p=2, dim=1)
    z2_norm = F.normalize(z2, p=2, dim=1)
    cos_sim = torch.sum(z1_norm * z2_norm, dim=1)
    return torch.mean(cos_sim ** 2)

# =============================================================================
# 2. Dataset Loader (DTU 2-Second Pre-sliced Windows)
# =============================================================================
class DTU_AAD_Dataset_2s(Dataset):
    """
    Dataset loader for the DTU dataset using pre-sliced 2-second windows.
    Data is pre-loaded into memory to accelerate the training process.
    Note: Requires sufficient RAM if loading many subjects simultaneously.
    """
    def __init__(self, root, subject_ids, all_subject_ids_map):
        self.root = root
        self.data_cache = {} 
        self.index_map = []
        self.domain_label_map = {str(k): v for k, v in all_subject_ids_map.items()}

        for s_id in subject_ids:
            # 注意：这里假设您已经提前生成了 2秒 的数据文件
            # 如果您的文件名不同，请相应修改此处
            file_name = f"S{s_id}_Dataset_2s.npz"
            file_path = os.path.join(root, file_name)
            
            if not os.path.exists(file_path):
                warnings.warn(f"Warning: Data file not found for subject {s_id}: {file_path}")
                continue
            
            # Load and normalize data
            data = np.load(file_path, allow_pickle=True)
            labels = torch.tensor([int(item[0]) for item in data['event_slices']], dtype=torch.long) - 1
            # 此时的 eeg_slices 形状应该已经是 (N, Channels, 256) 即 2秒的数据
            eeg_data = torch.tensor(data['eeg_slices'], dtype=torch.float32)
            
            eeg_min = eeg_data.min(dim=-1, keepdim=True)[0]
            eeg_max = eeg_data.max(dim=-1, keepdim=True)[0]
            eeg_normalized = (eeg_data - eeg_min) / (eeg_max - eeg_min + 1e-8)
            
            self.data_cache[str(s_id)] = {
                'eeg': eeg_normalized,
                'direction_label': labels
            }
            
            # Directly map all valid samples since they are already 2-second windows
            for sample_idx in range(len(labels)):
                self.index_map.append({
                    'subject_id': str(s_id), 
                    'sample_idx': sample_idx
                })

        if not self.index_map: 
            raise RuntimeError(f"Dataset is empty for subjects: {subject_ids}. Check data root.")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        map_entry = self.index_map[idx]
        s_id_str, sample_idx = map_entry['subject_id'], map_entry['sample_idx']
        
        subject_data = self.data_cache[s_id_str]
        
        # Directly retrieve the pre-sliced 2-second EEG data
        eeg_2s = subject_data['eeg'][sample_idx]
        direction_label = subject_data['direction_label'][sample_idx]
        
        # Get the domain label for DANN
        domain_label = self.domain_label_map.get(s_id_str, -1) 
        
        return eeg_2s, direction_label, torch.tensor(domain_label, dtype=torch.long)
    
# =============================================================================
# 3. SNN Surrogate Gradients & GRL
# =============================================================================
class GradientReversalFunction(Function):
    """Core function for the Gradient Reversal Layer (GRL)."""
    @staticmethod
    def forward(ctx, x, alpha): 
        ctx.alpha = alpha
        return x.view_as(x)
        
    @staticmethod
    def backward(ctx, grad_output): 
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    """Inverts gradients during backpropagation to enable adversarial training."""
    def __init__(self, alpha: float = 1.0): 
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x, alpha=None): 
        return GradientReversalFunction.apply(x, alpha if alpha is not None else self.alpha)

class Quant(torch.autograd.Function):
    """
    Surrogate Gradient Function for SNN.
    Implements the Straight-Through Estimator (STE) for discrete spike backpropagation.
    """
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, i, min_value=0.0, max_value=4.0):
        ctx.min, ctx.max = min_value, max_value
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))
        
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        i, = ctx.saved_tensors
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        return grad_input, None, None

class Multispike(nn.Module):
    """Multi-bit spiking neuron activation."""
    def __init__(self, min_value=0.0, max_value=4.0):
        super().__init__()
        self.min_value, self.max_value = min_value, max_value
        
    def forward(self, x): 
        return Quant.apply(x, self.min_value, self.max_value) / 2.0

# =============================================================================
# 4. Spiking Network Modules
# =============================================================================
class SNN_Attention_BNC(nn.Module):
    """Spike-Driven Self-Attention (SDSA) mechanism."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads, self.scale = num_heads, 0.125
        self.head_lif, self.q_lif, self.k_lif, self.v_lif, self.attn_lif = [Multispike() for _ in range(5)]
        self.q_conv, self.k_conv, self.v_conv, self.proj_conv = [nn.Conv1d(dim, dim, 1, bias=False) for _ in range(4)]
        self.q_bn, self.k_bn, self.v_bn, self.proj_bn = [nn.BatchNorm1d(dim) for _ in range(4)]

    def forward(self, x):
        B, N, C = x.shape
        x = self.head_lif(x)
        x_t = x.transpose(1, 2)
        
        q = self.q_lif(self.q_bn(self.q_conv(x_t))).transpose(1, 2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_lif(self.k_bn(self.k_conv(x_t))).transpose(1, 2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_lif(self.v_bn(self.v_conv(x_t))).transpose(1, 2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x_attn = (q @ (k.transpose(-2, -1) @ v)) * self.scale    
        x = self.attn_lif(x_attn.transpose(1, 2).reshape(B, N, C))
        return self.proj_bn(self.proj_conv(x.transpose(1, 2))).transpose(1, 2)

class SNN_MLP_BNC(nn.Module):
    """Spiking Multi-Layer Perceptron (Feed-Forward Network)."""
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hf = hidden_features or in_features
        self.mlp1_conv, self.mlp2_conv = nn.Conv1d(in_features, hf, 1), nn.Conv1d(hf, in_features, 1)
        self.mlp1_bn, self.mlp2_bn = nn.BatchNorm1d(hf), nn.BatchNorm1d(in_features)
        self.mlp1_lif, self.mlp2_lif = Multispike(), Multispike()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.mlp1_bn(self.mlp1_conv(self.mlp1_lif(x)))
        return self.mlp2_bn(self.mlp2_conv(self.mlp2_lif(x))).transpose(1, 2)

class SNN_TransformerBlock(nn.Module):
    """Spiking Transformer Encoder Block."""
    def __init__(self, emb_size, num_heads):
        super().__init__()
        self.attn = SNN_Attention_BNC(emb_size, num_heads)
        self.mlp = SNN_MLP_BNC(emb_size, emb_size * 4) 
        
    def forward(self, x): 
        return x + self.mlp(x + self.attn(x))

class SNN_Stem(nn.Module):
    """Spiking convolutional stem for temporal feature extraction."""
    def __init__(self, in_planes, out_planes, kernel_size, patch_size, radix=1):
        super().__init__()
        self.radix, self.out_planes = radix, out_planes
        self.sconv = nn.Conv1d(in_planes, out_planes * radix, 1, bias=False, groups=radix)
        self.bn1 = nn.BatchNorm1d(out_planes * radix)
        self.tconv = nn.ModuleList([
            nn.Sequential(nn.Conv1d(out_planes, out_planes, ks, 1, padding=ks//2, bias=False, groups=out_planes), nn.BatchNorm1d(out_planes)) 
            for ks in [kernel_size // (2**i) for i in range(radix)]
        ])
        self.lif1 = Multispike()
        self.downSampling = nn.AvgPool1d(patch_size, stride=patch_size)

    def forward(self, x):
        out = self.bn1(self.sconv(x))
        branches = torch.split(out, self.out_planes, dim=1) if self.radix > 1 else [out]
        return self.downSampling(self.lif1(sum([conv(b) for conv, b in zip(self.tconv, branches)])))

class SNN_PatchEmbeddingTemporal(nn.Module):
    def __init__(self, chn, patch_size, emb_size):
        super().__init__()
        self.stem = SNN_Stem(in_planes=chn, out_planes=emb_size, kernel_size=63, patch_size=patch_size)
    def forward(self, x): 
        return self.stem(x).permute(0, 2, 1)

class SNN_PatchEmbeddingSpatial(nn.Module):
    def __init__(self, spa_dim, emb_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, spa_dim, 25, 5, 12), nn.BatchNorm1d(spa_dim), Multispike(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), 
            nn.Linear(spa_dim, emb_size), nn.BatchNorm1d(emb_size), Multispike()
        )
    def forward(self, x):
        B, C, T = x.shape
        return self.encoder(x.unsqueeze(1).reshape(B * C, 1, T)).view(B, C, -1)

# =============================================================================
# 5. Model Architecture (Ablation-Ready SDBformer)
# =============================================================================
class Ablation_SNN_Disentangled_DBformer(nn.Module):
    """
    SDBformer architecture designed specifically for ablation studies.
    Supports dynamic enabling/disabling of:
      - Domain Adversarial Classifier (use_dom_adv)
      - Specific Domain Classifier (use_dom_spec)
      - Feature Fusion Strategies (attentive, spatial_only, temporal_only)
    """
    def __init__(self, chn, time_sample_num, patch_size, spa_dim, num_domain_classes,
                 emb_size=128, depth=2, num_heads=8,
                 use_dom_adv=True, use_dom_spec=True, fusion_type='attentive'):
        super().__init__()
        self.use_dom_adv = use_dom_adv
        self.use_dom_spec = use_dom_spec
        self.fusion_type = fusion_type
        
        # Encoders & Positional Embeddings
        self.temporal_embedding = SNN_PatchEmbeddingTemporal(chn, patch_size, emb_size)
        self.spatial_embedding = SNN_PatchEmbeddingSpatial(spa_dim, emb_size)
        self.pos_embedding_temporal = nn.Parameter(torch.randn(1, time_sample_num // patch_size, emb_size))
        self.pos_embedding_spatial = nn.Parameter(torch.randn(1, chn, emb_size))
        
        self.temporal_transformer = nn.ModuleList([SNN_TransformerBlock(emb_size, num_heads) for _ in range(depth)])
        self.spatial_transformer = nn.ModuleList([SNN_TransformerBlock(emb_size, num_heads) for _ in range(depth)])

        # Fusion Pooling
        if fusion_type == 'attentive':
            self.spatial_attn_pool = nn.Sequential(nn.Linear(emb_size, emb_size), Multispike(), nn.Linear(emb_size, 1))
        
        fusion_dim = emb_size * 2 if fusion_type == 'attentive' else emb_size
        self.is_disentangled = use_dom_adv or use_dom_spec
        
        # Feature Disentanglement Mapping
        if self.is_disentangled:
            self.encoding_head = nn.Sequential(nn.Linear(fusion_dim, emb_size * 2), nn.BatchNorm1d(emb_size * 2), Multispike())
            self.grl = GradientReversalLayer()
            
        # Classifiers
        clf_in_dim = emb_size if self.is_disentangled else fusion_dim
        self.label_classifier = nn.Sequential(nn.Linear(clf_in_dim, 64), nn.BatchNorm1d(64), Multispike(), nn.Linear(64, 2))
        
        if use_dom_adv:
            self.domain_classifier_adv = nn.Sequential(nn.Linear(emb_size, 64), nn.BatchNorm1d(64), Multispike(), nn.Linear(64, num_domain_classes))
        if use_dom_spec:
            self.domain_classifier_spec = nn.Sequential(nn.Linear(emb_size, 64), nn.BatchNorm1d(64), Multispike(), nn.Linear(64, num_domain_classes))
            
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear): 
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d): 
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)

    def forward(self, x, alpha=1.0):
        x_t, x_s = None, None
        
        # Temporal Branch
        if self.fusion_type != 'spatial_only':
            x_temp = self.temporal_embedding(x) + self.pos_embedding_temporal
            for block in self.temporal_transformer: x_temp = block(x_temp)
            x_t = x_temp.mean(dim=1)
            
        # Spatial Branch
        if self.fusion_type != 'temporal_only':
            x_spat = self.spatial_embedding(x) + self.pos_embedding_spatial
            for block in self.spatial_transformer: x_spat = block(x_spat)
            if self.fusion_type == 'attentive':
                x_s = torch.sum(torch.softmax(self.spatial_attn_pool(x_spat), dim=1) * x_spat, dim=1)
            else:
                x_s = x_spat.mean(dim=1)

        # Feature Fusion
        if self.fusion_type == 'attentive': x_fused = torch.cat([x_t, x_s], dim=-1)
        elif self.fusion_type == 'temporal_only': x_fused = x_t
        elif self.fusion_type == 'spatial_only': x_fused = x_s

        label_out, dom_adv_out, dom_spec_out, z_task, z_subj = None, None, None, None, None
        
        # Classification & Disentanglement
        if self.is_disentangled:
            z_task, z_subj = torch.chunk(self.encoding_head(x_fused), 2, dim=1)
            label_out = self.label_classifier(z_task)
            
            if self.use_dom_adv: 
                dom_adv_out = self.domain_classifier_adv(self.grl(z_task, alpha))
            if self.use_dom_spec: 
                dom_spec_out = self.domain_classifier_spec(z_subj)
        else:
            z_task = x_fused
            label_out = self.label_classifier(z_task)

        return label_out, dom_adv_out, dom_spec_out, z_task, z_subj

# =============================================================================
# 6. Ablation Trainer
# =============================================================================
class AblationTrainer:
    """Trainer class engineered to handle varying combinations of disentanglement losses."""
    def __init__(self, model, optimizer, config):
        self.model, self.optimizer, self.config = model, optimizer, config
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = {
            'train_loss': AvgMeter("Loss"), 
            'train_acc': AvgMeter("L_Acc"),
            'test_acc': AvgMeter("Test_Acc")
        }

    def run_epoch(self, dataloader, mode='train', epoch=0):
        self.model.train() if mode == 'train' else self.model.eval()
        for m in self.metrics.values(): m.reset()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.config.epochs} [{mode.capitalize()}]")
        with torch.set_grad_enabled(mode == 'train'):
            for eeg, labels, domains in pbar:
                eeg = eeg.to(self.config.device)
                labels = labels.to(self.config.device)
                domains = domains.to(self.config.device)
                
                if mode == 'train':
                    # Dynamic Alpha for GRL scheduling
                    progress = (epoch * len(dataloader) + pbar.n) / (self.config.epochs * len(dataloader))
                    alpha = 2. / (1. + math.exp(-10 * progress)) - 1
                    
                    self.optimizer.zero_grad()
                    l_out, da_out, ds_out, z_t, z_s = self.model(eeg, alpha)
                    
                    # Accumulate Active Losses
                    loss = self.criterion(l_out, labels)
                    if self.config.use_dom_adv and da_out is not None:
                        loss += self.criterion(da_out, domains)
                    if self.config.use_dom_spec and ds_out is not None:
                        loss += self.criterion(ds_out, domains)
                    if self.config.use_ortho and z_t is not None and z_s is not None:
                        loss += self.config.lambda_ortho * calc_orthogonal_loss(z_t, z_s)

                    loss.backward()
                    self.optimizer.step()
                    
                    self.metrics['train_loss'].update(loss.item(), eeg.size(0))
                    self.metrics['train_acc'].update((l_out.argmax(1) == labels).float().mean().item(), eeg.size(0))
                    pbar.set_postfix({'Loss': f"{self.metrics['train_loss'].avg:.4f}", 'Acc': f"{self.metrics['train_acc'].avg:.2%}"})
                
                else:
                    # Test Phase (No GRL impact needed)
                    l_out, _, _, _, _ = self.model(eeg, alpha=0)
                    self.metrics['test_acc'].update((l_out.argmax(1) == labels).float().mean().item(), eeg.size(0))
                    pbar.set_postfix({'Acc': f"{self.metrics['test_acc'].avg:.2%}"})
                    
        return self.metrics

# =============================================================================
# 7. Experiment Configuration & Execution Loop
# =============================================================================
class TrainingConfig:
    def __init__(self):
        # IMPORTANT: Update this path to where your DTU dataset is stored
        self.data_root = "./data/DTU_Dataset" 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set num_workers to 0 to prevent shared memory issues during parallel loading
        self.batch_size, self.num_workers = 64, 0 
        self.learning_rate, self.epochs, self.weight_decay = 1e-4, 50, 1e-3
        
        # DTU Dataset parameters
        self.channel_size = 66       
        self.time_sample_num = 256   
        self.patch_size = 16
        self.spa_dim = 16
        self.emb_size, self.depth, self.num_heads = 128, 3, 8
        
        self.lambda_ortho = 0.1

def main():
    # Parse arguments to allow running specific ablations on clusters
    parser = argparse.ArgumentParser(description="Run specific DTU SNN Ablation experiments.")
    parser.add_argument(
        '--exp_indices', 
        type=int, 
        nargs='+', 
        default=[0, 1, 2, 3, 4, 5, 6], 
        help='Indices of ablation experiments to run (0 to 6). Example: --exp_indices 0 1 2'
    )
    args = parser.parse_args()

    # The comprehensive list of all defined ablation conditions
    all_ablation_configs = [
        {'name': 'Ours (Full)', 'use_dom_adv': True, 'use_dom_spec': True, 'use_ortho': True, 'fusion_type': 'attentive'},
        {'name': 'w/o Dom_Adv', 'use_dom_adv': False, 'use_dom_spec': True, 'use_ortho': True, 'fusion_type': 'attentive'},
        {'name': 'w/o Dom_Spec', 'use_dom_adv': True, 'use_dom_spec': False, 'use_ortho': True, 'fusion_type': 'attentive'},
        {'name': 'w/o Ortho', 'use_dom_adv': True, 'use_dom_spec': True, 'use_ortho': False, 'fusion_type': 'attentive'},
        {'name': 'w/o Disentangle (Base)', 'use_dom_adv': False, 'use_dom_spec': False, 'use_ortho': False, 'fusion_type': 'attentive'},
        {'name': 'Temporal-Only', 'use_dom_adv': True, 'use_dom_spec': True, 'use_ortho': True, 'fusion_type': 'temporal_only'},
        {'name': 'Spatial-Only', 'use_dom_adv': True, 'use_dom_spec': True, 'use_ortho': True, 'fusion_type': 'spatial_only'},
    ]

    try:
        ablation_configs = [all_ablation_configs[i] for i in args.exp_indices]
    except IndexError:
        print(f"Error: Provided indices are out of range. Max index is {len(all_ablation_configs)-1}")
        return

    print("\n" + "*"*80)
    print(f"Target Ablation Experiments for this run:")
    for idx, cfg in zip(args.exp_indices, ablation_configs):
        print(f"  [{idx}] -> {cfg['name']}")
    print("*"*80 + "\n")

    all_ablation_results = []
    
    # DTU contains 18 subjects
    all_subject_ids = [str(i) for i in range(1, 19)] 
    
    for exp_cfg in ablation_configs:
        config = TrainingConfig()
        config.use_dom_adv = exp_cfg['use_dom_adv']
        config.use_dom_spec = exp_cfg['use_dom_spec']
        config.use_ortho = exp_cfg['use_ortho']
        config.fusion_type = exp_cfg['fusion_type']
        
        print(f"\n{'='*80}\nInitiating Ablation: {exp_cfg['name']}\n{'='*80}")
        fold_results = []

        for i, test_sub in enumerate(all_subject_ids):
            train_ids = [s for s in all_subject_ids if s != test_sub]
            domain_map = {s: idx for idx, s in enumerate(train_ids)}
            
            train_loader = DataLoader(DTU_AAD_Dataset_2s(config.data_root, train_ids, domain_map), 
                                      batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
            test_loader = DataLoader(DTU_AAD_Dataset_2s(config.data_root, [test_sub], domain_map), 
                                     batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
            
            model = Ablation_SNN_Disentangled_DBformer(
                config.channel_size, config.time_sample_num, config.patch_size, config.spa_dim, len(train_ids),
                config.emb_size, config.depth, config.num_heads,
                use_dom_adv=config.use_dom_adv, use_dom_spec=config.use_dom_spec, fusion_type=config.fusion_type
            ).to(config.device)
            
            optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            trainer = AblationTrainer(model, optimizer, config)
            
            best_acc = 0
            for epoch in range(config.epochs):
                trainer.run_epoch(train_loader, 'train', epoch)
                acc = trainer.run_epoch(test_loader, 'test', epoch)['test_acc'].avg
                if acc > best_acc: 
                    best_acc = acc
            
            print(f"[{exp_cfg['name']}] FOLD {i+1} (Test Sub {test_sub}) Best Accuracy: {best_acc:.4f}")
            fold_results.append(best_acc)

        mean_acc, std_acc = np.mean(fold_results), np.std(fold_results)
        all_ablation_results.append({
            'Experiment': exp_cfg['name'], 
            'Mean Accuracy': f"{mean_acc:.2%}", 
            'Std Dev': f"{std_acc:.2%}"
        })
        print(f"\nExperiment '{exp_cfg['name']}' Completed! Mean: {mean_acc:.4f} ± {std_acc:.4f}")

    print("\n" + "="*80 + "\nAll Scheduled Ablations Completed! Summary:\n" + "="*80)
    results_df = pd.DataFrame(all_ablation_results)
    print(results_df.to_string(index=False))
    
    # Save results to a CSV file dynamically named based on the running indices
    idx_str = "_".join(map(str, args.exp_indices))
    file_name = f'./dtu_snn_ablation_summary_exp_{idx_str}_{datetime.now().strftime("%Y%m%d-%H%M")}.csv'
    results_df.to_csv(file_name, index=False)
    print(f"\nResults successfully exported to: {file_name}")

if __name__ == "__main__":
    main()