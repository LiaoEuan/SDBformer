# =============================================================================
# 1. Imports
# =============================================================================
import json
import math
import os
import random
import time
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torch.nn.init import trunc_normal_
import matplotlib.pyplot as plt
from tqdm import tqdm
from mne.decoding import CSP

# =============================================================================
# 2. Utility Classes & Functions
# =============================================================================

class AvgMeter:
    """Computes and stores the average and current value."""
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

def count_parameters(model):
    """Counts the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =============================================================================
# 3. Dataset Loader
# =============================================================================

class DTU_AAD_Dataset_WS(Dataset):
    """
    Dataset loader for DTU Differential Entropy (DE) features.
    Designed for Within-Subject (WS) analysis.
    """
    def __init__(self, root, subject_id):
        self.root = root
        self.subject_id = str(subject_id)
        
        # Ensure the filename matches your data generation script
        file_name = f"S{self.subject_id}_DE_Features_1s.npz"
        file_path = os.path.join(root, file_name)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Error: DE feature file not found: {file_path}")
            
        print(f"  Loading {file_name} ...")
        try:
            data = np.load(file_path, allow_pickle=True)
            # Load 'DE' features and 'labels'
            # Shape: (N_samples, Channels, Frequency_Bands) -> (N, 66, 5)
            self.eeg_features = torch.tensor(data['DE'], dtype=torch.float32)
            self.direction_label = torch.tensor(data['labels'], dtype=torch.long)
            
        except Exception as e:
            raise IOError(f"Error loading {file_path}: {e}")

        if len(self.eeg_features) != len(self.direction_label):
            raise ValueError("Mismatch between DE features and label samples!")
            
    def __len__(self):
        return len(self.eeg_features)
    
    def __getitem__(self, idx):
        return self.eeg_features[idx], self.direction_label[idx]

# =============================================================================
# 4. SNN Core Components (Surrogate Gradients)
# =============================================================================

class Quant(torch.autograd.Function):
    """
    Surrogate gradient function for SNN training.
    Implements a custom backward pass to handle non-differentiable spikes.
    """
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, i, min_value=0.0, max_value=4.0):
        ctx.min = min_value
        ctx.max = max_value
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))

    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        i, = ctx.saved_tensors
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        return grad_input, None, None

class Multispike(nn.Module):
    """Multi-bit spike activation function."""
    def __init__(self, min_value=0.0, max_value=4.0):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    @staticmethod
    def spike_function(x, min_value, max_value):
        return Quant.apply(x, min_value, max_value)

    def forward(self, x):
        return self.spike_function(x, self.min_value, self.max_value) / 2.0

# =============================================================================
# 5. Fused SNN Layers (Conv/Linear + BN)
# =============================================================================

class SNN_Conv1d_BN(nn.Module):
    """
    Fused 1D Convolution + BatchNorm + Spike module.
    Allows for merging BN parameters into Conv weights during inference for efficiency.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.spike = Multispike()
        
    def forward(self, x):
        return self.spike(self.bn(self.conv(x)))
        
    def fuse_module(self):
        """Returns a new fused module (nn.Sequential) with BN parameters folded into Conv."""
        if self.training:
            print("Warning: fuse_module() should only be called in .eval() mode.")
            return self
            
        conv_w = self.conv.weight
        conv_b = self.conv.bias if self.conv.bias is not None else torch.zeros(self.conv.out_channels, device=conv_w.device)
        bn_rm = self.bn.running_mean
        bn_rv = self.bn.running_var
        bn_eps = self.bn.eps
        bn_w = self.bn.weight
        bn_b = self.bn.bias

        scale = bn_w / torch.sqrt(bn_rv + bn_eps)
        fused_w = conv_w * scale.view(-1, 1, 1)
        fused_b = (conv_b - bn_rm) * scale + bn_b

        fused_conv = nn.Conv1d(
            self.conv.in_channels, self.conv.out_channels, self.conv.kernel_size,
            self.conv.stride, self.conv.padding, groups=self.conv.groups, bias=True
        ).to(fused_w.device)
        
        fused_conv.weight.data = fused_w
        fused_conv.bias.data = fused_b
        return nn.Sequential(fused_conv, self.spike)

class SNN_Linear_BN(nn.Module):
    """
    Fused Linear + BatchNorm + Spike module.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features)
        self.spike = Multispike()

    def forward(self, x):
        # Flatten: (B, N, C_in) -> (B*N, C_in) for Linear layer
        B, N, C_in = x.shape
        x_lin = self.linear(x.reshape(B * N, C_in))
        x_bn = self.bn(x_lin)
        x_spiked = self.spike(x_bn)
        # Reshape back: (B, N, C_out)
        return x_spiked.reshape(B, N, -1)

    def fuse_module(self):
        """Returns a fused Sequential module."""
        if self.training:
            print("Warning: fuse_module() should only be called in .eval() mode.")
            return self
        
        lin_w = self.linear.weight
        lin_b = self.linear.bias
        if lin_b is None:
            lin_b = torch.zeros(self.linear.out_features, device=lin_w.device)

        bn_rm = self.bn.running_mean
        bn_rv = self.bn.running_var
        bn_eps = self.bn.eps
        bn_w = self.bn.weight
        bn_b = self.bn.bias

        scale = bn_w / torch.sqrt(bn_rv + bn_eps)
        fused_w = lin_w * scale.view(-1, 1)
        fused_b = (lin_b - bn_rm) * scale + bn_b

        fused_linear = nn.Linear(self.linear.in_features, self.linear.out_features, bias=True).to(lin_w.device)
        fused_linear.weight.data = fused_w
        fused_linear.bias.data = fused_b

        # Define a temporary container class to handle reshaping logic
        class FusedLinearBNSequential(nn.Module):
            def __init__(self, linear, spike):
                super().__init__()
                self.linear = linear
                self.spike = spike
            def forward(self, x):
                B, N, C_in = x.shape
                x = x.reshape(B * N, C_in)
                x = self.linear(x)
                x = self.spike(x)
                return x.reshape(B, N, -1)
                
        return FusedLinearBNSequential(fused_linear, self.spike)

# =============================================================================
# 6. SNN Transformer Components
# =============================================================================

class Optimized_SNN_Attention_BNC(nn.Module):
    """Spike-Driven Self-Attention (SDSA) Module."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.head_spike = Multispike()
        self.q_conv = SNN_Conv1d_BN(dim, dim, kernel_size=1, bias=False)
        self.k_conv = SNN_Conv1d_BN(dim, dim, kernel_size=1, bias=False)
        self.v_conv = SNN_Conv1d_BN(dim, dim, kernel_size=1, bias=False)
        self.attn_spike = Multispike()
        self.proj_conv = SNN_Conv1d_BN(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        x = self.head_spike(x) # Pre-Spike
        x_for_qkv = x.transpose(1, 2)
        
        q = self.q_conv(x_for_qkv) 
        k = self.k_conv(x_for_qkv) 
        v = self.v_conv(x_for_qkv) 
        
        q = q.transpose(1, 2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.transpose(1, 2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.transpose(1, 2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x_attn = k.transpose(-2, -1) @ v 
        x = (q @ x_attn) * self.scale    
        x = x.transpose(1, 2).reshape(B, N, C) 
        x = self.attn_spike(x)
        
        x_proj = x.transpose(1, 2) 
        x = self.proj_conv(x_proj) 
        return x.transpose(1, 2)

class Optimized_SNN_MLP_BNC(nn.Module):
    """Spiking Multi-Layer Perceptron (MLP) Block."""
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.head_spike = Multispike()
        self.mlp1 = SNN_Conv1d_BN(in_features, hidden_features, 1, bias=False)
        self.mlp2 = SNN_Conv1d_BN(hidden_features, in_features, 1, bias=False)
    
    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2)
        x = self.head_spike(x) 
        x = self.mlp1(x)
        x = self.mlp2(x)
        return x.transpose(1, 2)

class Optimized_SNN_TransformerBlock(nn.Module):
    """Standard Spiking Transformer Block."""
    def __init__(self, emb_size, num_heads):
        super().__init__()
        self.attention_layer = Optimized_SNN_Attention_BNC(emb_size, num_heads)
        self.MLP = Optimized_SNN_MLP_BNC(emb_size, emb_size * 2)

    def forward(self, x):
        x = x + self.attention_layer(x)
        x = x + self.MLP(x)
        return x

# =============================================================================
# 7. Model Architecture: SDBformer
# =============================================================================

class Optimized_SNN_DE_DBformer(nn.Module):
    """
    Spiking Dual-Branch Transformer (SDBformer) optimized for DE features.
    Inputs: (Batch, Channels, Bands)
    """
    def __init__(self, chn: int, num_bands: int,
                 num_direction_classes: int = 2, emb_size: int = 128, depth: int = 2, num_heads: int = 8):
        super().__init__()
        
        # --- 1. Dual-Branch SNN Embedding ---
        # "Frequential" Branch: Processes frequency bands (treated as temporal sequence)
        self.temporal_embedding = SNN_Linear_BN(chn, emb_size) # (B, Bands, Channels) -> (B, Bands, Embed)
        # "Spatial" Branch: Processes channels
        self.spatial_embedding = SNN_Linear_BN(num_bands, emb_size) # (B, Channels, Bands) -> (B, Channels, Embed)
        
        self.P = num_bands 
        self.C = chn       
        self.emb_size = emb_size
        
        self.pos_embedding_temporal = nn.Parameter(torch.randn(1, self.P, emb_size)) 
        self.pos_embedding_spatial = nn.Parameter(torch.randn(1, self.C, emb_size)) 
        
        # --- 2. Spiking Transformer Encoders ---
        self.temporal_transformer = nn.ModuleList([
            Optimized_SNN_TransformerBlock(emb_size, num_heads) for _ in range(depth)
        ])
        self.spatial_transformer = nn.ModuleList([
            Optimized_SNN_TransformerBlock(emb_size, num_heads) for _ in range(depth)
        ])

        # --- 3. Feature Fusion ---
        self.spatial_attn_pool = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            Multispike(),
            nn.Linear(emb_size, 1)
        )
        
        # --- 4. Classification Head ---
        self.label_classifier = nn.Sequential(
            nn.Linear(emb_size * 2, 64), 
            nn.LayerNorm(64),
            Multispike(),       
            nn.Linear(64, num_direction_classes)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d) and m.bias is not None:
             nn.init.constant_(m.bias, 0)
            
    def fuse_model_bn(self):
        """
        Recursively fuses all SNN_Conv1d_BN and SNN_Linear_BN modules for inference efficiency.
        """
        print("="*80)
        print("Fusing BatchNorm layers for inference/energy analysis...")
        if self.training:
            print("Warning: Fusion should be done in .eval() mode.")
            self.eval()
        
        def _recursive_fuse(module):
            for name, child in module.named_children():
                # Recursive call
                _recursive_fuse(child)
                
                # Check for fusable layers
                if isinstance(child, (SNN_Conv1d_BN, SNN_Linear_BN)):
                    print(f"  -> Fusing: {name}")
                    fused_sequential = child.fuse_module()
                    setattr(module, name, fused_sequential)
        
        _recursive_fuse(self)
        print("BatchNorm fusion complete.")
        print("="*80)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, Channels, Bands)
        
        # --- 1. Dual-Branch Embedding ---
        
        # Spatial Branch
        x_embed_spatial = self.spatial_embedding(x) + self.pos_embedding_spatial
        x_spatial = x_embed_spatial
        for block in self.spatial_transformer:
            x_spatial = block(x_spatial)
        
        # Frequential Branch (Permute to: B, Bands, Channels)
        x_permuted = x.permute(0, 2, 1) 
        x_embed_temporal = self.temporal_embedding(x_permuted) + self.pos_embedding_temporal
        x_temporal = x_embed_temporal
        for block in self.temporal_transformer:
            x_temporal = block(x_temporal)

        # --- 2. Feature Fusion ---
        x_t = x_temporal.mean(dim=1) # Global Average Pooling
        
        attn_scores = self.spatial_attn_pool(x_spatial) 
        attn_weights = torch.softmax(attn_scores, dim=1) 
        x_s = torch.sum(attn_weights * x_spatial, dim=1) # Attention-weighted Sum
        
        x_fused = torch.cat([x_t, x_s], dim=-1) 

        # --- 3. Classification ---
        label_output = self.label_classifier(x_fused)
        return label_output
    
# =============================================================================
# 8. Energy Analysis Tools
# =============================================================================

class ComprehensiveSNNEnergyCalculator:
    """
    Calculates energy consumption (FLOPs vs. SOPs) for SNNs.
    """
    
    def __init__(self, model, batch_size=32, device='cuda'):
        self.model = model.to(device)
        self.batch_size = batch_size
        self.device = device
        self.layer_analysis = []
        self.activations = {}
        self.hooks = []
        
    def register_hooks(self):
        """Register forward hooks to capture input/output tensors."""
        def hook_fn(module, input, output):
            module_id = id(module)
            # Store original tensors for ID comparison later
            input_tensor = input[0] if input and input[0] is not None else None
            output_tensor = output if output is not None else None

            self.activations[module_id] = {
                'module': module,
                'module_type': module.__class__.__name__,
                'input_tensor_orig': input_tensor,   
                'output_tensor_orig': output_tensor, 
            }
        
        self.remove_hooks()
        
        target_modules = (
            nn.Conv1d, nn.Linear, 
            nn.AvgPool1d, nn.MaxPool1d, nn.AdaptiveAvgPool1d,
            nn.BatchNorm1d, nn.LayerNorm, 
            nn.ReLU, nn.Sigmoid, nn.Tanh, 
            Multispike,
            SNN_Conv1d_BN, SNN_Linear_BN 
        )

        for name, module in self.model.named_modules():
            if isinstance(module, target_modules):
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _get_kernel_and_stride(self, module):
        kernel_size = getattr(module, 'kernel_size', 1)
        if isinstance(kernel_size, tuple): kernel_size = kernel_size[0]
        stride = getattr(module, 'stride', 1)
        if isinstance(stride, tuple): stride = stride[0] if stride else 1
        elif stride is None: stride = 1
        return kernel_size, stride
    
    def calculate_spiking_rate(self, tensor):
        if tensor is None: return 0.0, 0, 0
        total_elements = tensor.numel()
        spiking_sum = torch.count_nonzero(tensor).item()
        spiking_rate = spiking_sum / total_elements if total_elements > 0 else 0
        return spiking_rate, spiking_sum, total_elements
        
    def calculate_conv1d_energy(self, module, input_tensor, is_spiking):
        if input_tensor is None: return 0, 0, 0
        B, C_in, L = input_tensor.shape
        C_out = module.out_channels
        kernel_size = module.kernel_size[0]
        stride = module.stride[0]
        padding = module.padding[0]
        dilation = module.dilation[0]
        groups = module.groups
        L_out = (L + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        
        if is_spiking:
            operations_per_output = kernel_size * (C_in // groups)
            total_operations = B * C_out * L_out * operations_per_output
            energy_coefficient = 0.9 # AC (0.9 pJ)
        else:
            operations_per_output = 2 * kernel_size * (C_in // groups)
            total_operations = B * C_out * L_out * operations_per_output
            energy_coefficient = 4.6 # MAC (4.6 pJ)
            
        spiking_rate, _, _ = self.calculate_spiking_rate(input_tensor)
        if not is_spiking: spiking_rate = 1.0
        energy = total_operations * energy_coefficient * spiking_rate
        return energy, total_operations, spiking_rate
    
    def calculate_linear_energy(self, module, input_tensor, is_spiking):
        if input_tensor is None: return 0, 0, 0
        if input_tensor.dim() == 3:
            B, N, C_in = input_tensor.shape
            input_flat = input_tensor.reshape(B * N, C_in)
            batch_dim = B * N
        else:
            input_flat = input_tensor
            batch_dim = input_flat.shape[0]
        in_features = module.in_features
        out_features = module.out_features
        
        if is_spiking:
            operations_per_output = in_features
            total_operations = batch_dim * out_features * operations_per_output
            energy_coefficient = 0.9 # AC
        else:
            operations_per_output = 2 * in_features
            total_operations = batch_dim * out_features * operations_per_output
            energy_coefficient = 4.6 # MAC
            
        spiking_rate, _, _ = self.calculate_spiking_rate(input_tensor)
        if not is_spiking: spiking_rate = 1.0
        energy = total_operations * energy_coefficient * spiking_rate
        return energy, total_operations, spiking_rate
        
    def calculate_pooling_energy(self, module, input_tensor):
        if input_tensor is None: return 0, 0
        if input_tensor.dim() == 3:
             B, C, L = input_tensor.shape
        elif input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(1)
            B, C, L = input_tensor.shape
        else:
            return 0, 0
            
        energy_coefficient = 0.5
        total_operations = 0
        
        if isinstance(module, (nn.AvgPool1d, nn.MaxPool1d)):
            kernel_size, stride = self._get_kernel_and_stride(module)
            padding = getattr(module, 'padding', 0)
            if isinstance(padding, tuple): padding = padding[0]
            L_out = (L + 2 * padding - kernel_size) // stride + 1
            operations_per_output = kernel_size - 1
            total_operations = B * C * L_out * operations_per_output
        elif isinstance(module, nn.AdaptiveAvgPool1d):
            if module.output_size == 1:
                operations_per_output = L - 1
                total_operations = B * C * operations_per_output

        energy = total_operations * energy_coefficient
        return energy, total_operations
    
    def calculate_norm_energy(self, module, input_tensor):
        if input_tensor is None: return 0, 0
        total_elements = input_tensor.numel()
        if isinstance(module, nn.BatchNorm1d):
            total_operations = 2 * total_elements
            energy_coefficient = 0.3
        elif isinstance(module, nn.LayerNorm):
            total_operations = 5 * total_elements
            energy_coefficient = 0.4
        else:
            return 0, 0
        energy = total_operations * energy_coefficient
        return energy, total_operations
    
    def calculate_activation_energy(self, module, input_tensor):
        if input_tensor is None: return 0, 0
        total_elements = input_tensor.numel()
        if isinstance(module, (nn.ReLU, Multispike)):
            total_operations = total_elements
            energy_coefficient = 0.2
        elif isinstance(module, (nn.Sigmoid, nn.Tanh)):
            total_operations = 5 * total_elements
            energy_coefficient = 0.5
        else:
            return 0, 0
        energy = total_operations * energy_coefficient
        return energy, total_operations
        
    def analyze_layer_energy(self, module, input_tensor, is_spiking):
        input_tensor_cpu = input_tensor.detach().cpu()
        module_type = module.__class__.__name__
        energy, operations, spiking_rate = 0, 0, 0.0
        
        if isinstance(module, nn.Conv1d):
            energy, operations, spiking_rate = self.calculate_conv1d_energy(module, input_tensor_cpu, is_spiking)
        elif isinstance(module, nn.Linear):
            energy, operations, spiking_rate = self.calculate_linear_energy(module, input_tensor_cpu, is_spiking)
        elif isinstance(module, (nn.AvgPool1d, nn.MaxPool1d, nn.AdaptiveAvgPool1d)):
            energy, operations = self.calculate_pooling_energy(module, input_tensor_cpu)
            spiking_rate = 1.0
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            energy, operations = self.calculate_norm_energy(module, input_tensor_cpu)
            spiking_rate = 1.0
        elif isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh, Multispike)):
            energy, operations = self.calculate_activation_energy(module, input_tensor_cpu)
            spiking_rate = 1.0
        else:
            return None
        
        return {
            'module_name': f"{module_type}_{id(module)}",
            'module_type': module_type,
            'input_shape': tuple(input_tensor_cpu.shape) if input_tensor_cpu is not None else None,
            'is_spiking': is_spiking,
            'total_operations': float(operations),
            'spiking_rate': float(spiking_rate),
            'layer_energy_pJ': float(energy),
            'energy_per_sample_pJ': float(energy / self.batch_size)
        }
    
    def run_comprehensive_analysis(self, input_tensor):
        """Runs the complete energy analysis pipeline."""
        print("Starting comprehensive SNN energy analysis...")
        self.register_hooks()
        
        print("Step 1/3: Forward pass to capture computation graph...")
        self.activations.clear(); self.layer_analysis.clear()
        
        input_tensor_device = input_tensor.to(self.device)
        with torch.no_grad():
            _ = self.model(input_tensor_device)
        print("Graph captured.")

        print("Step 2/3: Building Spiking Tensor Database...")
        spiking_tensor_ids = set()
        target_spike_producers = ('Multispike', 'SNN_Conv1d_BN', 'SNN_Linear_BN')
        
        for module_id, activation_info in self.activations.items():
            if activation_info['module_type'] in target_spike_producers:
                output_tensor = activation_info['output_tensor_orig'] 
                if output_tensor is not None:
                    spiking_tensor_ids.add(id(output_tensor))
        print(f"Found {len(spiking_tensor_ids)} spiking activation sources.")

        print("Step 3/3: Analyzing layer energy...")
        total_energy, analyzed_layers = 0, 0
        total_flops_batch = 0.0 
        total_sops_batch = 0.0  
        
        for module_id, activation_info in self.activations.items():
            module = activation_info['module']
            input_tensor_orig = activation_info['input_tensor_orig']
            if input_tensor_orig is None: continue
            
            is_spiking = id(input_tensor_orig) in spiking_tensor_ids
            
            # Skip container layers, analyze their children (Conv/Linear)
            if isinstance(module, (SNN_Conv1d_BN, SNN_Linear_BN)):
                continue 
            
            layer_info = self.analyze_layer_energy(module, input_tensor_orig, is_spiking)
            
            if layer_info is not None:
                self.layer_analysis.append(layer_info)
                total_energy += layer_info['layer_energy_pJ']
                if layer_info['is_spiking']:
                    total_sops_batch += layer_info['total_operations']
                else:
                    total_flops_batch += layer_info['total_operations']
                analyzed_layers += 1
        
        energy_per_sample_pJ = total_energy / self.batch_size
        flops_per_sample_G = (total_flops_batch / self.batch_size) / 1e9
        sops_per_sample_G = (total_sops_batch / self.batch_size) / 1e9
        energy_per_sample_mJ = energy_per_sample_pJ / 1e9
        
        self.remove_hooks()
        print(f"\nAnalysis Complete!")
        
        # Summary printing logic...
        try:
            df_detailed = pd.DataFrame(self.layer_analysis)
            if df_detailed.empty:
                print("No layers analyzed.")
                return energy_per_sample_pJ, flops_per_sample_G, sops_per_sample_G
                
            summary_by_type = df_detailed.groupby('module_type').agg(
                layer_count=('module_name', 'count'),
                total_energy_pJ=('layer_energy_pJ', 'sum'),
                total_operations=('total_operations', 'sum')
            )
            total_energy_all = df_detailed['layer_energy_pJ'].sum()
            summary_by_type['energy_percentage (%)'] = (
                summary_by_type['total_energy_pJ'] / (total_energy_all + 1e-9) * 100
            )
            summary_by_type = summary_by_type.sort_values(by='energy_percentage (%)', ascending=False)
            pd.set_option('display.width', 1000)
            print("\nEnergy Summary by Module Type:")
            print(summary_by_type.to_string(float_format="%.2f"))
        except Exception as e:
            print(f"Error printing report: {e}")

        print("\n--- Final Summary ---")
        print(f"Total Layers: {analyzed_layers}")
        print(f"FLOPs (G) / sample: {flops_per_sample_G:.4f} G")
        print(f"SOPs  (G) / sample: {sops_per_sample_G:.4f} G")
        print(f"Energy (mJ) / sample: {energy_per_sample_mJ:.6f} mJ")
        
        return energy_per_sample_pJ, flops_per_sample_G, sops_per_sample_G
    
    def convert_to_serializable(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)): return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: self.convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list): return [self.convert_to_serializable(item) for item in obj]
        if isinstance(obj, tuple): return tuple(self.convert_to_serializable(item) for item in obj)
        return obj

    def generate_detailed_report(self, save_path):
        if not self.layer_analysis:
            print("No analysis data found.")
            return
        os.makedirs(save_path, exist_ok=True)
        df_detailed = pd.DataFrame(self.layer_analysis)
        for col in df_detailed.select_dtypes(include=[np.number]).columns:
            df_detailed[col] = df_detailed[col].apply(lambda x: float(x) if pd.notnull(x) else x)
        df_detailed.to_excel(os.path.join(save_path, "detailed_energy_analysis.xlsx"), index=False)
        
        # Save JSON summary
        total_energy = float(df_detailed['layer_energy_pJ'].sum())
        total_operations = float(df_detailed['total_operations'].sum())
        summary = {
            "total_energy_pJ": total_energy,
            "energy_per_sample_mJ": total_energy / self.batch_size / 1e9,
            "total_operations": total_operations,
            "operations_per_sample": total_operations / self.batch_size,
        }
        with open(os.path.join(save_path, "energy_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nDetailed report saved to: {save_path}")

# =============================================================================
# 9. Trainer & Training Loop
# =============================================================================

class SimpleTrainer:
    """Standard Trainer for Within-Subject experiments."""
    def __init__(self, model, optimizer=None, lr_scheduler=None, config=None):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.label_criterion = nn.CrossEntropyLoss()
        self.metrics = self._init_metrics()

    def _init_metrics(self):
        return {'loss': AvgMeter("Loss"), 'acc': AvgMeter("Acc")}

    def run_epoch(self, dataloader, mode='train', epoch=0, num_epochs=1):
        self.model.train() if mode == 'train' else self.model.eval()
        for key in self.metrics: self.metrics[key].reset()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [{mode.capitalize()}]")
        
        with torch.set_grad_enabled(mode == 'train'):
            for batch in pbar:
                eeg, direction_labels = [d.to(self.config.device) for d in batch]
                
                if mode == 'train':
                    self.optimizer.zero_grad()
                    label_output = self.model(eeg) 
                    loss = self.label_criterion(label_output, direction_labels)
                    loss.backward()
                    self.optimizer.step()
                else:
                    label_output = self.model(eeg)
                    loss = self.label_criterion(label_output, direction_labels)
                
                self.metrics['loss'].update(loss.item(), eeg.size(0))
                acc = (label_output.argmax(1) == direction_labels).float().mean().item()
                self.metrics['acc'].update(acc, eeg.size(0))
                pbar.set_postfix({'Loss': f"{self.metrics['loss'].avg:.4f}", 'Acc': f"{self.metrics['acc'].avg:.2%}"})
        return self.metrics

    def train_epoch(self, dataloader, epoch, num_epochs):
        return self.run_epoch(dataloader, mode='train', epoch=epoch, num_epochs=num_epochs)
    
    def test_epoch(self, dataloader, epoch, num_epochs):
        metrics = self.run_epoch(dataloader, mode='test', epoch=epoch, num_epochs=num_epochs)
        if self.lr_scheduler:
            self.lr_scheduler.step(metrics['acc'].avg)
        return metrics

def run_training_fold(train_loader, test_loader, config, fold_save_dir):
    """
    Executes training for a single CV fold.
    Returns best accuracy, history history, and path to the best model.
    """
    model = Optimized_SNN_DE_DBformer( 
        chn=config.channel_size,
        num_bands=config.num_bands,
        num_direction_classes=2,
        emb_size=config.emb_size,
        depth=config.depth,
        num_heads=config.num_heads
    ).to(config.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=config.patience, factor=config.factor)
    
    trainer = SimpleTrainer(model, optimizer, lr_scheduler, config)
    
    best_test_acc = 0
    training_history = []
    best_model_path = os.path.join(fold_save_dir, "best_model.pt") 
    
    print(f"--- Starting Training ({config.epochs} Epochs) ---")
    for epoch in range(config.epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch=epoch, num_epochs=config.epochs)
        test_metrics = trainer.test_epoch(test_loader, epoch=epoch, num_epochs=config.epochs)
        
        test_acc = test_metrics['acc'].avg
        
        epoch_info = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'].avg,
            'train_acc': train_metrics['acc'].avg,
            'test_loss': test_metrics['loss'].avg,
            'test_acc': test_acc,
            'learning_rate': get_lr(optimizer)
        }
        training_history.append(epoch_info)
        
        if (epoch + 1) % 10 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch + 1:03d}/{config.epochs} | Train Acc: {epoch_info['train_acc']:.2%} | Test Acc: {epoch_info['test_acc']:.2%}")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), best_model_path) 

    print(f"Fold Complete. Best Test Acc: {best_test_acc:.4f}\n")
    return best_test_acc, training_history, best_model_path 

def save_history_plot(history, save_dir, title_prefix):
    """Saves loss and accuracy curves."""
    if not history: 
        warnings.warn("No history to plot.")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    epochs = [h['epoch'] for h in history]
    ax1.plot(epochs, [h['train_loss'] for h in history], 'o-', label='Train Loss')
    ax1.plot(epochs, [h['test_loss'] for h in history], 'o-', label='Test Loss')
    ax1.set_title('Loss Curves'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True)
    ax2.plot(epochs, [h['train_acc'] for h in history], 'o-', label='Train Acc')
    ax2.plot(epochs, [h['test_acc'] for h in history], 'o-', label='Test Acc')
    ax2.axhline(0.5, color='r', linestyle='--', label='Chance Level (0.5)')
    ax2.set_title('Accuracy Curves'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.set_ylim(0, 1.05); ax2.legend(); ax2.grid(True)
    fig.suptitle(f'Optimized SNN-DE on {title_prefix}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close(fig)

# =============================================================================
# 10. Configuration
# =============================================================================

class TrainingConfig:
    """Hyperparameters and Paths."""
    def __init__(self):
        self.dataset_name = "DTU-DE" 
        # Update this path for GitHub
        self.data_root = "/share/home/yuan/LY/SNN_DBformer/DE_DTU/DE_1s" 
        self.save_dir = './Results/WS_5FoldCV_OptimizedSNN_DE_CSP' 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 128
        self.num_workers = 4 
        self.learning_rate = 1e-4
        self.epochs = 60
        self.weight_decay = 1e-3
        self.patience = 10 
        self.factor = 0.5  
        
        # Model Params
        self.channel_size = 66
        self.num_bands = 5 
        self.emb_size = 48
        self.depth = 3
        self.num_heads = 8  
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# =============================================================================
# 14. Helper Function: Post-Training Energy Analysis
# =============================================================================

def run_energy_analysis_for_model(config, subject_id, fold_num, model_path):
    """
    Loads a specific saved model and runs the energy analysis pipeline.
    This requires reloading the data and recreating the CSP transformation 
    to ensure input data distribution matches the training phase.
    """
    print("\n" + "#"*80)
    print(f"Starting Post-Training Energy Analysis")
    print(f"Target: Subject {subject_id} | Fold {fold_num}")
    print(f"Model Path: {model_path}")
    print("#"*80)

    try:
        # 1. Reload Data for the target subject
        print("Reloading data for analysis...")
        subject_dataset = DTU_AAD_Dataset_WS(config.data_root, subject_id)
        X_all = subject_dataset.eeg_features.numpy() 
        y_all = subject_dataset.direction_label.numpy()
        
        # 2. Re-create the specific Fold split
        # We need the exact test set used in Fold 1
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        target_test_tensor = None
        
        # Iterate to find the specific fold indices
        for current_fold_idx, (train_indices, test_indices) in enumerate(kf.split(X_all)):
            if (current_fold_idx + 1) == fold_num:
                # Recover the data splits
                X_train_np, y_train_np = X_all[train_indices], y_all[train_indices]
                X_test_np, _ = X_all[test_indices], y_all[test_indices]
                
                # Re-fit CSP (Strictly speaking, we should load saved CSP, 
                # but fitting on train indices again produces identical results for deterministic algorithms)
                csp = CSP(
                    n_components=config.channel_size, 
                    reg=None, log=None, cov_est='concat', 
                    transform_into='csp_space', norm_trace=True
                )
                csp.fit(X_train_np, y_train_np)
                
                # Transform test data
                X_test_csp = csp.transform(X_test_np)
                target_test_tensor = torch.tensor(X_test_csp, dtype=torch.float32)
                break
        
        if target_test_tensor is None:
            raise ValueError(f"Could not split data for Fold {fold_num}")

        # 3. Instantiate Model
        model = Optimized_SNN_DE_DBformer(
            chn=config.channel_size,
            num_bands=config.num_bands, 
            num_direction_classes=2,
            emb_size=config.emb_size,
            depth=config.depth,
            num_heads=config.num_heads
        ).to(config.device)

        # 4. Load Weights
        print("Loading model weights...")
        model.load_state_dict(torch.load(model_path, map_location=config.device))

        # 5. Fuse BatchNorm
        model.fuse_model_bn()

        # 6. Calculate Parameters
        total_params = count_parameters(model)
        print(f"Model Params: {total_params / 1e6:.2f} M")

        # 7. Run Energy Calculator
        # Take a single batch-sized slice or the whole test set (if memory allows)
        # Using batch_size from config to simulate real inference
        input_sample = target_test_tensor[:config.batch_size].to(config.device)
        print(f"Input shape for analysis: {input_sample.shape}")

        calculator = ComprehensiveSNNEnergyCalculator(
            model=model,
            batch_size=config.batch_size,
            device=config.device
        )
        
        energy_pJ, flops_G, sops_G = calculator.run_comprehensive_analysis(input_sample)
        energy_mJ = energy_pJ / 1e9

        print(f"\n[Final Analysis Result] S{subject_id}-F{fold_num}")
        print(f"Energy: {energy_mJ:.6f} mJ")
        print(f"FLOPs:  {flops_G:.4f} G")
        print(f"SOPs:   {sops_G:.4f} G")
        
        # Save Report to the model directory
        report_dir = os.path.dirname(model_path)
        calculator.generate_detailed_report(os.path.join(report_dir, "energy_report"))
        print("-" * 40)

    except Exception as e:
        print(f"Error during energy analysis: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# 15. Main Execution
# =============================================================================

def main():
    """
    Main pipeline: 
    1. Run 5-Fold Cross Validation for ALL subjects.
    2. Collect and save statistical results (Mean, Std per subject & Global).
    3. After all training is done, select one representative model for Energy Analysis.
    """
    try:
        config = TrainingConfig()
        main_results_dir = f"{config.save_dir}_{config.timestamp}"
        os.makedirs(main_results_dir, exist_ok=True)
        
        print("="*80)
        print(f"Task: Optimized SNN-DE + CSP | 5-Fold CV | Statistical Analysis")
        print(f"Device: {config.device}")
        print(f"Output: {main_results_dir}")
        print("="*80)
        
        # ---------------------------------------------------------------------
        # Phase 1: Training Loop & Statistics Collection
        # ---------------------------------------------------------------------
        
        # [Modify here] Uncomment the next line to run all subjects
        all_subject_ids = [str(i) for i in range(1, 17)] 
        # all_subject_ids = ["1", "2"] # Example: Running S1 and S2 for testing
        
        # List to store summary data for all subjects
        global_stats = [] 
        
        # Store path of a representative model (S1-F1) for later energy analysis
        representative_model_path = None
        
        for subject_id in all_subject_ids:
            print(f"\n{'='*30} Processing Subject: S{subject_id} {'='*30}")
            
            subject_save_dir = os.path.join(main_results_dir, f"subject_{subject_id}")
            os.makedirs(subject_save_dir, exist_ok=True)
            
            try:
                subject_dataset = DTU_AAD_Dataset_WS(config.data_root, subject_id)
                X_all = subject_dataset.eeg_features.numpy() 
                y_all = subject_dataset.direction_label.numpy()
            except FileNotFoundError as e:
                print(f"Skipping S{subject_id}: {e}")
                continue
            
            N_SPLITS = 5
            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
            
            # Store accuracies for the current subject's 5 folds
            subject_fold_accs = [] 
            
            for fold_idx, (train_indices, test_indices) in enumerate(kf.split(X_all)):
                fold_num = fold_idx + 1
                print(f"\n--- Subject S{subject_id} | Fold {fold_num}/{N_SPLITS} ---")
                
                fold_save_dir = os.path.join(subject_save_dir, f"fold_{fold_num}")
                os.makedirs(fold_save_dir, exist_ok=True)

                # 1. CSP Preprocessing
                X_train_np, y_train_np = X_all[train_indices], y_all[train_indices]
                X_test_np, y_test_np = X_all[test_indices], y_all[test_indices]
                
                csp = CSP(n_components=config.channel_size, reg=None, log=None, 
                          cov_est='concat', transform_into='csp_space', norm_trace=True)
                
                X_train_csp = csp.fit_transform(X_train_np, y_train_np)
                X_test_csp = csp.transform(X_test_np)
                
                # 2. Prepare Tensors & Loaders
                train_dataset_csp = TensorDataset(
                    torch.tensor(X_train_csp, dtype=torch.float32),
                    torch.tensor(y_train_np, dtype=torch.long)
                )
                test_dataset_csp = TensorDataset(
                    torch.tensor(X_test_csp, dtype=torch.float32),
                    torch.tensor(y_test_np, dtype=torch.long)
                )
                
                train_loader = DataLoader(train_dataset_csp, config.batch_size, shuffle=True, 
                                          num_workers=config.num_workers, pin_memory=True)
                test_loader = DataLoader(test_dataset_csp, config.batch_size, shuffle=False, 
                                         num_workers=config.num_workers, pin_memory=True)
                
                # 3. Train
                best_acc, history, best_model_path = run_training_fold(
                    train_loader, test_loader, config, fold_save_dir
                )
                
                # [Stat Collection] Append best accuracy of this fold
                subject_fold_accs.append(best_acc) 

                # Save history
                pd.DataFrame(history).to_csv(os.path.join(fold_save_dir, "training_history.csv"), index=False)
                save_history_plot(history, fold_save_dir, f"{config.dataset_name}_S{subject_id}_F{fold_num}")

                # Capture representative model
                if subject_id == "1" and fold_num == 1:
                    representative_model_path = best_model_path

            # --- End of Subject Loop: Calculate Subject Statistics ---
            if subject_fold_accs:
                mean_acc = np.mean(subject_fold_accs)
                std_acc = np.std(subject_fold_accs)
                print(f"S{subject_id} Completed. Mean Acc: {mean_acc:.4f} | Std: {std_acc:.4f}")
                
                # Add to global stats list
                stat_entry = {
                    'Subject': f"S{subject_id}",
                    'Mean_Acc': mean_acc,
                    'Std_Dev': std_acc
                }
                # Add individual fold results (e.g., Fold_1, Fold_2...)
                for i, acc in enumerate(subject_fold_accs):
                    stat_entry[f'Fold_{i+1}'] = acc
                
                global_stats.append(stat_entry)

        # ---------------------------------------------------------------------
        # Phase 2: Global Statistical Report
        # ---------------------------------------------------------------------
        print("\n" + "="*80)
        print("Generating Final Statistical Report...")
        print("="*80)
        
        if global_stats:
            # Convert to DataFrame
            df_stats = pd.DataFrame(global_stats)
            
            # Reorder columns: Subject, Mean, Std, Fold_1, Fold_2...
            cols = ['Subject', 'Mean_Acc', 'Std_Dev'] + [c for c in df_stats.columns if 'Fold_' in c]
            df_stats = df_stats[cols]
            
            # Calculate Grand Average (Average of all subjects)
            grand_mean = df_stats['Mean_Acc'].mean()
            grand_std = df_stats['Mean_Acc'].std() # Std across subjects
            
            # Append Grand Average row
            grand_avg_row = {k: '' for k in df_stats.columns}
            grand_avg_row['Subject'] = 'Grand Average'
            grand_avg_row['Mean_Acc'] = grand_mean
            grand_avg_row['Std_Dev'] = grand_std
            
            # Append using pd.concat
            df_final = pd.concat([df_stats, pd.DataFrame([grand_avg_row])], ignore_index=True)
            
            # Save to CSV
            csv_path = os.path.join(main_results_dir, "final_performance_summary.csv")
            df_final.to_csv(csv_path, index=False)
            
            # Print to Console
            pd.set_option('display.float_format', '{:.4f}'.format)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print(df_final)
            print(f"\nReport saved to: {csv_path}")
        else:
            print("No training data collected.")

        # ---------------------------------------------------------------------
        # Phase 3: Post-Training Energy Analysis (unchanged)
        # ---------------------------------------------------------------------
        if representative_model_path and os.path.exists(representative_model_path):
            run_energy_analysis_for_model(
                config=config,
                subject_id="1",
                fold_num=1,
                model_path=representative_model_path
            )
        else:
            print("Warning: Representative model (S1-F1) not found. Skipping energy analysis.")

        print("\nAll tasks finished.")
        
    except Exception as e:
        print(f"\nCritical Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()