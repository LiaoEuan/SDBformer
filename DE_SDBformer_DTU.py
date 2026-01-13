
# =============================================================================
# 1. 导入库
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
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from timm.models.layers import trunc_normal_
import matplotlib.pyplot as plt

from mne.decoding import CSP 
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

# =============================================================================
# 3. 工具类
# =============================================================================
class AvgMeter:
    """计算并存储指标的平均值和当前值"""
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
    """从优化器中获取当前学习率"""
    for param_group in optimizer.param_groups:
        return param_group["lr"]

# =============================================================================
# 4. [修改] DTU DE 特征数据集加载器
# =============================================================================
class DTU_AAD_Dataset_WS(Dataset):
    """DTU DE 特征数据集加载器 (Within-Subject / 被试内)"""
    def __init__(self, root, subject_id):
        self.root = root
        self.subject_id = str(subject_id)
        
        # [修改] 假设文件名与您上一个脚本中保存的一致
        file_name = f"S{self.subject_id}_DE_Features_1s.npz" 
        file_path = os.path.join(root, file_name)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"警告: 找不到DE特征文件: {file_path}")
            
        print(f"  正在加载 {file_name} ...")
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # [修改] 加载 'DE' 和 'labels' 键
            self.eeg_features = torch.tensor(data['DE'], dtype=torch.float32) # (N_samples, 66, 5)
            self.direction_label = torch.tensor(data['labels'], dtype=torch.long) # (N_samples,)
            
        except Exception as e:
            raise IOError(f"加载或解析 {file_path} 时出错: {e}")

        if len(self.eeg_features) != len(self.direction_label):
            raise ValueError("DE 特征和标签的样本数不匹配!")
            
    def __len__(self):
        return len(self.eeg_features)
    
    def __getitem__(self, idx):
        # [修改] 返回 DE 特征
        return self.eeg_features[idx], self.direction_label[idx]

# =============================================================================
# 5. SNN 核心组件
# =============================================================================
class Quant(torch.autograd.Function):
    """SNN 替代梯度函数"""
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
    """多脉冲神经元激活模块"""
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
# 6. [优化] SNN 融合模块 (Conv/Linear + BN)
# =============================================================================

class SNN_Conv1d_BN(nn.Module):
    """(修正版) 卷积+BatchNorm融合层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.spike = Multispike()
        
    def forward(self, x):
        return self.spike(self.bn(self.conv(x)))
        
    def fuse_module(self):
        """返回一个 *已融合* 的新模块 (nn.Sequential)"""
        if self.training:
            print("警告: 仅应在 .eval() 模式下调用 fuse_module()")
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
        fused_conv = nn.Conv1d(self.conv.in_channels,
                                self.conv.out_channels,
                                self.conv.kernel_size, 
                                self.conv.stride,
                                self.conv.padding, 
                                groups=self.conv.groups, 
                               bias=True
        ).to(fused_w.device)
        fused_conv.weight.data = fused_w
        fused_conv.bias.data = fused_b
        return nn.Sequential(fused_conv, self.spike)

class SNN_Linear_BN(nn.Module):
    """(新增) 线性+BatchNorm融合层"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features)
        self.spike = Multispike()

    def forward(self, x):
        # (B, N, C_in) -> (B*N, C_in)
        B, N, C_in = x.shape
        x_lin = self.linear(x.reshape(B * N, C_in))
        
        # BN1d 期望 (B, C) or (B, C, L)
        # 我们有 (B*N, C_out), 这是正确的
        x_bn = self.bn(x_lin)
        
        # Spike
        x_spiked = self.spike(x_bn)
        
        # (B*N, C_out) -> (B, N, C_out)
        return x_spiked.reshape(B, N, -1)

    def fuse_module(self):
        """返回一个 *已融合* 的新模块 (nn.Sequential)"""
        if self.training:
            print("警告: 仅应在 .eval() 模式下调用 fuse_module()")
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

        # [修改] 返回的 Sequential 必须能处理 (B, N, C)
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
# 8. [优化] SNN 优化的 Transformer 模块
# =============================================================================

class Optimized_SNN_Attention_BNC(nn.Module):
    """(来自用户) 优化的注意力模块"""
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
        
        q = self.q_conv(x_for_qkv) # Conv->BN->Spike
        k = self.k_conv(x_for_qkv) # Conv->BN->Spike
        v = self.v_conv(x_for_qkv) # Conv->BN->Spike
        
        q = q.transpose(1, 2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.transpose(1, 2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.transpose(1, 2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x_attn = k.transpose(-2, -1) @ v 
        x = (q @ x_attn) * self.scale    
        x = x.transpose(1, 2).reshape(B, N, C) 
        x = self.attn_spike(x)
        
        x_proj = x.transpose(1, 2) 
        x = self.proj_conv(x_proj) # Conv->BN->Spike
        return x.transpose(1, 2)

class Optimized_SNN_MLP_BNC(nn.Module):
    """(新增, 补全) 优化的MLP块"""
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.head_spike = Multispike() # 1. Pre-Spike
        self.mlp1 = SNN_Conv1d_BN(in_features, hidden_features, 1, bias=False) # 2. Conv+BN+Spike
        self.mlp2 = SNN_Conv1d_BN(hidden_features, in_features, 1, bias=False) # 3. Conv+BN+Spike
    
    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2)
        x = self.head_spike(x) # Pre-Spike
        x = self.mlp1(x)
        x = self.mlp2(x)
        return x.transpose(1, 2)

class Optimized_SNN_TransformerBlock(nn.Module):
    """(来自用户) 优化的Transformer块"""
    def __init__(self, emb_size, num_heads):
        super().__init__()
        self.attention_layer = Optimized_SNN_Attention_BNC(emb_size, num_heads)
        self.MLP = Optimized_SNN_MLP_BNC(emb_size, emb_size * 2)

    def forward(self, x):
        x = x + self.attention_layer(x)
        x = x + self.MLP(x)
        return x

# =============================================================================
# 9. [修改] 优化的 SNN DE (DE特征) 基础模型
# =============================================================================

class Optimized_SNN_DE_DBformer(nn.Module):
    """
    [修改] 优化后的 SNN DBformer
    - 适配 (B, 66, 5) DE 特征输入
    - 移除了 Stem
    - 使用 SNN_Linear_BN 作为嵌入层
    """
    def __init__(self, chn: int, num_bands: int,
                 num_direction_classes: int = 2, emb_size: int = 128, depth: int = 2, num_heads: int = 8):
        super().__init__()
        
        # --- 1. [修改] SNN 线性嵌入 ---
        # "Temporal" 分支现在处理 "Spectral" (频段)
        self.temporal_embedding = SNN_Linear_BN(chn, emb_size) # (B, 5, 66) -> (B, 5, 128)
        # "Spatial" 分支现在处理 "Spatial" (通道)
        self.spatial_embedding = SNN_Linear_BN(num_bands, emb_size) # (B, 66, 5) -> (B, 66, 128)
        
        self.P = num_bands # "Temporal" 维度现在是 5
        self.C = chn       # "Spatial" 维度是 66
        self.emb_size = emb_size
        
        self.pos_embedding_temporal = nn.Parameter(torch.randn(1, self.P, emb_size)) # (1, 5, 128)
        self.pos_embedding_spatial = nn.Parameter(torch.randn(1, self.C, emb_size))  # (1, 66, 128)
        
        # --- 2. 优化的SNN Transformer ---
        self.temporal_transformer = nn.ModuleList([
            Optimized_SNN_TransformerBlock(emb_size, num_heads) for _ in range(depth)
        ])
        self.spatial_transformer = nn.ModuleList([
            Optimized_SNN_TransformerBlock(emb_size, num_heads) for _ in range(depth)
        ])

        # --- 3. 优化的融合模块 ---
        self.spatial_attn_pool = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            Multispike(),
            nn.Linear(emb_size, 1)
        )
        
        # --- 4. 优化的分类器头 ---
        self.label_classifier = nn.Sequential(
            nn.Linear(emb_size * 2, 64), 
            nn.LayerNorm(64), # [优化] 使用 LayerNorm 替换 BN
            Multispike(),       
            nn.Linear(64, num_direction_classes)
        )
        
        self.apply(self._init_weights) # 应用权重初始化

    def _init_weights(self, m: nn.Module):
        """权重初始化 (BN除外)"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d) and m.bias is not None:
             nn.init.constant_(m.bias, 0)
            
    def fuse_model_bn(self):
        """
        (新增) 递归地融合模型中所有 SNN_Conv1d_BN 和 SNN_Linear_BN 模块
        """
        print("="*80)
        print("开始融合 BatchNorm (BN) 层 (用于推理)...")
        if self.training:
            print("警告: 融合应在 .eval() 模式下进行。")
            self.eval()
        
        def _recursive_fuse(module):
            for name, child in module.named_children():
                # 1. 首先, 递归到底层
                _recursive_fuse(child)
                
                # 2. 然后, 检查当前子模块是否需要融合
                if isinstance(child, (SNN_Conv1d_BN, SNN_Linear_BN)):
                    print(f"  -> 正在融合: {name}")
                    fused_sequential = child.fuse_module()
                    setattr(module, name, fused_sequential) # 替换模块
        
        _recursive_fuse(self)
        print("BatchNorm 融合完成。")
        print("="*80)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 形状: (B, 66, 5)
        
        # --- 1. [修改] 并行线性嵌入 ---
        
        # 空间分支: (B, 66, 5) -> (B, 66, 128)
        # self.spatial_embedding 是 MAC 操作
        x_embed_spatial = self.spatial_embedding(x) + self.pos_embedding_spatial
        x_spatial = x_embed_spatial
        for block in self.spatial_transformer:
            x_spatial = block(x_spatial) # 内部是 AC 操作
        
        # "Temporal" (Spectral) 分支: (B, 5, 66) -> (B, 5, 128)
        x_permuted = x.permute(0, 2, 1) # (B, 5, 66)
        # self.temporal_embedding 是 MAC 操作
        x_embed_temporal = self.temporal_embedding(x_permuted) + self.pos_embedding_temporal
        x_temporal = x_embed_temporal
        for block in self.temporal_transformer:
            x_temporal = block(x_temporal) # 内部是 AC 操作

        # --- 2. 中期融合 ---
        x_t = x_temporal.mean(dim=1) # (B, 128)
        
        attn_scores = self.spatial_attn_pool(x_spatial) # (B, 66, 1)
        attn_weights = torch.softmax(attn_scores, dim=1) 
        x_s = torch.sum(attn_weights * x_spatial, dim=1) # (B, 128)
        
        x_fused = torch.cat([x_t, x_s], dim=-1) # (B, 256)

        # --- 3. 分类 ---
        label_output = self.label_classifier(x_fused)
        return label_output
    
# =============================================================================
# 10. 确定性 SNN 能耗计算器
# =============================================================================

class ComprehensiveSNNEnergyCalculator:
    """
    (已修正) 修正版的SNN能耗计算器
    - 修复: 比较原始张量的 ID，而不是 CPU 副本的 ID，以正确计算 SOPs。
    - 修复: 确保正确捕获 SNN_Linear_BN 的输入/输出。
    """
    
    def __init__(self, model, batch_size=32, device='cuda'):
        self.model = model.to(device)
        self.batch_size = batch_size
        self.device = device
        self.layer_analysis = []
        self.activations = {}
        self.hooks = []
        
    def register_hooks(self):
        """注册前向钩子捕获所有层的输入输出"""
        def hook_fn(module, input, output):
            module_id = id(module)
            
            # [!!! 修复 !!!] 存储原始张量以进行 ID 比较
            input_tensor = input[0] if input and input[0] is not None else None
            output_tensor = output if output is not None else None

            self.activations[module_id] = {
                'module': module,
                'module_type': module.__class__.__name__,
                'input_tensor_orig': input_tensor,   # <-- 存储原始张量
                'output_tensor_orig': output_tensor, # <-- 存储原始张量
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
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    # ... ( _get_kernel_and_stride, calculate_spiking_rate 保持不变 ) ...
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
        
    # ... ( calculate_conv1d_energy, calculate_linear_energy 保持不变 ) ...
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
            energy_coefficient = 0.9 # AC
        else:
            operations_per_output = 2 * kernel_size * (C_in // groups)
            total_operations = B * C_out * L_out * operations_per_output
            energy_coefficient = 4.6 # MAC
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
        
    # ... ( calculate_pooling_energy, calculate_norm_energy, calculate_activation_energy 保持不变 ) ...
    def calculate_pooling_energy(self, module, input_tensor):
        if input_tensor is None: return 0, 0
        if input_tensor.dim() == 3 and isinstance(module, (nn.AvgPool1d, nn.MaxPool1d)):
             B, C, L = input_tensor.shape
        elif input_tensor.dim() == 3 and isinstance(module, nn.AdaptiveAvgPool1d):
            B, C, L = input_tensor.shape
        elif input_tensor.dim() == 2 and isinstance(module, nn.AdaptiveAvgPool1d):
            input_tensor = input_tensor.unsqueeze(1)
            B, C, L = input_tensor.shape
        else:
            print(f"警告: Pooling 层 {module} 形状不匹配 {input_tensor.shape}")
            return 0,0
        energy_coefficient = 0.5
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
            else:
                total_operations = 0
        else:
            total_operations = 0
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
        """分析单层能耗 (is_spiking 作为参数传入)"""
        # [!!! 修复 !!!] 我们需要 .detach().cpu() 副本在这里进行
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
        """
        运行全面的能耗分析
        [修改] 增加打印详细的逐层和汇总分析
        """
        print("开始全面SNN能耗分析 (确定性版本)...")
        self.register_hooks()
        
        print("步骤 1/3: 运行前向传播以捕获计算图...")
        self.activations.clear(); self.layer_analysis.clear()
        
        input_tensor_device = input_tensor.to(self.device)
        with torch.no_grad():
            _ = self.model(input_tensor_device)
        print("计算图捕获完成。")

        print("步骤 2/3: 建立脉冲张量数据库 (Spiking Tensor DB)...")
        spiking_tensor_ids = set()
        target_spike_producers = ('Multispike', 'SNN_Conv1d_BN', 'SNN_Linear_BN')
        
        for module_id, activation_info in self.activations.items():
            if activation_info['module_type'] in target_spike_producers:
                output_tensor = activation_info['output_tensor_orig'] # [!!! 修复 !!!]
                if output_tensor is not None:
                    spiking_tensor_ids.add(id(output_tensor)) # [!!! 修复 !!!]
        print(f"找到 {len(spiking_tensor_ids)} 个脉冲激活输出源。")

        print("步骤 3/3: 逐层分析能耗...")
        total_energy, analyzed_layers = 0, 0
        total_flops_batch = 0.0 
        total_sops_batch = 0.0  
        
        for module_id, activation_info in self.activations.items():
            module = activation_info['module']
            input_tensor_orig = activation_info['input_tensor_orig'] # [!!! 修复 !!!]
            if input_tensor_orig is None: continue
            
            is_spiking = id(input_tensor_orig) in spiking_tensor_ids # [!!! 修复 !!!]
            
            if isinstance(module, (SNN_Conv1d_BN, SNN_Linear_BN)):
                continue 
            
            layer_info = self.analyze_layer_energy(module, input_tensor_orig, is_spiking) # [!!! 修复 !!!]
            
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
        
        print(f"\n分析完成!")
        
        # ... (打印部分保持不变) ...
        print("\n" + "="*80)
        print("能耗汇总 (按模块类型):")
        print("="*80)
        try:
            df_detailed = pd.DataFrame(self.layer_analysis)
            if df_detailed.empty:
                print("未分析到任何层。")
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
            print(summary_by_type.to_string(float_format="%.2f"))
            
            print("\n" + "="*80)
            print("详细能耗分析 (逐层):")
            print("="*80)
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            print(df_detailed.to_string(index=False, float_format="%.2f"))
        except Exception as e:
            print(f"打印详细报告时出错: {e}")

        print("\n--- 最终总结 ---")
        print(f"总分析层数: {analyzed_layers}")
        print(f"FLOPs (G) / sample: {flops_per_sample_G:.4f} G")
        print(f"SOPs  (G) / sample: {sops_per_sample_G:.4f} G")
        print(f"Energy (mJ) / sample: {energy_per_sample_mJ:.6f} mJ")
        
        return energy_per_sample_pJ, flops_per_sample_G, sops_per_sample_G # [!!! 修改 !!!]
    
    # ... (convert_to_serializable 和 generate_detailed_report 保持不变) ...
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
            print("没有分析数据，请先运行run_comprehensive_analysis")
            return
        os.makedirs(save_path, exist_ok=True)
        df_detailed = pd.DataFrame(self.layer_analysis)
        for col in df_detailed.select_dtypes(include=[np.number]).columns:
            df_detailed[col] = df_detailed[col].apply(lambda x: float(x) if pd.notnull(x) else x)
        df_detailed.to_excel(os.path.join(save_path, "detailed_energy_analysis.xlsx"), index=False)
        summary_by_type = df_detailed.groupby('module_type').agg(
            layer_count=('module_name', 'count'),
            total_energy_pJ=('layer_energy_pJ', 'sum'),
            total_operations=('total_operations', 'sum')
        )
        total_energy_all = df_detailed['layer_energy_pJ'].sum()
        summary_by_type['energy_percentage (%)'] = (
            summary_by_type['total_energy_pJ'] / (total_energy_all + 1e-9) * 100
        )
        summary_by_type = summary_by_type.apply(lambda x: x.map(float) if x.name in ['total_energy_pJ', 'total_operations', 'energy_percentage (%)'] else x.map(int))
        summary_by_type.to_excel(os.path.join(save_path, "energy_summary_by_type.xlsx"))
        total_energy = float(df_detailed['layer_energy_pJ'].sum())
        total_operations = float(df_detailed['total_operations'].sum())
        
        summary = {
            "total_energy_pJ": total_energy,
            "total_energy_nJ": total_energy / 1e3,
            "total_energy_uJ": total_energy / 1e6,
            "energy_per_sample_pJ": total_energy / self.batch_size,
            "energy_per_sample_nJ": total_energy / self.batch_size / 1e3,
            "energy_per_sample_uJ": total_energy / self.batch_size / 1e6,
            "total_operations": total_operations,
            "operations_per_sample": total_operations / self.batch_size,
            "total_layers_analyzed": len(self.layer_analysis),
            "batch_size": self.batch_size,
            "module_type_breakdown": self.convert_to_serializable(summary_by_type.to_dict('index'))
        }
        summary = self.convert_to_serializable(summary)
        with open(os.path.join(save_path, "comprehensive_energy_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n详细能耗报告已保存到: {save_path}")
        return summary
        
# =============================================================================
# 11. 简化版标准训练器 (SimpleTrainer)
# =============================================================================
class SimpleTrainer:
    """简化的标准训练器 (Within-Subject 版本)"""
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

# =============================================================================
# 12. 训练主流程与可视化 (5-Fold CV 版本)
# =============================================================================
def run_training_fold(train_loader, test_loader, config, fold_save_dir):
    """
    为 *单个* 交叉验证折 (Fold) 运行训练流程。
    [修改] 使用 Optimized_SNN_DE_DBformer
    """
    
    # [修改] 实例化 *优化后* 的 SNN DE 模型
    model = Optimized_SNN_DE_DBformer( 
        chn=config.channel_size,
        num_bands=config.num_bands, # [修改]
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
    
    print(f"--- [优化的 SNN-DE] 开始训练 {config.epochs} 个 Epochs ---")
    for epoch in range(config.epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch=epoch, num_epochs=config.epochs)
        
        model.eval() 
        test_metrics = trainer.test_epoch(test_loader, epoch=epoch, num_epochs=config.epochs)
        model.train() 
        
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

    print(f"本折训练完成！最佳测试准确率: {best_test_acc:.4f}\n")
    
    return best_test_acc, training_history, best_model_path 

def save_history_plot_simple(history, save_dir, title_prefix):
    """保存训练曲线 (简化版：仅 Loss 和 Acc)"""
    if not history: 
        warnings.warn("训练历史为空，跳过绘图。")
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
    fig.suptitle(f'Optimized SNN-DE on {title_prefix} - Training History')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close(fig)

# =============================================================================
# 13. [修改] 实验配置 (DE 特征版本)
# =============================================================================
class TrainingConfig:
    """存储所有实验超参数和路径的配置类 (DE 特征版本)"""
    def __init__(self):
        self.dataset_name = "DTU-DE" 
        self.data_root = "/share/home/yuan/LY/SNN_DBformer/DE_DTU/DE_1s" # [修改]
        self.save_dir = './WS_5FoldCV_OptimizedSNN_DE_RESULTS_DTU' # [修改]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 128
        self.num_workers = 8 
        self.learning_rate = 1e-4
        self.epochs = 20 # [修改] 减少 Epochs 以便快速测试
        self.weight_decay = 1e-3
        self.patience = 10 
        self.factor = 0.5  
        
        # --- [修改] 模型参数 ---
        self.channel_size = 66
        self.num_bands = 5 # (取代 time_sample_num)
        # (移除 patch_size, spa_dim)
        
        self.emb_size = 48
        self.depth = 3
        self.num_heads = 8  
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# =============================================================================
# 14. [修改] 主执行脚本 (S1-F1, CSP 处理, 训后能耗分析)
# =============================================================================

def main():
    """
    主执行函数：[修改] 加入 CSP 预处理，仅训练被试1的折1, 并在之后运行能耗分析
    """
    try:
        config = TrainingConfig()
        main_results_dir = f"{config.save_dir}_{config.timestamp}"
        os.makedirs(main_results_dir, exist_ok=True)
        
        print("="*80)
        print(f"任务: [优化版 SNN-DE + CSP] 训练 S1-F1 并在训后分析能耗")
        print(f"设备: {config.device}")
        print(f"主结果路径: {main_results_dir}")
        print("="*80)
        
        all_subject_ids = ["1"] # [修改] 仅被试 1
        
        for subject_id in all_subject_ids:
            print(f"\n{'='*30} 开始处理被试: S{subject_id} {'='*30}")
            
            subject_save_dir = os.path.join(main_results_dir, f"subject_{subject_id}")
            os.makedirs(subject_save_dir, exist_ok=True)
            
            print(f"正在加载被试 {subject_id} 的 DE 特征数据...")
            try:
                # 1. 加载原始数据集对象
                subject_dataset = DTU_AAD_Dataset_WS(config.data_root, subject_id)
                print(f"被试 {subject_id} 数据加载完成，总样本数: {len(subject_dataset)}")
                
                # [新增] 提取全部数据为 Numpy 格式，方便 sklearn/MNE 处理
                # 假设 dataset.eeg_features 是 Tensor (N, 66, 5)
                # 假设 dataset.direction_label 是 Tensor (N,)
                X_all = subject_dataset.eeg_features.numpy() 
                y_all = subject_dataset.direction_label.numpy()
                
            except FileNotFoundError as e:
                print(e)
                print(f"!!! 关键错误: 确保 S{subject_id}_DE_Features_1s.npz 在 {config.data_root} 中")
                continue
            
            N_SPLITS = 5
            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
            
            # 使用 enumerate 获取 fold 索引
            for fold_idx, (train_indices, test_indices) in enumerate(kf.split(X_all)):
                
                # [修改] 仅折 1
                if fold_idx > 0: # fold_idx 从 0 开始，0 代表第1折
                    print(f"跳过 Fold {fold_idx+1}, 只训练 F1...")
                    break 

                fold_num = fold_idx + 1
                print(f"\n--- 被试 S{subject_id} | 折 {fold_num}/{N_SPLITS} ---")
                
                fold_save_dir = os.path.join(subject_save_dir, f"fold_{fold_num}")
                os.makedirs(fold_save_dir, exist_ok=True)

                # =========================================================
                # [核心修改] CSP 处理部分
                # =========================================================
                print("正在进行 CSP 数据处理...")
                
                # 1. 根据索引分割数据 (Numpy)
                X_train_np, y_train_np = X_all[train_indices], y_all[train_indices]
                X_test_np, y_test_np = X_all[test_indices], y_all[test_indices]
                
                # 2. 定义 CSP
                # 注意: CSP 期望输入形状为 (n_epochs, n_channels, n_times)
                # 您的数据形状为 (N, 66, 5)，这里 66 是通道，5 是频带(充当时间维)，符合 CSP 接口要求
                csp = CSP(
                    n_components=config.channel_size, # 66
                    reg=None, 
                    log=None, 
                    cov_est='concat', 
                    transform_into='csp_space', # 输出空间滤波后的信号，而不是功率
                    norm_trace=True
                )
                
                # 3. Fit 和 Transform
                # 注意：必须只在训练集上 fit，防止数据泄露
                print(f"  CSP Fitting on Train Shape: {X_train_np.shape}")
                X_train_csp = csp.fit_transform(X_train_np, y_train_np)
                X_test_csp = csp.transform(X_test_np)
                print(f"  CSP Transformed Train Shape: {X_train_csp.shape}")
                
                # 4. 转换回 PyTorch Tensor
                # 此时数据已经是 float64 (numpy默认)，建议转为 float32
                train_data_tensor = torch.tensor(X_train_csp, dtype=torch.float32)
                train_label_tensor = torch.tensor(y_train_np, dtype=torch.long)
                
                test_data_tensor = torch.tensor(X_test_csp, dtype=torch.float32)
                test_label_tensor = torch.tensor(y_test_np, dtype=torch.long)
                
                # 5. 封装为 TensorDataset (替代原来的 Subset)
                train_dataset_csp = TensorDataset(train_data_tensor, train_label_tensor)
                test_dataset_csp = TensorDataset(test_data_tensor, test_label_tensor)
                
                # =========================================================
                # [结束 CSP 修改]
                # =========================================================
                
                train_loader = DataLoader(train_dataset_csp, 
                                          config.batch_size, shuffle=True, 
                                          num_workers=config.num_workers, pin_memory=True)
                test_loader = DataLoader(test_dataset_csp, 
                                         config.batch_size, shuffle=False, 
                                         num_workers=config.num_workers, pin_memory=True)
                
                print(f"训练集: {len(train_dataset_csp)} 样本, 测试集: {len(test_dataset_csp)} 样本")

                # --- 6. 运行该折的训练 ---
                best_acc, history, best_model_path = run_training_fold(
                    train_loader, 
                    test_loader, 
                    config, 
                    fold_save_dir
                )
                
                pd.DataFrame(history).to_csv(os.path.join(fold_save_dir, "training_history.csv"), index=False)
                save_history_plot_simple(history, fold_save_dir, f"{config.dataset_name}_S{subject_id}_F{fold_num}")
                
                # --- 7. [新增] 训练完成后立即进行能耗分析 ---
                print("\n" + "="*80)
                print(f"训练完成。开始对 {best_model_path} 进行能耗分析...")
                
                # 7.1. 实例化 *优化后* 的模型
                model_for_energy = Optimized_SNN_DE_DBformer(
                    chn=config.channel_size,
                    num_bands=config.num_bands, 
                    num_direction_classes=2,
                    emb_size=config.emb_size,
                    depth=config.depth,
                    num_heads=config.num_heads
                ).to(config.device)
                
                # 7.2. 加载最佳权重
                model_for_energy.load_state_dict(torch.load(best_model_path, map_location=config.device))
                
                # 7.3. [关键] 融合BN层
                model_for_energy.fuse_model_bn()
                
                # 7.3b. 计算参数
                total_params = count_parameters(model_for_energy)
                params_m = total_params / 1_000_000
                print(f"模型参数 (Params): {params_m:.2f} M")
                
                # 7.4. 获取一个测试数据批次 (注意：这里已经是 CSP 处理过的数据)
                input_tensor, _ = next(iter(test_loader))
                print(f"使用一个测试 batch 进行分析，形状: {input_tensor.shape}")

                # 7.5. 初始化并运行能耗计算器
                energy_calculator = ComprehensiveSNNEnergyCalculator(
                    model=model_for_energy,
                    batch_size=config.batch_size,
                    device=config.device
                )
                
                # 7.6. 运行分析
                energy_pJ, flops_G, sops_G = \
                    energy_calculator.run_comprehensive_analysis(input_tensor.to(config.device))
                
                energy_mJ = energy_pJ / 1e9 # 转换为 mJ
                
                # 7.7. 打印所有 4 个指标
                print(f"\n--- 最终估算结果 (Optimized SNN-DE + CSP @ S{subject_id}-F{fold_num}) ---")
                print(f"Params (M): {params_m:.2f} M")
                print(f"FLOPs (G): {flops_G:.4f} G")
                print(f"SOPs  (G): {sops_G:.4f} G")
                print(f"Energy (mJ): {energy_mJ:.6f} mJ")
                
                # 7.8. 保存详细报告
                energy_calculator.generate_detailed_report(os.path.join(fold_save_dir, "energy_report_fused"))


        print("\n" + "="*80)
        print("🎉🎉🎉 S1-F1 训练及能耗分析已全部完成！ 🎉🎉🎉")
        
    except Exception as e:
        print(f"\n❌ 脚本执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# 15. 主执行入口
# =============================================================================

if __name__ == "__main__":
    
    # [修改] 始终运行 main()，它现在只处理 S1-F1 并进行分析
    print("模式: 运行优化的 SNN-DE 训练 (S1-F1) 及训后能耗分析...")
    
    # [新增] 添加 count_parameters 函数的定义
    def count_parameters(model):
        """计算模型的可训练参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    main()