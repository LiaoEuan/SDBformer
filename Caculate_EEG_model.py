import os
import torch
import torch.nn as nn
import pandas as pd
import json
import numpy as np

# =============================================================================
# 1. 核心 SNN 组件
# =============================================================================
class Quant(torch.autograd.Function):
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
    def __init__(self, min_value=0.0, max_value=4.0):
        super().__init__()
        self.min_value, self.max_value = min_value, max_value
    def forward(self, x):
        return Quant.apply(x, self.min_value, self.max_value) / 2.0

class SNN_Conv1d_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.spike = Multispike()
        
    def forward(self, x):
        return self.spike(self.bn(self.conv(x)))
        
    def fuse_module(self):
        """融合 BN 层用于推理和能耗计算"""
        if self.training: return self
        conv_w, conv_b = self.conv.weight, self.conv.bias if self.conv.bias is not None else torch.zeros(self.conv.out_channels, device=self.conv.weight.device)
        bn_rm, bn_rv, bn_eps, bn_w, bn_b = self.bn.running_mean, self.bn.running_var, self.bn.eps, self.bn.weight, self.bn.bias
        scale = bn_w / torch.sqrt(bn_rv + bn_eps)
        fused_w = conv_w * scale.view(-1, 1, 1)
        fused_b = (conv_b - bn_rm) * scale + bn_b
        fused_conv = nn.Conv1d(self.conv.in_channels, self.conv.out_channels, self.conv.kernel_size, self.conv.stride, self.conv.padding, groups=self.conv.groups, bias=True).to(fused_w.device)
        fused_conv.weight.data, fused_conv.bias.data = fused_w, fused_b
        return nn.Sequential(fused_conv, self.spike)

# =============================================================================
# 2. EEG 专用的 SNN 嵌入与 Transformer 层
# =============================================================================
class SNN_Stem(nn.Module):
    """用于处理时序 EEG 信号的卷积 Stem"""
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
    def forward(self, x): return self.stem(x).permute(0, 2, 1)

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

class SNN_Attention_BNC(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads, self.scale = num_heads, 0.125
        self.head_spike = Multispike()
        self.q_conv, self.k_conv, self.v_conv, self.proj_conv = [SNN_Conv1d_BN(dim, dim, 1) for _ in range(4)]
        self.attn_spike = Multispike()

    def forward(self, x):
        B, N, C = x.shape
        x = self.head_spike(x)
        x_t = x.transpose(1, 2)
        q = self.q_conv(x_t).transpose(1, 2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_conv(x_t).transpose(1, 2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_conv(x_t).transpose(1, 2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x_attn = (q @ (k.transpose(-2, -1) @ v)) * self.scale    
        x = self.attn_spike(x_attn.transpose(1, 2).reshape(B, N, C))
        return self.proj_conv(x.transpose(1, 2)).transpose(1, 2)

class SNN_MLP_BNC(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hf = hidden_features or in_features
        self.head_spike = Multispike()
        self.mlp1, self.mlp2 = SNN_Conv1d_BN(in_features, hf, 1), SNN_Conv1d_BN(hf, in_features, 1)
    def forward(self, x):
        return self.mlp2(self.mlp1(self.head_spike(x.transpose(1, 2)))).transpose(1, 2)

class SNN_TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads):
        super().__init__()
        self.attn = SNN_Attention_BNC(emb_size, num_heads)
        self.mlp = SNN_MLP_BNC(emb_size, emb_size * 2) 
    def forward(self, x): return x + self.mlp(x + self.attn(x))

# =============================================================================
# 3. 基于 EEG 的主模型 (SNN_EEG_DBformer)
# =============================================================================
class SNN_EEG_DBformer(nn.Module):
    """基于原始 EEG 数据的 SNN 模型"""
    def __init__(self, chn=64, time_sample_num=128, patch_size=16, spa_dim=16,
                 num_direction_classes=2, emb_size=128, depth=2, num_heads=8):
        super().__init__()
        
        self.temporal_embedding = SNN_PatchEmbeddingTemporal(chn, patch_size, emb_size)
        self.spatial_embedding = SNN_PatchEmbeddingSpatial(spa_dim, emb_size)
        
        self.pos_embedding_temporal = nn.Parameter(torch.randn(1, time_sample_num // patch_size, emb_size))
        self.pos_embedding_spatial = nn.Parameter(torch.randn(1, chn, emb_size))
        
        self.temporal_transformer = nn.ModuleList([SNN_TransformerBlock(emb_size, num_heads) for _ in range(depth)])
        self.spatial_transformer = nn.ModuleList([SNN_TransformerBlock(emb_size, num_heads) for _ in range(depth)])

        self.spatial_attn_pool = nn.Sequential(nn.Linear(emb_size, emb_size), Multispike(), nn.Linear(emb_size, 1))
        
        self.label_classifier = nn.Sequential(
            nn.Linear(emb_size * 2, 64), 
            nn.BatchNorm1d(64), 
            Multispike(),       
            nn.Linear(64, num_direction_classes)
        )

    def fuse_model_bn(self):
        """融合模型中的 BN 层"""
        self.eval()
        def _recursive_fuse(module):
            for name, child in module.named_children():
                _recursive_fuse(child)
                if isinstance(child, SNN_Conv1d_BN):
                    setattr(module, name, child.fuse_module())
        _recursive_fuse(self)

    def forward(self, x):
        # x: (B, Channels, Time)
        x_t = self.temporal_embedding(x) + self.pos_embedding_temporal
        for block in self.temporal_transformer: x_t = block(x_t)
        x_t = x_t.mean(dim=1)
        
        x_s = self.spatial_embedding(x) + self.pos_embedding_spatial
        for block in self.spatial_transformer: x_s = block(x_s)
        attn_scores = self.spatial_attn_pool(x_s)
        x_s = torch.sum(torch.softmax(attn_scores, dim=1) * x_s, dim=1)

        x_fused = torch.cat([x_t, x_s], dim=-1)
        return self.label_classifier(x_fused)

# =============================================================================
# 4. 能耗计算器 (完全保持原样，直接复用)
# =============================================================================
class ComprehensiveSNNEnergyCalculator:
    def __init__(self, model, batch_size=32, device='cuda'):
        self.model = model.to(device)
        self.batch_size = batch_size
        self.device = device
        self.layer_analysis = []
        self.activations = {}
        self.hooks = []
        
    def register_hooks(self):
        def hook_fn(module, input, output):
            module_id = id(module)
            self.activations[module_id] = {
                'module': module,
                'module_type': module.__class__.__name__,
                'input_tensor_orig': input[0] if input and input[0] is not None else None,
                'output_tensor_orig': output if output is not None else None,
            }
        self.remove_hooks()
        target_modules = (nn.Conv1d, nn.Linear, nn.AvgPool1d, nn.MaxPool1d, nn.AdaptiveAvgPool1d, nn.BatchNorm1d, nn.LayerNorm, nn.ReLU, nn.Sigmoid, nn.Tanh, Multispike, SNN_Conv1d_BN)
        for name, module in self.model.named_modules():
            if isinstance(module, target_modules):
                self.hooks.append(module.register_forward_hook(hook_fn))
    
    def remove_hooks(self):
        for hook in self.hooks: hook.remove()
        self.hooks = []
    
    def calculate_spiking_rate(self, tensor):
        if tensor is None: return 0.0, 0, 0
        total_elements = tensor.numel()
        spiking_sum = torch.count_nonzero(tensor).item()
        return (spiking_sum / total_elements if total_elements > 0 else 0), spiking_sum, total_elements
        
    def calculate_conv1d_energy(self, module, input_tensor, is_spiking):
        if input_tensor is None: return 0, 0, 0
        B, C_in, L = input_tensor.shape
        L_out = (L + 2 * module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) - 1) // module.stride[0] + 1
        ops_per_out = module.kernel_size[0] * (C_in // module.groups)
        if not is_spiking: ops_per_out *= 2
        total_ops = B * module.out_channels * L_out * ops_per_out
        energy_coeff = 0.9 if is_spiking else 4.6
        spiking_rate = self.calculate_spiking_rate(input_tensor)[0] if is_spiking else 1.0
        return total_ops * energy_coeff * spiking_rate, total_ops, spiking_rate
    
    def calculate_linear_energy(self, module, input_tensor, is_spiking):
        if input_tensor is None: return 0, 0, 0
        batch_dim = input_tensor.shape[0] if input_tensor.dim() == 2 else input_tensor.shape[0] * input_tensor.shape[1]
        ops_per_out = module.in_features if is_spiking else 2 * module.in_features
        total_ops = batch_dim * module.out_features * ops_per_out
        energy_coeff = 0.9 if is_spiking else 4.6
        spiking_rate = self.calculate_spiking_rate(input_tensor)[0] if is_spiking else 1.0
        return total_ops * energy_coeff * spiking_rate, total_ops, spiking_rate
        
    def calculate_pooling_energy(self, module, input_tensor):
        if input_tensor is None or input_tensor.dim() < 2: return 0, 0
        B, C, L = input_tensor.shape if input_tensor.dim() == 3 else (input_tensor.shape[0], 1, input_tensor.shape[1])
        total_ops = 0
        if isinstance(module, (nn.AvgPool1d, nn.MaxPool1d)):
            L_out = (L + 2 * (module.padding[0] if isinstance(module.padding, tuple) else module.padding) - (module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size)) // (module.stride[0] if isinstance(module.stride, tuple) else module.stride) + 1
            total_ops = B * C * L_out * ((module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size) - 1)
        elif isinstance(module, nn.AdaptiveAvgPool1d) and module.output_size == 1:
            total_ops = B * C * (L - 1)
        return total_ops * 0.5, total_ops
    
    def calculate_norm_energy(self, module, input_tensor):
        if input_tensor is None: return 0, 0
        total_elements = input_tensor.numel()
        total_ops = 2 * total_elements if isinstance(module, nn.BatchNorm1d) else (5 * total_elements if isinstance(module, nn.LayerNorm) else 0)
        return total_ops * (0.3 if isinstance(module, nn.BatchNorm1d) else 0.4), total_ops
    
    def calculate_activation_energy(self, module, input_tensor):
        if input_tensor is None: return 0, 0
        total_elements = input_tensor.numel()
        total_ops = total_elements if isinstance(module, (nn.ReLU, Multispike)) else (5 * total_elements if isinstance(module, (nn.Sigmoid, nn.Tanh)) else 0)
        return total_ops * (0.2 if isinstance(module, (nn.ReLU, Multispike)) else 0.5), total_ops
        
    def analyze_layer_energy(self, module, input_tensor, is_spiking):
        input_cpu = input_tensor.detach().cpu()
        module_type = module.__class__.__name__
        energy, ops, sr = 0, 0, 0.0
        
        if isinstance(module, nn.Conv1d): energy, ops, sr = self.calculate_conv1d_energy(module, input_cpu, is_spiking)
        elif isinstance(module, nn.Linear): energy, ops, sr = self.calculate_linear_energy(module, input_cpu, is_spiking)
        elif isinstance(module, (nn.AvgPool1d, nn.MaxPool1d, nn.AdaptiveAvgPool1d)): energy, ops = self.calculate_pooling_energy(module, input_cpu); sr = 1.0
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)): energy, ops = self.calculate_norm_energy(module, input_cpu); sr = 1.0
        elif isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh, Multispike)): energy, ops = self.calculate_activation_energy(module, input_cpu); sr = 1.0
        else: return None
        
        return {
            'module_name': f"{module_type}_{id(module)}", 'module_type': module_type,
            'is_spiking': is_spiking, 'total_operations': float(ops), 'spiking_rate': float(sr),
            'layer_energy_pJ': float(energy), 'energy_per_sample_pJ': float(energy / self.batch_size)
        }
    
    def run_comprehensive_analysis(self, input_tensor):
        self.register_hooks()
        self.activations.clear(); self.layer_analysis.clear()
        
        with torch.no_grad(): _ = self.model(input_tensor.to(self.device))

        spiking_tensor_ids = {id(info['output_tensor_orig']) for info in self.activations.values() if info['module_type'] in ('Multispike', 'SNN_Conv1d_BN') and info['output_tensor_orig'] is not None}
        
        total_energy, total_flops, total_sops = 0, 0.0, 0.0  
        for info in self.activations.values():
            module, input_orig = info['module'], info['input_tensor_orig']
            if input_orig is None or isinstance(module, SNN_Conv1d_BN): continue
            
            is_spiking = id(input_orig) in spiking_tensor_ids
            layer_info = self.analyze_layer_energy(module, input_orig, is_spiking)
            
            if layer_info:
                self.layer_analysis.append(layer_info)
                total_energy += layer_info['layer_energy_pJ']
                if layer_info['is_spiking']: total_sops += layer_info['total_operations']
                else: total_flops += layer_info['total_operations']
        
        self.remove_hooks()
        energy_per_sample_mJ = (total_energy / self.batch_size) / 1e9
        flops_G = (total_flops / self.batch_size) / 1e9
        sops_G = (total_sops / self.batch_size) / 1e9
        
        return energy_per_sample_mJ, flops_G, sops_G

# =============================================================================
# 5. 执行计算的主函数
# =============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 设置 EEG 参数 (根据您的数据集调整，例如 KUL 是 64通道 128采样点，DTU 是 66通道 256采样点)
    BATCH_SIZE = 32
    CHANNELS = 64       # EEG 通道数
    TIME_SAMPLES = 128  # EEG 时序采样点

    print(f"\n--- 初始化 EEG SNN 模型 ---")
    model = SNN_EEG_DBformer(
        chn=CHANNELS, 
        time_sample_num=TIME_SAMPLES, 
        patch_size=16, 
        spa_dim=16,
        emb_size=128, 
        depth=2, 
        num_heads=8
    ).to(device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = total_params / 1_000_000

    # 融合 BN 层 (模拟真实推理阶段)
    model.fuse_model_bn()

    # 2. 创建模拟的 EEG 输入张量
    # 形状: (Batch, Channels, Time)
    dummy_input = torch.randn(BATCH_SIZE, CHANNELS, TIME_SAMPLES).to(device)
    print(f"模拟 EEG 输入形状: {dummy_input.shape}")

    # 3. 运行能耗计算器
    print("\n--- 开始进行能耗计算 ---")
    calculator = ComprehensiveSNNEnergyCalculator(model, batch_size=BATCH_SIZE, device=device)
    energy_mJ, flops_G, sops_G = calculator.run_comprehensive_analysis(dummy_input)

    # 4. 打印最终结果
    print("\n" + "="*50)
    print("最终估算结果 (SNN EEG DBformer):")
    print("="*50)
    print(f"Params (M): {params_m:.4f} M")
    print(f"FLOPs (G):  {flops_G:.4f} G")
    print(f"SOPs  (G):  {sops_G:.4f} G")
    print(f"Energy (mJ): {energy_mJ:.6f} mJ")
    print("="*50)

if __name__ == "__main__":
    main()