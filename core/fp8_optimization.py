"""
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import math


HAS_FP8 = hasattr(torch, 'float8_e4m3fn') and hasattr(torch, 'float8_e5m2')

@dataclass
class FP8Config:
    """
"""
    use_fp8: bool = True
    e4m3_for_forward: bool = True  
    e5m2_for_backward: bool = True  
    dynamic_scaling: bool = True
    scale_update_freq: int = 100
    initial_scale: float = 1.0
    max_scale: float = 65536.0
    min_scale: float = 1e-6
    
    
    uncertainty_threshold: float = 0.1
    importance_threshold: float = 0.05

class FP8ScaleManager:
    """
"""
    def __init__(self, config: FP8Config):
        self.config = config
        self.forward_scale = config.initial_scale
        self.backward_scale = config.initial_scale
        self.step_count = 0
        
        
        self.overflow_history = []
        self.underflow_history = []
        
    def get_forward_scale(self) -> float:
        """
"""
        return self.forward_scale
    
    def get_backward_scale(self) -> float:
        """
"""
        return self.backward_scale
    
    def update_scales(self, forward_overflow: bool = False, 
                     backward_overflow: bool = False):
        """
"""
        self.step_count += 1
        
        
        self.overflow_history.append(forward_overflow)
        self.underflow_history.append(not forward_overflow and not backward_overflow)
        
        
        if len(self.overflow_history) > self.config.scale_update_freq:
            self.overflow_history.pop(0)
            self.underflow_history.pop(0)
        
        
        if self.step_count % self.config.scale_update_freq == 0:
            self._adjust_scales()
    
    def _adjust_scales(self):
        """
"""
        overflow_rate = sum(self.overflow_history) / len(self.overflow_history)
        underflow_rate = sum(self.underflow_history) / len(self.underflow_history)
        
        
        if overflow_rate > 0.05:  
            self.forward_scale = max(self.config.min_scale, self.forward_scale * 0.5)
        elif underflow_rate > 0.8:  
            self.forward_scale = min(self.config.max_scale, self.forward_scale * 2.0)
        
        
        if overflow_rate > 0.02:
            self.backward_scale = max(self.config.min_scale, self.backward_scale * 0.5)
        elif underflow_rate > 0.9:
            self.backward_scale = min(self.config.max_scale, self.backward_scale * 1.5)

class FP8Tensor:
    """
"""
    def __init__(self, data: torch.Tensor, scale: float, dtype_str: str):
        self.data = data
        self.scale = scale
        self.dtype_str = dtype_str
        self.original_shape = data.shape
        self.original_dtype = data.dtype
    
    def to_fp32(self) -> torch.Tensor:
        """
"""
        return self.data.to(torch.float32) * self.scale
    
    def to_fp16(self) -> torch.Tensor:
        """
"""
        return self.data.to(torch.float16) * self.scale

def to_fp8_e4m3(tensor: torch.Tensor, scale: float = 1.0) -> FP8Tensor:
    """
"""
    if not HAS_FP8:
        
        scaled = tensor / scale
        quantized = scaled.to(torch.float16)
        return FP8Tensor(quantized, scale, "fp16_fallback")
    
    
    scaled = tensor / scale
    
    
    clamped = torch.clamp(scaled, -448.0, 448.0)
    
    
    fp8_tensor = clamped.to(torch.float8_e4m3fn)
    
    return FP8Tensor(fp8_tensor, scale, "fp8_e4m3")

def to_fp8_e5m2(tensor: torch.Tensor, scale: float = 1.0) -> FP8Tensor:
    """
"""
    if not HAS_FP8:
        
        scaled = tensor / scale
        quantized = scaled.to(torch.float16)
        return FP8Tensor(quantized, scale, "fp16_fallback")
    
    
    scaled = tensor / scale
    
    
    clamped = torch.clamp(scaled, -57344.0, 57344.0)
    
    
    fp8_tensor = clamped.to(torch.float8_e5m2)
    
    return FP8Tensor(fp8_tensor, scale, "fp8_e5m2")

class FP8Linear(nn.Module):
    """
"""
    def __init__(self, in_features: int, out_features: int, 
                 bias: bool = True, config: FP8Config = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or FP8Config()
        
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        
        self.scale_manager = FP8ScaleManager(self.config)
        
        
        self._cached_fp8_weight = None
        self._cache_step = -1
    
    def _get_fp8_weight(self) -> FP8Tensor:
        """
"""
        current_step = self.scale_manager.step_count
        
        if (self._cached_fp8_weight is None or 
            self._cache_step != current_step):
            
            scale = self.scale_manager.get_forward_scale()
            self._cached_fp8_weight = to_fp8_e4m3(self.weight, scale)
            self._cache_step = current_step
        
        return self._cached_fp8_weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        original_shape = x.shape
        is_batched = x.dim() == 3
        if is_batched:
            B, N, C = original_shape
            x2d = x.reshape(B * N, C)
        else:
            x2d = x
        
        if not self.config.use_fp8 or not HAS_FP8:
            
            out2d = F.linear(x2d, self.weight, self.bias)
            if is_batched:
                return out2d.reshape(B, N, self.out_features)
            return out2d
        
        
        input_scale = self.scale_manager.get_forward_scale()
        fp8_input = to_fp8_e4m3(x2d, input_scale)
        
        
        fp8_weight = self._get_fp8_weight()
        
        
        x_mat = fp8_input.data.to(torch.float16)
        w_mat = fp8_weight.data.to(torch.float16)
        
        out2d = x_mat @ w_mat.T
        
        out2d = out2d.to(torch.float32) * (fp8_input.scale * fp8_weight.scale)
        
        
        if self.bias is not None:
            out2d = out2d + self.bias
        
        
        if is_batched:
            return out2d.reshape(B, N, self.out_features)
        return out2d

class FP8Attention(nn.Module):
    """
"""
    def __init__(self, dim: int, n_heads: int, config: FP8Config = None):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.config = config or FP8Config()
        
        
        self.q_proj = FP8Linear(dim, dim, bias=False, config=config)
        self.k_proj = FP8Linear(dim, dim, bias=False, config=config)
        self.v_proj = FP8Linear(dim, dim, bias=False, config=config)
        self.out_proj = FP8Linear(dim, dim, config=config)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, 
                uncertainty: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        
        
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim)
        
        
        if uncertainty is not None and self.config.use_fp8:
            
            high_uncertainty = uncertainty > self.config.uncertainty_threshold
            
            
            if high_uncertainty.any():
                q_mixed = q.clone()
                k_mixed = k.clone()
                v_mixed = v.clone()
                
                
                q_mixed[high_uncertainty] = q[high_uncertainty].to(torch.float32)
                k_mixed[high_uncertainty] = k[high_uncertainty].to(torch.float32)
                v_mixed[high_uncertainty] = v[high_uncertainty].to(torch.float32)
                
                q, k, v = q_mixed, k_mixed, v_mixed
        
        
        q = q.transpose(1, 2)  
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        
        if self.config.use_fp8 and HAS_FP8:
            
            scale = 1.0 / math.sqrt(self.head_dim)
            
            
            fp8_q = to_fp8_e4m3(q, scale)
            fp8_k = to_fp8_e4m3(k, scale)
            
            attn = torch.matmul(fp8_q.to_fp16(), fp8_k.to_fp16().transpose(-2, -1))
        else:
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        
        if self.config.use_fp8 and HAS_FP8:
            fp8_attn = to_fp8_e4m3(attn, 1.0)
            fp8_v = to_fp8_e4m3(v, 1.0)
            out = torch.matmul(fp8_attn.to_fp16(), fp8_v.to_fp16())
        else:
            out = torch.matmul(attn, v)
        
        
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        
        return out

class FP8KVCache:
    """
"""
    def __init__(self, max_size: int, n_heads: int, head_dim: int, 
                 config: FP8Config = None):
        self.max_size = max_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.config = config or FP8Config()
        
        
        if HAS_FP8 and config.use_fp8:
            self.k_cache = torch.zeros(
                (n_heads, max_size, head_dim),
                dtype=torch.float8_e4m3fn,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            self.v_cache = torch.zeros(
                (n_heads, max_size, head_dim),
                dtype=torch.float8_e4m3fn,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            
            self.k_cache = torch.zeros(
                (n_heads, max_size, head_dim),
                dtype=torch.float16,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            self.v_cache = torch.zeros(
                (n_heads, max_size, head_dim),
                dtype=torch.float16,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        
        
        self.scales_k = torch.ones(max_size, device=self.k_cache.device)
        self.scales_v = torch.ones(max_size, device=self.v_cache.device)
        self.current_pos = 0
        self.is_full = False
        
        
        self.total_stored = 0
        self.fp8_ratio = 0.0
    
    def add(self, k: torch.Tensor, v: torch.Tensor, 
            uncertainty: Optional[torch.Tensor] = None):
        """
"""
        batch_size, seq_len, n_heads, head_dim = k.shape
        
        for i in range(seq_len):
            if self.current_pos >= self.max_size:
                self.current_pos = 0
                self.is_full = True
            
            
            use_fp8 = True
            if uncertainty is not None:
                token_uncertainty = uncertainty[0, i].item()
                if token_uncertainty > self.config.uncertainty_threshold:
                    use_fp8 = False  
            
            
            k_token = k[0, i].T  
            v_token = v[0, i].T
            
            if use_fp8 and self.config.use_fp8 and HAS_FP8:
                
                scale_k = torch.max(torch.abs(k_token)).item() / 448.0
                scale_v = torch.max(torch.abs(v_token)).item() / 448.0
                
                self.k_cache[:, self.current_pos] = (k_token / scale_k).to(torch.float8_e4m3fn)
                self.v_cache[:, self.current_pos] = (v_token / scale_v).to(torch.float8_e4m3fn)
                
                self.scales_k[self.current_pos] = scale_k
                self.scales_v[self.current_pos] = scale_v
                
                self.fp8_ratio = (self.fp8_ratio * self.total_stored + 1.0) / (self.total_stored + 1)
            else:
                
                self.k_cache[:, self.current_pos] = k_token.to(self.k_cache.dtype)
                self.v_cache[:, self.current_pos] = v_token.to(self.v_cache.dtype)
                
                self.scales_k[self.current_pos] = 1.0
                self.scales_v[self.current_pos] = 1.0
                
                self.fp8_ratio = (self.fp8_ratio * self.total_stored) / (self.total_stored + 1)
            
            self.current_pos += 1
            self.total_stored += 1
    
    def get_range(self, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
"""
        if end > self.current_pos and not self.is_full:
            end = self.current_pos
        
        k_slice = self.k_cache[:, start:end]  
        v_slice = self.v_cache[:, start:end]
        
        
        if HAS_FP8 and self.config.use_fp8:
            scales_k = self.scales_k[start:end].unsqueeze(0).unsqueeze(-1)
            scales_v = self.scales_v[start:end].unsqueeze(0).unsqueeze(-1)
            
            k_slice = k_slice.to(torch.float32) * scales_k
            v_slice = v_slice.to(torch.float32) * scales_v
        else:
            k_slice = k_slice.to(torch.float32)
            v_slice = v_slice.to(torch.float32)
        
        return k_slice, v_slice
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
"""
        if HAS_FP8 and self.config.use_fp8:
            
            kv_size = self.k_cache.numel() + self.v_cache.numel()
            scales_size = self.scales_k.numel() + self.scales_v.numel()
            total_bytes = kv_size + scales_size * 4  
        else:
            
            total_bytes = (self.k_cache.numel() + self.v_cache.numel()) * 2
        
        return {
            'total_mb': total_bytes / (1024 * 1024),
            'fp8_ratio': self.fp8_ratio,
            'compression_ratio': 2.0 if self.fp8_ratio > 0.5 else 1.0
        }

class FP8Optimizer:
    """
"""
    def __init__(self, optimizer: torch.optim.Optimizer, config: FP8Config = None):
        self.optimizer = optimizer
        self.config = config or FP8Config()
        self.scale_manager = FP8ScaleManager(config)
    
    def step(self, closure=None):
        """
"""
        
        overflow = False
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        overflow = True
                        break
            if overflow:
                break
        
        
        self.scale_manager.update_scales(forward_overflow=overflow)
        
        if not overflow:
            
            return self.optimizer.step(closure)
        else:
            
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.zero_()
            return None

def optimize_model_for_fp8(model: nn.Module, config: FP8Config = None) -> nn.Module:
    """
"""
    if config is None:
        config = FP8Config()
    
    def replace_linear(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                
                fp8_linear = FP8Linear(
                    child.in_features, 
                    child.out_features,
                    bias=child.bias is not None,
                    config=config
                )
                
                
                fp8_linear.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    fp8_linear.bias.data = child.bias.data.clone()
                
                setattr(module, name, fp8_linear)
            else:
                replace_linear(child)
    
    
    replace_linear(model)
    
    print("FP8-optimized model (RTX 4000 series)")
    print(f"   FP8 available: {HAS_FP8}")
    print(f"   Mode: {'FP8 native' if HAS_FP8 else 'FP16 fallback'}")
    
    return model

__all__ = [
    'FP8Config',
    'FP8Linear', 
    'FP8Attention',
    'FP8KVCache',
    'FP8Optimizer',
    'optimize_model_for_fp8',
    'to_fp8_e4m3',
    'to_fp8_e5m2'
]