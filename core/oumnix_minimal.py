"""
"""
# import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
# import numpy as np
from dataclasses import dataclass
# from collections import defaultdict

@dataclass
class OumnixMinimalConfig:
    """
"""
    vocab_size: int = 32000
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    head_dim: int = 64
    
    
    local_window: int = 128
    global_stride: int = 64
    moop_top_k: int = 2
    
    
    use_uncertainty: bool = True
    uncertainty_dim: int = 32
    
    
    use_weave: bool = True
    weave_rank: int = 64
    codebook_size: int = 256
    
    
    n_islets: int = 8
    islet_seed_dim: int = 16
    islet_rank: int = 8
    
    
    max_depth_steps: int = 2
    depth_threshold: float = 0.1
    
    
    hot_kv_size: int = 4096
    warm_kv_windows: int = 16
    context_tree_fanout: int = 8
    
    
    use_neurochemistry: bool = True
    
    
    use_fp8: bool = True
    fp8_e4m3_forward: bool = True
    fp8_e5m2_backward: bool = True
    fp8_dynamic_scaling: bool = True
    fp8_uncertainty_threshold: float = 0.1
    
    
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    def __post_init__(self):
        
        if self.head_dim * self.n_heads != self.dim:
            self.head_dim = max(1, self.dim // self.n_heads)

class PFPBlock(nn.Module):
    """
"""
    def __init__(self, dim: int, uncertainty_dim: int = 32):
        super().__init__()
        self.dim = dim
        self.uncertainty_dim = uncertainty_dim
        
        
        self.mu_proj = nn.Linear(dim, dim)
        self.sigma_proj = nn.Linear(dim, uncertainty_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
"""
        mu = self.mu_proj(x)
        log_sigma = self.sigma_proj(x)
        sigma = F.softplus(log_sigma) + 1e-6  
        
        return mu, sigma

class WEAVELayer(nn.Module):
    """
"""
    def __init__(self, in_features: int, out_features: int, rank: int = 64, 
                 codebook_size: int = 256, lora_rank: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        
        self.U = nn.Parameter(torch.randn(in_features, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(out_features, rank) * 0.02)
        
        
        self.alpha = nn.Parameter(torch.randn(rank) * 0.1)
        
        
        self.lora_A = nn.Parameter(torch.randn(in_features, lora_rank) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(lora_rank, out_features) * 0.01)
        
        
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        main = x @ self.U @ torch.diag(self.alpha) @ self.V.T
        
        
        lora = x @ self.lora_A @ self.lora_B
        
        return main + lora + self.bias

class IsletGenerator(nn.Module):
    """
"""
    def __init__(self, seed_dim: int = 16, target_dim: int = 768, rank: int = 8):
        super().__init__()
        self.seed_dim = seed_dim
        self.target_dim = target_dim
        self.rank = rank
        
        
        self.seed_to_delta = nn.Sequential(
            nn.Linear(seed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, rank * target_dim * 2),  
        )
        
    def forward(self, seed: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        """
"""
        delta_params = self.seed_to_delta(seed)
        
        
        half = delta_params.shape[-1] // 2
        delta_A = delta_params[:half].view(self.rank, -1)
        delta_B = delta_params[half:].view(-1, self.rank)
        
        
        delta = delta_A.T @ delta_B
        delta = delta.view_as(base_weight)
        
        return base_weight + delta * 0.1  

class MoOpGate(nn.Module):
    """
"""
    def __init__(self, dim: int, n_operators: int = 3, top_k: int = 2):
        super().__init__()
        self.n_operators = n_operators
        self.top_k = top_k
        
        self.gate = nn.Linear(dim, n_operators)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
"""
        logits = self.gate(x)  
        
        
        top_logits, top_indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(top_logits, dim=-1)
        
        return weights, top_indices

class LocalGlobalAttention(nn.Module):
    """
"""
    def __init__(self, config: OumnixMinimalConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        
        self.head_dim = config.dim // config.n_heads
        self.local_window = config.local_window
        self.scale = self.head_dim ** -0.5
        
        
        if config.use_fp8:
            from core.fp8_optimization import FP8Linear, FP8Config
            fp8_config = FP8Config(
                use_fp8=config.use_fp8,
                e4m3_for_forward=config.fp8_e4m3_forward,
                e5m2_for_backward=config.fp8_e5m2_backward,
                dynamic_scaling=config.fp8_dynamic_scaling,
                uncertainty_threshold=config.fp8_uncertainty_threshold
            )
            
            
            if config.use_weave:
                
                self.q_proj = FP8Linear(config.dim, config.dim, bias=False, config=fp8_config)
                self.k_proj = FP8Linear(config.dim, config.dim, bias=False, config=fp8_config)
                self.v_proj = FP8Linear(config.dim, config.dim, bias=False, config=fp8_config)
            else:
                self.q_proj = FP8Linear(config.dim, config.dim, bias=False, config=fp8_config)
                self.k_proj = FP8Linear(config.dim, config.dim, bias=False, config=fp8_config)
                self.v_proj = FP8Linear(config.dim, config.dim, bias=False, config=fp8_config)
                
            self.out_proj = FP8Linear(config.dim, config.dim, config=fp8_config)
            
        else:
            
            if config.use_weave:
                self.q_proj = WEAVELayer(config.dim, config.dim)
                self.k_proj = WEAVELayer(config.dim, config.dim)
                self.v_proj = WEAVELayer(config.dim, config.dim)
            else:
                self.q_proj = nn.Linear(config.dim, config.dim)
                self.k_proj = nn.Linear(config.dim, config.dim)
                self.v_proj = nn.Linear(config.dim, config.dim)
                
            self.out_proj = nn.Linear(config.dim, config.dim)
            
        self.dropout = nn.Dropout(config.dropout)
        
        
        self.temporal_head = nn.Linear(self.head_dim, 2)  
        self.identity_head = nn.Linear(self.head_dim, 64)  
        self.citation_head = nn.Linear(self.head_dim, 32)  
        
        # OMNX deep metrics holders
        self.last_kv_hit: float = 0.0
        self.last_head_drop: float = 0.0
        
    def forward(self, x: torch.Tensor, 
                kv_cache: Optional[Dict] = None,
                memory_vectors: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                uncertainty: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        B, N, D = x.shape
        
        
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        
        local_attn = self._local_attention_fp8(q, k, v, mask, uncertainty)   
        
        
        global_attn = self._global_attention_fp8(q, k, v, mask, uncertainty) 
        
        
        if memory_vectors is not None:
            rag_attn = self._rag_attention_fp8(q, memory_vectors, uncertainty)
        else:
            rag_attn = torch.zeros_like(local_attn)
        
        
        combined = local_attn + global_attn + rag_attn
        # OMNX deep metrics: kv hit (rag mass ratio) and head drop (heads with low max)
        try:
            if memory_vectors is not None and memory_vectors.numel() > 0:
                mass_mem = rag_attn.abs().sum().item()
                mass_all = (local_attn.abs().sum() + global_attn.abs().sum() + rag_attn.abs().sum()).item()
                self.last_kv_hit = float(mass_mem / max(mass_all, 1e-12))
            else:
                self.last_kv_hit = 0.0
            max_attn = combined.amax(dim=-1)  # [B,H,N]
            self.last_head_drop = float((max_attn < 0.5).float().mean().item())
        except Exception:
            pass
        
        
        last_head = combined[:, 0, :, :]
        temporal_info = self.temporal_head(last_head)   
        identity_info = self.identity_head(last_head)   
        citation_info = self.citation_head(last_head)   
        
        
        fused = combined.transpose(1, 2).contiguous().view(B, N, self.n_heads * self.head_dim)
        out = self.out_proj(fused)
        
        
        aux_info = {
            'temporal': temporal_info,
            'identity': identity_info,
            'citation': citation_info,
            
            'kv_cache': {'k': k.detach(), 'v': v.detach()},
            'fp8_usage': self._get_fp8_stats(uncertainty) if self.config.use_fp8 else None
        }
        
        return out, aux_info
    
    def _local_attention_fp8(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                            mask: Optional[torch.Tensor] = None,
                            uncertainty: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
"""
        B, H, N, D = q.shape
        
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        
        device = q.device
        i = torch.arange(N, device=device).unsqueeze(1)
        j = torch.arange(N, device=device).unsqueeze(0)
        local_keep = (j - i).abs() <= self.local_window
        local_mask = ~local_keep  
        attn = attn.masked_fill(local_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        
        if mask is not None:
            attn = attn.masked_fill(~mask[:, None, None, :], float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)  
        return out
    
    def _global_attention_fp8(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                             mask: Optional[torch.Tensor] = None,
                             uncertainty: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
"""
        B, H, N, D = q.shape
        stride = max(1, self.config.global_stride)
        k_sparse = k[:, :, ::stride, :]  
        v_sparse = v[:, :, ::stride, :]
        
        attn = torch.matmul(q, k_sparse.transpose(-2, -1)) * self.scale  
        if mask is not None:
            attn = attn.masked_fill(~mask[:, None, None, ::stride], float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v_sparse)  
        return out
    
    def _rag_attention_fp8(self, q: torch.Tensor, memory_vectors: torch.Tensor,
                          uncertainty: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
"""
        B, H, N, D = q.shape
        if memory_vectors is None or memory_vectors.numel() == 0:
            return torch.zeros_like(q)
        
        mem = memory_vectors.unsqueeze(1).expand(-1, H, -1, -1)
        attn = torch.matmul(q, mem.transpose(-2, -1)) * self.scale  
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, mem)  
        return out
    
    def _get_fp8_stats(self, uncertainty: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
"""
        if uncertainty is None:
            return {
                'fp8_usage_ratio': 1.0,
                'high_precision_tokens': 0,
                'avg_uncertainty': 0.0
            }
        
        high_uncertainty = uncertainty > self.config.fp8_uncertainty_threshold
        total_tokens = uncertainty.numel()
        high_precision_tokens = high_uncertainty.sum().item()
        
        return {
            'fp8_usage_ratio': 1.0 - (high_precision_tokens / total_tokens),
            'high_precision_tokens': high_precision_tokens,
            'avg_uncertainty': uncertainty.mean().item(),
            'max_uncertainty': uncertainty.max().item()
        }

class SSMBlock(nn.Module):
    """
"""
    def __init__(self, config: OumnixMinimalConfig):
        super().__init__()
        self.dim = config.dim
        self.d_state = config.dim // 4
        
        
        
        
        self.A = nn.Parameter(torch.randn(self.d_state, self.d_state) * 0.02)
        self.B = nn.Parameter(torch.randn(self.dim, self.d_state) * 0.02)
        self.C = nn.Parameter(torch.randn(self.d_state, self.dim) * 0.02)
        self.D = nn.Parameter(torch.randn(self.dim) * 0.02)
        
        
        self.in_proj = nn.Linear(config.dim, config.dim * 2)
        self.out_proj = nn.Linear(config.dim, config.dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        
        
        x_proj = self.in_proj(x)
        x_gate, x_input = x_proj.chunk(2, dim=-1)
        x_gate = F.sigmoid(x_gate)
        
        
        h = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for t in range(N):
            
            h = h @ self.A.T + x_input[:, t] @ self.B
            
            
            y = h @ self.C + x_input[:, t] * self.D
            outputs.append(y)
        
        
        output = torch.stack(outputs, dim=1)  
        
        
        output = output * x_gate
        output = self.out_proj(output)
        
        return output

class ConvolutionBlock(nn.Module):
    """
"""
    def __init__(self, config: OumnixMinimalConfig):
        super().__init__()
        self.dim = config.dim
        kernel_size = 7
        
        
        self.conv = nn.Conv1d(
            config.dim, config.dim, 
            kernel_size=kernel_size,
            padding=kernel_size//2,
            groups=config.dim
        )
        
        
        self.in_proj = nn.Linear(config.dim, config.dim * 2)
        self.out_proj = nn.Linear(config.dim, config.dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        
        
        x_proj = self.in_proj(x)
        x_gate, x_input = x_proj.chunk(2, dim=-1)
        x_gate = F.sigmoid(x_gate)
        
        
        x_conv = x_input.transpose(1, 2)  
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1, 2)  
        
        
        output = x_conv * x_gate
        output = self.out_proj(output)
        
        return output

class OumnixCell(nn.Module):
    """
"""
    def __init__(self, config: OumnixMinimalConfig):
        super().__init__()
        self.dim = config.dim
        
        
        self.hypothesis_gen_1 = nn.Linear(config.dim, config.dim)
        self.hypothesis_gen_2 = nn.Linear(config.dim, config.dim)
        
        
        self.verifier = nn.Sequential(
            nn.Linear(config.dim * 2, config.dim),
            nn.ReLU(),
            nn.Linear(config.dim, 2)  
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        h1 = self.hypothesis_gen_1(x)
        h2 = self.hypothesis_gen_2(x)
        
        
        combined = torch.cat([h1, h2], dim=-1)
        choice_logits = self.verifier(combined)
        choice_probs = F.softmax(choice_logits, dim=-1)
        
        
        output = choice_probs[:, :, 0:1] * h1 + choice_probs[:, :, 1:2] * h2
        
        return output

class OumnixMinimalBlock(nn.Module):
    """
"""
    def __init__(self, config: OumnixMinimalConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        
        if config.use_uncertainty:
            self.pfp = PFPBlock(config.dim, config.uncertainty_dim)
        
        
        self.moop_gate = MoOpGate(config.dim, n_operators=3, top_k=config.moop_top_k)
        
        
        self.attention = LocalGlobalAttention(config)
        self.ssm = SSMBlock(config)
        self.conv = ConvolutionBlock(config)
        
        
        if layer_idx % 2 == 1:
            self.oumnix_cell = OumnixCell(config)
        else:
            self.oumnix_cell = None
        
        
        self.islet_seeds = nn.Parameter(torch.randn(config.n_islets, config.islet_seed_dim) * 0.02)
        self.islet_generator = IsletGenerator(config.islet_seed_dim, config.dim, config.islet_rank)
        
        
        self.norm1 = nn.LayerNorm(config.dim, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.dim, eps=config.layer_norm_eps)
        self.norm3 = nn.LayerNorm(config.dim, eps=config.layer_norm_eps)
        
        
        self.depth_controller = nn.Linear(config.dim, 1)
        
    def forward(self, x: torch.Tensor, 
                neuro_state: Optional[Dict] = None,
                kv_cache: Optional[Dict] = None,
                memory_vectors: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        
        residual = x
        
        
        if self.config.use_uncertainty:
            mu, sigma = self.pfp(x)
            x = mu  
        else:
            sigma = None
        
        
        x = self.norm1(x)
        
        
        gate_weights, gate_indices = self.moop_gate(x)
        
        
        attn_out, attn_aux = self.attention(x, kv_cache, memory_vectors, mask)
        ssm_out = self.ssm(x)
        conv_out = self.conv(x)
        operator_outputs = [attn_out, ssm_out, conv_out]
        
        
        combined_output = torch.zeros_like(x)
        top_k = gate_indices.shape[-1]
        for k in range(top_k):
            op_idx = gate_indices[..., k]            
            weight = gate_weights[..., k].unsqueeze(-1)  
            for j, op_out in enumerate(operator_outputs):
                mask_j = (op_idx == j).unsqueeze(-1).float()  
                combined_output = combined_output + weight * mask_j * op_out
        
        
        x = residual + combined_output
        
        
        if self.oumnix_cell is not None:
            x_oumnix = self.oumnix_cell(self.norm2(x))
            x = x + x_oumnix
        
        
        
        
        
        
        
        
        
        
        
        
        depth_score = torch.sigmoid(self.depth_controller(x.mean(dim=1)))  
        should_continue = depth_score > self.config.depth_threshold
        
        
        aux_info = {
            'sigma': sigma,
            'gate_weights': gate_weights,
            'gate_indices': gate_indices,
            'depth_score': depth_score,
            'should_continue': should_continue,
            **attn_aux
        }
        
        return x, aux_info

class OumnixMinimal(nn.Module):
    """
"""
    def __init__(self, config: OumnixMinimalConfig):
        super().__init__()
        self.config = config
        
        
        self.token_embed = nn.Embedding(config.vocab_size, config.dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 8192, config.dim) * 0.02)
        
        
        self.blocks = nn.ModuleList([
            OumnixMinimalBlock(config, i) for i in range(config.n_layers)
        ])
        
        
        self.norm = nn.LayerNorm(config.dim, eps=config.layer_norm_eps)
        
        
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, 
                input_ids: torch.Tensor,
                neuro_state: Optional[Dict] = None,
                memory_vectors: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                past_key_values: Optional[List[Dict]] = None) -> Dict:
        
        B, N = input_ids.shape
        device = input_ids.device
        
        
        x = self.token_embed(input_ids)
        
        
        pos_ids = torch.arange(N, device=device).unsqueeze(0)
        pos_ids = torch.clamp(pos_ids, 0, self.pos_embed.shape[1] - 1)
        x = x + self.pos_embed[:, :N, :]
        
        
        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)
        
        new_key_values = []
        all_aux_info = []
        
        
        for i, (block, past_kv) in enumerate(zip(self.blocks, past_key_values)):
            x, aux_info = block(
                x, 
                neuro_state=neuro_state,
                kv_cache=past_kv,
                memory_vectors=memory_vectors,
                mask=attention_mask
            )
            
            all_aux_info.append(aux_info)
            
            if use_cache:
                new_key_values.append(aux_info.get('kv_cache'))
            
            
            if not self.training and aux_info.get('should_continue') is not None:
                if not aux_info['should_continue'].any():
                    break  
        
        
        x = self.norm(x)
        
        
        logits = self.lm_head(x)
        
        return {
            'logits': logits,
            'past_key_values': new_key_values if use_cache else None,
            'aux_info': all_aux_info
        }


def create_oumnix_minimal(vocab_size: int = 32000, **kwargs) -> OumnixMinimal:
    """
"""
    config = OumnixMinimalConfig(
        vocab_size=vocab_size,
        dim=768,  
        n_layers=12,
        n_heads=12,
        **kwargs
    )
    return OumnixMinimal(config)

__all__ = [
    'OumnixMinimal', 
    'OumnixMinimalConfig', 
    'create_oumnix_minimal'
]
