"""
"""
import math
import torch
import torch.nn as nn




class LocalGlobalAttention(nn.Module):
    """
"""
    def __init__(self, dim: int, heads: int = 8, local_window: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.local_window = local_window
        self.scale = (dim // heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        q, k, v = qkv.unbind(2)  

        
        pad = self.local_window // 2
        
        q_padded = torch.nn.functional.pad(q, (0, 0, pad, pad), value=-1e9)
        local_q = q_padded.unfold(1, self.local_window, 1)  
        
        attn_local = (local_q @ k.transpose(-2, -1)) * self.scale  
        if mask is not None:
            
            mask_exp = ~mask[:, None, None, :, None]
            attn_local = attn_local.masked_fill(mask_exp, -1e9)
        attn_local = torch.softmax(attn_local, dim=-2)
        attn_local = self.dropout(attn_local)
        out_local = (attn_local @ v).sum(dim=-2)  

        
        
        k_global = k[:, ::self.local_window, :, :]  
        v_global = v[:, ::self.local_window, :, :]
        attn_global = (q @ k_global.transpose(-2, -1)) * self.scale  
        attn_global = torch.softmax(attn_global, dim=-1)
        out_global = (attn_global @ v_global)  

        
        out = out_local + out_global  
        out = out.reshape(B, N, C)
        return self.out(out)




class SSMBlock(nn.Module):
    """
"""
    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.A = nn.Parameter(torch.randn(dim, d_state))
        self.D = nn.Parameter(torch.randn(dim, d_state))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        B, N, C = x.shape
        
        x_state = x @ self.A  
        
        x_state = torch.cumsum(x_state, dim=1)
        
        out = x_state @ self.D  
        return out




class LoRALayer(nn.Module):
    """
"""
    def __init__(self, base_layer: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base = base_layer
        self.r = r
        self.alpha = alpha
        
        self.lora_A = nn.Parameter(torch.randn(base_layer.in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(r, base_layer.out_features) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A @ self.lora_B) * (self.alpha / self.r)




class OumnixSimpleAI(nn.Module):
    """
"""
    def __init__(self, vocab_size: int, dim: int = 256, n_layers: int = 6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            att = LocalGlobalAttention(dim)
            ssm = SSMBlock(dim)
            
            att.qkv = LoRALayer(att.qkv, r=8, alpha=2.0)
            att.out = LoRALayer(att.out, r=8, alpha=2.0)
            self.layers.append(nn.ModuleList([att, ssm]))
        
        self.lm_head = LoRALayer(nn.Linear(dim, vocab_size), r=8, alpha=1.5)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        
        x = self.embed(ids)
        for att, ssm in self.layers:
            x = x + att(x, mask)
            x = x + ssm(x)
        logits = self.lm_head(x)
        return logits




__all__ = [
    "LocalGlobalAttention",
    "SSMBlock",
    "LoRALayer",
    "OumnixSimpleAI",
]
