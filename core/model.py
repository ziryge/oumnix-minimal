import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary position embedding to query and key tensors."""
    if position_ids is None:
        # Create position ids for the sequence
        seq_len = q.shape[-2]
        position_ids = torch.arange(seq_len, device=q.device).unsqueeze(0)
    
    # Ensure position_ids is within bounds
    position_ids = position_ids % cos.shape[-2]
    
    cos = cos[position_ids].unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin = sin[position_ids].unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute the frequency tensor for complex exponentials."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)

class SwiGLU(nn.Module):
    """SwiGLU activation function."""
    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2.67 * dim)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class LocalGlobalAttention(nn.Module):
    """Modern causal local-global attention with RoPE, ALiBi, multi-query attention, and RAG fusion.

    Features:
    - Rotary Position Embedding (RoPE) for better position encoding
    - ALiBi (Attention with Linear Biases) for extrapolation
    - Multi-Query Attention for efficiency
    - Flash Attention-style optimizations
    - RAG fusion and token-flow gating
    """
    def __init__(self, dim: int, heads: int = 8, local_window: int = 64, dropout: float = 0.0, 
                 use_rag: bool = False, rag_topk: int = 8, enable_token_flow: bool = False, 
                 token_flow_thresh: float = 0.0, use_rope: bool = True, use_alibi: bool = True,
                 use_mqa: bool = False, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.local_window = local_window
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope
        self.use_alibi = use_alibi
        self.use_mqa = use_mqa
        self.max_seq_len = max_seq_len
        
        # Multi-query attention: fewer key/value heads
        if use_mqa:
            self.kv_heads = max(1, heads // 4)  # 4x fewer KV heads
            self.kv_dim = self.kv_heads * self.head_dim
        else:
            self.kv_heads = heads
            self.kv_dim = dim
        
        # Projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.kv_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self.dropout = nn.Identity() if dropout <= 0.0 else nn.Dropout(dropout)
        
        # RAG components
        self.use_rag = use_rag
        self.rag_topk = rag_topk
        self.enable_token_flow = enable_token_flow
        self.token_flow_thresh = token_flow_thresh
        self.rag_provider = None
        self.mem_k = nn.Linear(dim, self.kv_dim)
        self.mem_v = nn.Linear(dim, self.kv_dim)
        
        # RoPE
        if use_rope:
            self.freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len)
        
        # ALiBi
        if use_alibi:
            self.alibi_slopes = self._get_alibi_slopes(heads)
        
        # Metrics
        self.last_kv_hit = 0.0
        self.last_head_drop = 0.0
    
    def _get_alibi_slopes(self, n_heads: int) -> torch.Tensor:
        """Get ALiBi slopes for each head."""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(n_heads).is_integer():
            slopes = get_slopes_power_of_2(n_heads)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            slopes.extend(get_slopes_power_of_2(2*closest_power_of_2)[0::2][:n_heads-closest_power_of_2])
        
        return torch.tensor(slopes, dtype=torch.float32)

    def set_rag_provider(self, provider):
        self.rag_provider = provider

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, 
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Modern attention with RoPE, ALiBi, multi-query attention, and optimizations."""
        B, N, C = x.shape
        base_N = N
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        k = self.k_proj(x).view(B, N, self.kv_heads, self.head_dim).transpose(1, 2)  # [B, KV_H, N, D]
        v = self.v_proj(x).view(B, N, self.kv_heads, self.head_dim).transpose(1, 2)  # [B, KV_H, N, D]
        
        # Apply RoPE
        if self.use_rope:
            cos = self.freqs_cis[:N, :self.head_dim//2].real.to(x.device)
            sin = self.freqs_cis[:N, :self.head_dim//2].imag.to(x.device)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        # Multi-query attention: repeat K, V for all query heads
        if self.use_mqa and self.kv_heads < self.heads:
            k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
            v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply ALiBi
        if self.use_alibi:
            alibi_slopes = self.alibi_slopes.to(x.device)
            # Create distance matrix
            seq_len = scores.size(-1)
            distance = torch.arange(seq_len, device=x.device).unsqueeze(0) - torch.arange(seq_len, device=x.device).unsqueeze(1)
            distance = distance.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
            alibi_bias = distance * alibi_slopes.view(1, -1, 1, 1)
            scores = scores + alibi_bias
        
        # RAG fusion
        if self.use_rag and self.rag_provider is not None:
            with torch.no_grad():
                mem = self.rag_provider(x)
            if mem is not None and isinstance(mem, torch.Tensor) and mem.dim() == 3 and mem.size(0) == B and mem.size(2) == C:
                mk = self.mem_k(mem)
                mv = self.mem_v(mem)
                mk = mk.view(B, -1, self.kv_heads, self.head_dim).transpose(1, 2)  # [B, KV_H, K, D]
                mv = mv.view(B, -1, self.kv_heads, self.head_dim).transpose(1, 2)  # [B, KV_H, K, D]
                
                # Multi-query attention: repeat memory K, V for all query heads
                if self.use_mqa and self.kv_heads < self.heads:
                    mk = mk.repeat_interleave(self.heads // self.kv_heads, dim=1)
                    mv = mv.repeat_interleave(self.heads // self.kv_heads, dim=1)
                
                k = torch.cat([k, mk], dim=2)
                v = torch.cat([v, mv], dim=2)
                extra = mk.size(2)
                extra_mask = torch.ones(B, extra, dtype=torch.bool, device=x.device)
                if mask is not None:
                    mask = torch.cat([mask, extra_mask], dim=1)
                else:
                    mask = torch.cat([torch.ones(B, N, dtype=torch.bool, device=x.device), extra_mask], dim=1)
                
                # Recompute scores with extended K
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                N = k.size(2)
        # Apply masks
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        
        # Causal mask
        causal = torch.tril(torch.ones(scores.size(-2), scores.size(-1), device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal[None, None, :, :], float("-inf"))
        
        # Local attention with window
        if self.local_window and self.local_window > 0:
            Nq = scores.size(-2)
            Nk = scores.size(-1)
            idx_i = torch.arange(Nq, device=x.device).view(Nq, 1)
            idx_j = torch.arange(base_N, device=x.device).view(1, base_N)
            band_base = (idx_i - idx_j).abs() > self.local_window  # [Nq, base_N]
            if Nk > base_N:
                pad = torch.zeros(Nq, Nk - base_N, dtype=torch.bool, device=x.device)
                band = torch.cat([band_base, pad], dim=1)
            else:
                band = band_base[:, :Nk]
            scores_local = scores.masked_fill(band[None, None, :, :], float("-inf"))
        else:
            scores_local = scores
        
        # Compute local attention
        attn_local = torch.softmax(scores_local, dim=-1)
        
        # Update metrics
        if self.use_rag and 'mv' in locals():
            Nk = attn_local.size(-1)
            Kmem = mv.size(2) if isinstance(mv, torch.Tensor) else 0
            if Kmem > 0 and Nk >= Kmem:
                mass_mem = attn_local[..., -Kmem:].sum().item()
                mass_all = attn_local.sum().item()
                self.last_kv_hit = float(mass_mem / max(mass_all, 1e-12))
        
        max_attn = attn_local.amax(dim=-1)  # [B,H,N]
        self.last_head_drop = float((max_attn < 0.5).float().mean().item())
        
        attn_local = self.dropout(attn_local)
        
        # Token flow gating
        if self.enable_token_flow and self.token_flow_thresh > 0.0:
            p = attn_local.max(dim=-1).values.mean(dim=1, keepdim=True)  
            mask_tokens = (p < self.token_flow_thresh)  
            v = v.masked_fill(mask_tokens[:, :, :, None], 0.0)
        
        out_local = torch.matmul(attn_local, v)
        
        # Global attention with subsampling
        stride = max(1, int(self.local_window))
        k_global = k[:, :, ::stride, :]
        v_global = v[:, :, ::stride, :]
        scores_global = torch.matmul(q, k_global.transpose(-2, -1)) * self.scale
        
        Nq = q.size(2)
        Nk_g = k_global.size(2)
        causal_g = torch.ones(1, 1, Nq, Nk_g, dtype=torch.bool, device=x.device)
        scores_global = scores_global.masked_fill(~causal_g, float("-inf"))
        attn_global = torch.softmax(scores_global, dim=-1)
        out_global = torch.matmul(attn_global, v_global)
        
        # Combine local and global
        out = out_local + out_global
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, base_N, C)
        return self.out_proj(out)

class SSMBlock(nn.Module):
    """Modern State Space Model block with selective mechanisms and better parameterization."""
    def __init__(self, dim: int, d_state: int = 16, dt_rank: int = None, dt_min: float = 0.001, 
                 dt_max: float = 0.1, dt_init: str = "random", dt_scale: float = 1.0, 
                 dt_init_floor: float = 1e-4, conv_bias: bool = True, bias: bool = False):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else max(16, dim // 16)
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        
        # Input projection
        self.in_proj = nn.Linear(dim, dim * 2, bias=bias)
        
        # Conv1d for local dependencies
        self.conv1d = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            bias=conv_bias,
            kernel_size=3,
            groups=dim,
            padding=1,
        )
        
        # State space parameters
        self.A_log = nn.Parameter(torch.randn(dim, d_state) * 0.02)
        self.D = nn.Parameter(torch.randn(dim) * 0.02)
        
        # Input-dependent parameters
        self.x_proj = nn.Linear(dim, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, dim, bias=True)
        
        # Initialize dt projection
        dt_init_std = self.dt_rank**-0.5 * self.dt_scale
        with torch.no_grad():
            dt = torch.exp(
                torch.rand(dim) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)
            ).clamp_(min=self.dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.weight.copy_(inv_dt.unsqueeze(1).repeat(1, self.dt_rank) * dt_init_std)
            self.dt_proj.bias.copy_(inv_dt)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        Returns: same shape as x
        """
        B, L, D = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # (B, L, 2*D)
        x, z = xz.chunk(2, dim=-1)  # each (B, L, D)
        
        # Conv1d
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.conv1d(x)[:, :, :L]  # (B, D, L)
        x = x.transpose(1, 2)  # (B, L, D)
        
        # State space parameters
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt, B_param, C_param = x_dbl.split(split_size=[self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # Discretize
        dt = F.softplus(self.dt_proj(dt))  # (B, L, D)
        
        # Get A, B, C
        A = -torch.exp(self.A_log.float())  # (D, d_state)
        B = B_param.float()  # (B, L, d_state)
        C = C_param.float()  # (B, L, d_state)
        
        # Selective scan
        y = self.selective_scan(x, dt, A, B, C, D=self.D.float())
        
        # Output projection
        y = y * F.silu(z)
        out = self.out_proj(y)
        
        return out
    
    def selective_scan(self, u, delta, A, B, C, D):
        """Selective scan algorithm."""
        B, L, D = u.shape
        N = A.shape[1]
        
        # Discretize
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, D, d_state)
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # (B, L, D, d_state)
        
        # Scan
        x = torch.zeros((B, D, N), device=u.device, dtype=u.dtype)
        ys = []
        for i in range(L):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = (x @ C[:, i].unsqueeze(-1)).squeeze(-1)  # (B, D)
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # (B, L, D)
        y = y + u * D
        return y

class LoRALayer(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.lora_A = nn.Parameter(torch.randn(base_layer.in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(r, base_layer.out_features) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A @ self.lora_B) * (self.alpha / self.r)

class TokenOperatorMixer(nn.Module):
    def __init__(self, dim: int, n_ops: int = 2, temperature: float = 1.0):
        super().__init__()
        self.proj = nn.Linear(dim, n_ops)
        self.temperature = temperature
        self.last_entropy = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.proj(x) / max(self.temperature, 1e-6)
        w = torch.softmax(logits, dim=-1)
        p = w.clamp_min(1e-12)
        self.last_entropy = -(p * p.log()).sum(dim=-1).mean()
        return w

class WeaveLinear(nn.Module):
    def __init__(self, base_layer: nn.Module):
        super().__init__()
        self.base = base_layer
        if hasattr(base_layer, 'out_features'):
            out_feats = base_layer.out_features
        elif hasattr(base_layer, 'base') and hasattr(base_layer.base, 'out_features'):
            out_feats = base_layer.base.out_features
        else:
            raise ValueError('WeaveLinear expects a linear-like module with out_features')
        self.gate = nn.Parameter(torch.zeros(out_feats))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        g = torch.sigmoid(self.gate).view(1, 1, -1)
        return y * g

class DepthwiseConv1d(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return y

class OumnixSimpleAI(nn.Module):
    """Modern Oumnix model with transformer-equivalent improvements."""
    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        use_moop: bool = False,
        early_exit: bool = False,
        exit_threshold: float = 0.0,
        moop_temperature: float = 1.0,
        moop_entropy_reg: float = 0.0,
        use_bayesian_residuals: bool = False,
        residual_std: float = 0.0,
        residual_stochastic: bool = False,
        residual_prob: float = 0.0,
        use_weave: bool = False,
        use_islet_injection: bool = False,
        islet_capacity: int = 64,
        use_cell: bool = False,
        cell_threshold: float = 0.5,
        use_aux_heads: bool = False,
        use_rope: bool = True,
        use_alibi: bool = True,
        use_mqa: bool = False,
        use_rmsnorm: bool = True,
        use_swiglu: bool = True,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.use_moop = use_moop
        self.early_exit = early_exit
        self.exit_threshold = exit_threshold
        self.moop_entropy_reg = moop_entropy_reg
        self.use_bayesian_residuals = use_bayesian_residuals
        self.residual_std = residual_std
        self.residual_stochastic = residual_stochastic
        self.residual_prob = residual_prob
        self.use_weave = use_weave
        self.use_islet_injection = use_islet_injection
        self.use_cell = use_cell
        self.cell_threshold = cell_threshold
        self.use_aux_heads = use_aux_heads
        self.max_seq_len = max_seq_len
        
        # Embeddings with better initialization
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)
        
        # Normalization
        norm_class = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm = norm_class(dim)
        
        # Layers
        self.layers = nn.ModuleList()
        self.islet_cache = None
        if self.use_islet_injection:
            from collections import OrderedDict
            self.islet_cache = OrderedDict()
            self.islet_capacity = islet_capacity
            
        for i in range(n_layers):
            # Modern attention with RoPE, ALiBi, MQA
            att = LocalGlobalAttention(
                dim=dim, 
                heads=n_heads, 
                use_rope=use_rope, 
                use_alibi=use_alibi, 
                use_mqa=use_mqa,
                max_seq_len=max_seq_len
            )
            
            # Modern SSM with selective mechanisms
            ssm = SSMBlock(dim, d_state=dim//4)
            
            # Layer normalization
            att_norm = norm_class(dim)
            ssm_norm = norm_class(dim)
            
            if self.use_moop:
                mixer = TokenOperatorMixer(dim, n_ops=3, temperature=moop_temperature)
                dw = DepthwiseConv1d(dim)
                dw_norm = norm_class(dim)
                
                if self.use_cell:
                    # Modern MLP with SwiGLU
                    if use_swiglu:
                        mlp = SwiGLU(dim)
                    else:
                        mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
                    verifier = nn.Linear(dim, 1)
                    mlp_norm = norm_class(dim)
                    self.layers.append(nn.ModuleList([att, att_norm, ssm, ssm_norm, mixer, dw, dw_norm, mlp, mlp_norm, verifier]))
                else:
                    self.layers.append(nn.ModuleList([att, att_norm, ssm, ssm_norm, mixer, dw, dw_norm]))
            else:
                if self.use_cell:
                    if use_swiglu:
                        mlp = SwiGLU(dim)
                    else:
                        mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
                    verifier = nn.Linear(dim, 1)
                    mlp_norm = norm_class(dim)
                    self.layers.append(nn.ModuleList([att, att_norm, ssm, ssm_norm, mlp, mlp_norm, verifier]))
                else:
                    self.layers.append(nn.ModuleList([att, att_norm, ssm, ssm_norm]))
        
        # Output head with better initialization
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embed.weight
        
        if self.use_aux_heads:
            self.aux_temporal = nn.Linear(dim, dim)
            self.aux_identity = nn.Linear(dim, dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Modern weight initialization."""
        if isinstance(module, nn.Linear):
            # Xavier/Glorot initialization for linear layers
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Normal initialization for embeddings
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            # Standard initialization for normalization
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv1d):
            # Kaiming initialization for conv layers
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _apply_islet(self, ids: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.islet_cache is None or ids is None or ids.dim() != 2:
            return x
        b, n = ids.shape
        for bi in range(b):
            if n < 2:
                continue
            key = (int(ids[bi, -2].item()), int(ids[bi, -1].item()))
            vec = x[bi, -1]
            self.islet_cache[key] = vec.detach()
            while len(self.islet_cache) > self.islet_capacity:
                self.islet_cache.popitem(last=False)
            val = self.islet_cache.get(key)
            if val is not None:
                x[bi, -1] = x[bi, -1] + val.to(x.device)
        return x

    def forward(self, ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through stacked operator blocks with optional gating, residual noise, islet injection, and early-exit."""
        x = self.embed(ids)
        for layer in self.layers:
            if self.use_moop and (len(layer) == 4 or len(layer) == 6):
                if len(layer) == 6:
                    att, ssm, mixer, dw, mlp, verifier = layer
                else:
                    att, ssm, mixer, dw = layer
                    mlp = None
                    verifier = None
                a = att(x, mask)
                s = ssm(x)
                c = dw(x)
                w = mixer(x)
                wa = w[..., 0:1]
                ws = w[..., 1:2]
                wc = w[..., 2:3]
                mix = wa * a + ws * s + wc * c
                if mlp is not None and verifier is not None and self.use_cell:
                    branch = mlp(x)
                    score = torch.sigmoid(verifier(branch.mean(dim=1)))
                    gate = (score > self.cell_threshold).float().view(-1, 1, 1)
                    x = x + mix * gate + a * (1 - gate)
                else:
                    x = x + mix
            else:
                if len(layer) == 4:
                    att, ssm, mlp, verifier = layer
                    a = att(x, mask)
                    s = ssm(x)
                    branch = mlp(x)
                    score = torch.sigmoid(verifier(branch.mean(dim=1)))
                    gate = (score > self.cell_threshold).float().view(-1, 1, 1)
                    x = x + a + s
                    x = x + branch * gate
                else:
                    att, ssm = layer
                    x = x + att(x, mask)
                    x = x + ssm(x)
            if self.use_bayesian_residuals and self.residual_std > 0.0 and self.training:
                if (not self.residual_stochastic) or (torch.rand(()) < self.residual_prob):
                    noise = torch.randn_like(x) * self.residual_std
                    x = x + noise
            if self.use_islet_injection:
                x = self._apply_islet(ids, x)
            if self.early_exit and self.exit_threshold > 0.0:
                logits_tmp = self.lm_head(x)
                last = logits_tmp[:, -1]
                p = torch.softmax(last, dim=-1)
                ent = -(p * torch.clamp(p, min=1e-12).log()).sum(dim=-1).mean()
                if ent.item() < self.exit_threshold:
                    return logits_tmp
        logits = self.lm_head(x)
        return logits

__all__ = [
    "LocalGlobalAttention",
    "SSMBlock",
    "LoRALayer",
    "OumnixSimpleAI",
]
