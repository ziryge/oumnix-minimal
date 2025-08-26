import torch
import torch.nn as nn

class LocalGlobalAttention(nn.Module):
    """Causal local-global attention with optional RAG fusion and token-flow gating.

    Metrics (last forward):
    - last_kv_hit: fraction of attention mass assigned to extra RAG keys (0..1)
    - last_head_drop: fraction of heads whose max attn prob < 0.5 (0..1)
    """
    def __init__(self, dim: int, heads: int = 8, local_window: int = 64, dropout: float = 0.0, use_rag: bool = False, rag_topk: int = 8, enable_token_flow: bool = False, token_flow_thresh: float = 0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.local_window = local_window
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Identity() if dropout <= 0.0 else nn.Dropout(dropout)
        self.use_rag = use_rag
        self.rag_topk = rag_topk
        self.enable_token_flow = enable_token_flow
        self.token_flow_thresh = token_flow_thresh
        self.rag_provider = None
        self.mem_k = nn.Linear(dim, dim)
        self.mem_v = nn.Linear(dim, dim)
        self.last_kv_hit = 0.0
        self.last_head_drop = 0.0

    def set_rag_provider(self, provider):
        self.rag_provider = provider

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute attention with causal banding, local-global mixing, optional RAG fusion and token-flow gating."""
        B, N, C = x.shape
        base_N = N
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        q, k, v = qkv.unbind(2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.use_rag and self.rag_provider is not None:
            with torch.no_grad():
                mem = self.rag_provider(x)
            if mem is None:
                pass
            elif isinstance(mem, torch.Tensor) and mem.dim() == 3 and mem.size(0) == B and mem.size(2) == C:
                mk = self.mem_k(mem)
                mv = self.mem_v(mem)
                mk = mk.view(B, -1, self.heads, C // self.heads).permute(0, 2, 1, 3)
                mv = mv.view(B, -1, self.heads, C // self.heads).permute(0, 2, 1, 3)
                k = torch.cat([k, mk], dim=2)
                v = torch.cat([v, mv], dim=2)
                extra = mk.size(2)
                extra_mask = torch.ones(B, extra, dtype=torch.bool, device=x.device)
                if mask is not None:
                    mask = torch.cat([mask, extra_mask], dim=1)
                else:
                    mask = torch.cat([torch.ones(B, N, dtype=torch.bool, device=x.device), extra_mask], dim=1)
            B2, H, N2, Dh = q.size(0), q.size(1), k.size(2), q.size(3)
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            N = N2
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        causal = torch.tril(torch.ones(scores.size(-2), scores.size(-1), device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal[None, None, :, :], float("-inf"))
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
        attn_local = torch.softmax(scores_local, dim=-1)
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
        if self.enable_token_flow and self.token_flow_thresh > 0.0:
            p = attn_local.max(dim=-1).values.mean(dim=1, keepdim=True)  
            mask_tokens = (p < self.token_flow_thresh)  
            v = v.masked_fill(mask_tokens[:, :, :, None], 0.0)
        out_local = torch.matmul(attn_local, v)
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
        out = out_local + out_global
        out = out.permute(0, 2, 1, 3).contiguous().view(B, base_N, C)
        return self.out(out)

class SSMBlock(nn.Module):
    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.A = nn.Parameter(torch.randn(dim, d_state))
        self.D = nn.Parameter(torch.randn(d_state, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x_state = x @ self.A
        x_state = torch.cumsum(x_state, dim=1)
        out = x_state @ self.D
        return out

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
    """Simple Oumnix model with optional MoOp, WEAVE, Islet Injection, Cell, aux heads, and Bayesian residuals."""
    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        n_layers: int = 6,
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
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
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
        self.layers = nn.ModuleList()
        self.islet_cache = None
        if self.use_islet_injection:
            from collections import OrderedDict
            self.islet_cache = OrderedDict()
            self.islet_capacity = islet_capacity
        for _ in range(n_layers):
            att = LocalGlobalAttention(dim)
            ssm = SSMBlock(dim)
            if self.use_weave:
                att.qkv = WeaveLinear(LoRALayer(att.qkv, r=8, alpha=2.0))
                att.out = WeaveLinear(LoRALayer(att.out, r=8, alpha=2.0))
            else:
                att.qkv = LoRALayer(att.qkv, r=8, alpha=2.0)
                att.out = LoRALayer(att.out, r=8, alpha=2.0)
            if self.use_moop:
                mixer = TokenOperatorMixer(dim, n_ops=3, temperature=moop_temperature)
                dw = DepthwiseConv1d(dim)
                if self.use_cell:
                    mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
                    verifier = nn.Linear(dim, 1)
                    self.layers.append(nn.ModuleList([att, ssm, mixer, dw, mlp, verifier]))
                else:
                    self.layers.append(nn.ModuleList([att, ssm, mixer, dw]))
            else:
                if self.use_cell:
                    mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
                    verifier = nn.Linear(dim, 1)
                    self.layers.append(nn.ModuleList([att, ssm, mlp, verifier]))
                else:
                    self.layers.append(nn.ModuleList([att, ssm]))
        self.lm_head = LoRALayer(nn.Linear(dim, vocab_size), r=8, alpha=1.5)
        if self.use_aux_heads:
            self.aux_temporal = nn.Linear(dim, dim)
            self.aux_identity = nn.Linear(dim, dim)

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
