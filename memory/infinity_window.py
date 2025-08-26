"""
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import pickle
import hashlib
import time
from collections import defaultdict, deque

@dataclass
class MemoryConfig:
    """
"""
    hot_kv_size: int = 4096
    hot_kv_precision: str = "fp8"
    warm_window_size: int = 1024
    warm_max_windows: int = 32
    pq_clusters: int = 256
    lowrank_dim: int = 64
    tree_fanout: int = 8
    tree_max_depth: int = 6
    svo_max_length: int = 128
    max_anchors: int = 8
    anchor_fetch_size: int = 128
    anchor_similarity_threshold: float = 0.0
    memory_dir: str = ".memory"
    compression_level: int = 6
    # Eviction weights
    eviction_alpha: float = 0.7  # time penalty weight
    eviction_beta: float = 0.3   # importance complement weight

class ProductQuantizer:
    """
"""
    def __init__(self, dim: int, n_clusters: int = 256, n_subvectors: int = 8, seed: int = 1337):
        self.dim = dim
        self.n_clusters = n_clusters
        self.n_subvectors = n_subvectors
        self.subvector_dim = dim // n_subvectors
        self.seed = seed
        # Codebooks per subvector
        self.codebooks = []
        self.is_trained = False
        # incremental training buffers
        self._buffer = []  # type: List[np.ndarray]
        self._buffer_limit = 8192
        
    def train(self, vectors: np.ndarray):
        """
"""
        vectors = vectors.reshape(-1, self.dim)
        n_samples = vectors.shape[0]
        
        self.codebooks = []
        for i in range(self.n_subvectors):
            start_idx = i * self.subvector_dim
            end_idx = (i + 1) * self.subvector_dim
            subvectors = vectors[:, start_idx:end_idx]
            
            
            k = max(1, min(self.n_clusters, subvectors.shape[0]))
            kmeans = faiss.Kmeans(self.subvector_dim, k)
            try:
                kmeans.seed = int(self.seed)
            except Exception:
                pass
            kmeans.train(subvectors.astype('float32'))
            self.codebooks.append(kmeans.centroids)
        
        self.is_trained = True
        self._buffer.clear()

    def partial_fit(self, vectors: np.ndarray):
        """Incremental update: accumulate and retrain when buffer exceeds limit."""
        if vectors.size == 0:
            return
        self._buffer.append(vectors.reshape(-1, self.dim))
        total = sum(arr.shape[0] for arr in self._buffer)
        if total >= self._buffer_limit or not self.is_trained:
            data = np.concatenate(self._buffer, axis=0)
            self.train(data)
    
    def encode(self, vectors: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
"""
        if not self.is_trained:
            raise ValueError("Quantizer not trained")
        
        vectors = vectors.reshape(-1, self.dim)
        codes = np.zeros((vectors.shape[0], self.n_subvectors), dtype=np.uint8)
        
        for i, codebook in enumerate(self.codebooks):
            start_idx = i * self.subvector_dim
            end_idx = (i + 1) * self.subvector_dim
            subvectors = vectors[:, start_idx:end_idx]
            
            
            index = faiss.IndexFlatL2(self.subvector_dim)
            index.add(codebook.astype('float32'))
            _, indices = index.search(subvectors.astype('float32'), 1)
            codes[:, i] = indices.flatten()
        
        return codes, self.codebooks
    
    def decode(self, codes: np.ndarray, codebooks: List[np.ndarray]) -> np.ndarray:
        """
"""
        n_samples = codes.shape[0]
        vectors = np.zeros((n_samples, self.dim), dtype=np.float32)
        
        for i, codebook in enumerate(codebooks):
            start_idx = i * self.subvector_dim
            end_idx = (i + 1) * self.subvector_dim
            vectors[:, start_idx:end_idx] = codebook[codes[:, i]]
        
        return vectors

class LowRankCompressor:
    """
"""
    def __init__(self, rank: int = 64):
        self.rank = rank
        self.U = None
        self.s = None
        self.Vt = None
    
    def compress(self, matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
"""
        
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
        
        
        rank = min(self.rank, len(s))
        U_trunc = U[:, :rank]
        s_trunc = s[:rank]
        Vt_trunc = Vt[:rank, :]
        
        return {
            'U': U_trunc.astype(np.float16),
            's': s_trunc.astype(np.float16),
            'Vt': Vt_trunc.astype(np.float16),
            'shape': matrix.shape
        }
    
    def decompress(self, compressed: Dict[str, np.ndarray]) -> np.ndarray:
        """
"""
        U = compressed['U'].astype(np.float32)
        s = compressed['s'].astype(np.float32)
        Vt = compressed['Vt'].astype(np.float32)
        
        
        matrix = U @ np.diag(s) @ Vt
        
        
        original_shape = compressed['shape']
        if matrix.shape != original_shape:
            matrix = matrix.reshape(original_shape)
        
        return matrix

@dataclass
class WarmKVWindow:
    """
"""
    timestamp: float
    start_pos: int
    end_pos: int
    k_codes: np.ndarray  
    k_codebooks: List[np.ndarray]
    v_compressed: Dict[str, np.ndarray]  
    anchors: Dict[int, np.ndarray]  
    metadata: Dict[str, Any]

class ContextTreeNode:
    """
"""
    def __init__(self, level: int = 0):
        self.level = level
        self.children: List['ContextTreeNode'] = []
        self.start_pos: int = 0
        self.end_pos: int = 0
        self.timestamp: float = time.time()
        
        
        self.svo_summary: str = ""  
        self.embedding: Optional[np.ndarray] = None
        self.anchors_aggregated: Dict[int, np.ndarray] = {}
        self.text_hash: str = ""
        
        
        self.access_count: int = 0
        self.last_access: float = time.time()
        self.importance_score: float = 0.0

class HotKVCache:
    """
"""
    def __init__(self, config: MemoryConfig, n_heads: int, head_dim: int):
        self.config = config
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_size = config.hot_kv_size
        
        
        self.has_fp8 = hasattr(torch, 'float8_e4m3fn')
        self.use_fp8 = self.has_fp8 and getattr(config, 'use_fp8', True)
        
        
        if self.use_fp8:
            
            self.k_cache = torch.zeros(
                (n_heads, self.max_size, head_dim), 
                dtype=torch.float8_e4m3fn,  
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            self.v_cache = torch.zeros(
                (n_heads, self.max_size, head_dim),
                dtype=torch.float8_e4m3fn,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            
            self.k_scales = torch.ones(
                (n_heads, self.max_size), 
                dtype=torch.float32,
                device=self.k_cache.device
            )
            self.v_scales = torch.ones(
                (n_heads, self.max_size), 
                dtype=torch.float32,
                device=self.v_cache.device
            )
            
            print(f" HotKV Cache initialized with native FP8 (RTX 4000 optimized)")
            print(f"   Memory savings: ~50% vs FP16")
            
        else:
            
            self.k_cache = torch.zeros(
                (n_heads, self.max_size, head_dim), 
                dtype=torch.float16,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            self.v_cache = torch.zeros(
                (n_heads, self.max_size, head_dim),
                dtype=torch.float16,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            print(f" FP8 not available, using FP16 fallback")
        
        self.current_pos = 0
        self.is_full = False
        
        
        self.total_adds = 0
        self.fp8_ratio = 1.0 if self.use_fp8 else 0.0
        
    def add(self, k: torch.Tensor, v: torch.Tensor, 
            uncertainty: Optional[torch.Tensor] = None) -> None:
        """
"""
        batch_size, seq_len, n_heads, head_dim = k.shape
        
        for i in range(seq_len):
            if self.current_pos >= self.max_size:
                self.current_pos = 0
                self.is_full = True
            
            k_token = k[0, i]  
            v_token = v[0, i]
            
            if self.use_fp8:
                
                use_high_precision = False
                if uncertainty is not None:
                    token_uncertainty = uncertainty[0, i].item() if uncertainty.dim() > 1 else uncertainty[i].item()
                    use_high_precision = token_uncertainty > 0.1  
                
                if use_high_precision:
                    
                    k_max = torch.max(torch.abs(k_token))
                    v_max = torch.max(torch.abs(v_token))
                    
                    
                    k_scale = k_max / 224.0  
                    v_scale = v_max / 224.0
                else:
                    
                    k_max = torch.max(torch.abs(k_token))
                    v_max = torch.max(torch.abs(v_token))
                    
                    
                    k_scale = k_max / 448.0
                    v_scale = v_max / 448.0
                
                
                k_scale = max(k_scale.item(), 1e-8)
                v_scale = max(v_scale.item(), 1e-8)
                
                
                k_quantized = (k_token / k_scale).clamp(-448.0, 448.0)
                v_quantized = (v_token / v_scale).clamp(-448.0, 448.0)
                
                self.k_cache[:, self.current_pos] = k_quantized.to(torch.float8_e4m3fn)
                self.v_cache[:, self.current_pos] = v_quantized.to(torch.float8_e4m3fn)
                
                
                self.k_scales[:, self.current_pos] = k_scale
                self.v_scales[:, self.current_pos] = v_scale
                
            else:
                
                self.k_cache[:, self.current_pos] = k_token.to(self.k_cache.dtype)
                self.v_cache[:, self.current_pos] = v_token.to(self.v_cache.dtype)
            
            self.current_pos += 1
            self.total_adds += 1
    
    def get_range(self, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
"""
        if end > self.current_pos and not self.is_full:
            end = self.current_pos
        
        k_slice = self.k_cache[:, start:end]  
        v_slice = self.v_cache[:, start:end]
        
        if self.use_fp8:
            
            k_scales_slice = self.k_scales[:, start:end].unsqueeze(-1)  
            v_scales_slice = self.v_scales[:, start:end].unsqueeze(-1)
            
            
            k_slice = k_slice.to(torch.float32) * k_scales_slice
            v_slice = v_slice.to(torch.float32) * v_scales_slice
        else:
            
            k_slice = k_slice.to(torch.float32)
            v_slice = v_slice.to(torch.float32)
        
        return k_slice, v_slice
    
    def get_recent(self, length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
"""
        if length > self.max_size:
            length = self.max_size
        
        if self.is_full:
            
            start = (self.current_pos - length) % self.max_size
            if start < self.current_pos:
                k_slice = self.k_cache[:, start:self.current_pos]
                v_slice = self.v_cache[:, start:self.current_pos]
            else:
                
                k1 = self.k_cache[:, start:]
                k2 = self.k_cache[:, :self.current_pos]
                k_slice = torch.cat([k1, k2], dim=1)
                
                v1 = self.v_cache[:, start:]
                v2 = self.v_cache[:, :self.current_pos]
                v_slice = torch.cat([v1, v2], dim=1)
        else:
            start = max(0, self.current_pos - length)
            k_slice = self.k_cache[:, start:self.current_pos]
            v_slice = self.v_cache[:, start:self.current_pos]
        
        return k_slice, v_slice

class TeleportAttention(nn.Module):
    """
"""
    def __init__(self, config: MemoryConfig, dim: int, n_heads: int):
        super().__init__()
        self.config = config
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        
        self.anchor_query = nn.Linear(dim, dim)
        self.anchor_key = nn.Linear(dim, dim)
        
        
        self.scratch_size = config.anchor_fetch_size
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if dev == 'cuda' else torch.float32
        self.scratch_k = torch.zeros(
            (n_heads, self.scratch_size, self.head_dim),
            dtype=dtype,
            device=dev
        )
        self.scratch_v = torch.zeros(
            (n_heads, self.scratch_size, self.head_dim),
            dtype=dtype,
            device=dev
        )
    
    def select_anchors(self, query: torch.Tensor, 
                      context_tree: ContextTreeNode,
                      max_anchors: int = 8) -> List[Tuple[ContextTreeNode, float]]:
        """
"""
        candidates = []
        
        def traverse(node: ContextTreeNode):
            if node.embedding is not None:
                
                q_emb = query.mean(dim=1).cpu().numpy()  
                similarity = np.dot(q_emb.flatten(), node.embedding.flatten())
                candidates.append((node, similarity))
            
            for child in node.children:
                traverse(child)
        
        traverse(context_tree)
        
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        # Apply similarity threshold and cap
        thr = getattr(self.config, 'anchor_similarity_threshold', 0.0)
        filtered = [(n, s) for (n, s) in candidates if s >= thr]
        return filtered[:max_anchors]
    
    def fetch_mini_kv(self, anchor_node: ContextTreeNode,
                     warm_kv_storage: Dict[str, WarmKVWindow]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
"""
        
        window_key = f"{anchor_node.start_pos}_{anchor_node.end_pos}"
        
        if window_key not in warm_kv_storage:
            return None, None
        
        window = warm_kv_storage[window_key]
        
        
        pq = ProductQuantizer(self.head_dim)
        k_decompressed = pq.decode(window.k_codes, window.k_codebooks)
        
        
        lr_compressor = LowRankCompressor()
        v_decompressed = lr_compressor.decompress(window.v_compressed)
        
        
        k_tensor = torch.from_numpy(k_decompressed).to(self.scratch_k.device)
        v_tensor = torch.from_numpy(v_decompressed).to(self.scratch_v.device)
        
        
        total = k_tensor.shape[0]
        length = max(1, total // max(1, self.n_heads))
        # reshape to [n_heads, length, head_dim]
        k_tensor = k_tensor[: self.n_heads * length].contiguous().view(self.n_heads, length, self.head_dim)
        v_tensor = v_tensor[: self.n_heads * length].contiguous().view(self.n_heads, length, self.head_dim)
        return k_tensor, v_tensor
    
    def forward(self, query: torch.Tensor,
                hot_kv: HotKVCache,
                context_tree: ContextTreeNode,
                warm_kv_storage: Dict[str, WarmKVWindow],
                use_teleport: bool = True) -> torch.Tensor:
        """
"""
        batch_size, seq_len, dim = query.shape
        
        
        q = self.anchor_query(query)  
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        
        hot_k, hot_v = hot_kv.get_recent(self.config.hot_kv_size)
        
        
        hot_attn = self._compute_attention(q, hot_k, hot_v)
        
        if not use_teleport:
            return hot_attn
        
        
        anchors = self.select_anchors(query, context_tree, self.config.max_anchors)
        
        if not anchors:
            return hot_attn
        
        
        teleport_outputs = []
        weights = []
        
        for anchor_node, similarity in anchors:
            mini_k, mini_v = self.fetch_mini_kv(anchor_node, warm_kv_storage)
            
            if mini_k is not None and mini_v is not None:
                
                mini_attn = self._compute_attention(q, mini_k, mini_v)
                teleport_outputs.append(mini_attn)
                weights.append(max(similarity, 0.0))
        
        if teleport_outputs:
            w = torch.tensor(weights, dtype=torch.float32, device=hot_attn.device)
            if torch.sum(w) > 0:
                w = w / torch.sum(w)
            else:
                w = torch.full((len(teleport_outputs),), 1.0/len(teleport_outputs), device=hot_attn.device)
            stacked = torch.stack(teleport_outputs)
            teleport_combined = torch.sum(stacked * w.view(-1, 1, 1, 1), dim=0)
            
            output = 0.7 * hot_attn + 0.3 * teleport_combined
        else:
            output = hot_attn
        
        return output
    
    def _compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
"""
        
        
        
        batch_size, seq_len = q.shape[:2]
        kv_seq_len = k.shape[1]
        
        
        q = q.transpose(1, 2)  
        k = k.unsqueeze(0).expand(batch_size, -1, -1, -1)  
        v = v.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        
        # ensure same device and dtype for matmul
        dev = q.device
        q = q.to(torch.float32)
        k = k.to(device=dev, dtype=torch.float32)
        v = v.to(device=dev, dtype=torch.float32)
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        output = torch.matmul(attn_weights, v)
        
        
        output = output.transpose(1, 2)  
        output = output.contiguous().view(batch_size, seq_len, self.dim)
        
        return output

class InfinityWindow:
    """
"""
    def __init__(self, config: MemoryConfig, dim: int, n_heads: int, head_dim: int):
        self.config = config
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        
        
        self.hot_kv = HotKVCache(config, n_heads, head_dim)
        self.warm_kv_storage: Dict[str, WarmKVWindow] = {}
        self.context_tree = ContextTreeNode(level=0)
        self.teleport_attention = TeleportAttention(config, dim, n_heads)
        
        
        self.pq_compressor = ProductQuantizer(head_dim, config.pq_clusters)
        self.lr_compressor = LowRankCompressor(config.lowrank_dim)
        
        
        self.total_tokens = 0
        self.current_window_start = 0
        
        
        self.memory_dir = Path(config.memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        
        self.stats = {
            'hot_hits': 0,
            'warm_hits': 0,
            'cold_hits': 0,
            'teleport_calls': 0,
            'compression_ratio': 0.0
        }
    
    def add_tokens(self, k: torch.Tensor, v: torch.Tensor, 
                  text: str, embeddings: torch.Tensor) -> None:
        """
"""
        batch_size, seq_len, n_heads, head_dim = k.shape
        
        
        self.hot_kv.add(k, v)
        
        
        self.total_tokens += seq_len
        
        
        if self.total_tokens - self.current_window_start >= self.config.warm_window_size:
            self._compress_to_warm()
        
        
        self._update_context_tree(text, embeddings)
    
    def _compress_to_warm(self) -> None:
        """
"""
        window_size = self.config.warm_window_size
        start_pos = self.current_window_start
        end_pos = self.total_tokens
        
        
        k_window, v_window = self.hot_kv.get_range(
            start_pos % self.config.hot_kv_size,
            end_pos % self.config.hot_kv_size
        )
        
        
        k_np = k_window.cpu().numpy().reshape(-1, self.head_dim)
        v_np = v_window.cpu().numpy().reshape(-1, self.head_dim)
        
        
        if not self.pq_compressor.is_trained:
            self.pq_compressor.train(k_np)
        else:
            # incremental update of codebooks
            self.pq_compressor.partial_fit(k_np)
        
        
        k_codes, k_codebooks = self.pq_compressor.encode(k_np)
        
        
        v_compressed = self.lr_compressor.compress(v_np)
        # reconstruction error bound (store for diagnostics)
        try:
            v_rec = self.lr_compressor.decompress(v_compressed)
            err = float(np.linalg.norm(v_np - v_rec) / max(1.0, np.linalg.norm(v_np)))
        except Exception:
            err = 0.0
        
        
        anchors = {}
        for head in range(self.n_heads):
            head_k = k_np[head::self.n_heads]
            if len(head_k) > 0:
                anchors[head] = np.mean(head_k, axis=0)
        
        
        # compute simple compression ratio
        comp_ratio = float(k_codes.size * 1.0 / max(1, k_np.size))
        window = WarmKVWindow(
            timestamp=time.time(),
            start_pos=start_pos,
            end_pos=end_pos,
            k_codes=k_codes,
            k_codebooks=k_codebooks,
            v_compressed=v_compressed,
            anchors=anchors,
            metadata={'compression_time': time.time(), 'importance': 0.0, 'last_access': time.time(), 'compression_ratio': comp_ratio, 'recon_error': err}
        )
        
        
        window_key = f"{start_pos}_{end_pos}"
        self.warm_kv_storage[window_key] = window
        
        
        self.current_window_start = end_pos
        
        
        if len(self.warm_kv_storage) > self.config.warm_max_windows:
            self._evict_old_windows()
    
    def _evict_old_windows(self) -> None:
        """
"""
        
        windows = list(self.warm_kv_storage.items())
        # Compute priority: alpha*time_penalty + beta*(1-importance)
        now = time.time()
        def _priority(win: WarmKVWindow) -> float:
            meta = getattr(win, 'metadata', {})
            imp = float(meta.get('importance', 0.0))
            last = float(meta.get('last_access', win.timestamp))
            time_penalty = max(0.0, (now - last))
            return self.config.eviction_alpha * time_penalty + self.config.eviction_beta * (1.0 - imp)
        windows.sort(key=lambda x: _priority(x[1]), reverse=True)
        
        to_remove = len(windows) - self.config.warm_max_windows
        for i in range(max(0, to_remove)):
            key, _window = windows[i]
            del self.warm_kv_storage[key]
    
    def _update_context_tree(self, text: str, embeddings: torch.Tensor) -> None:
        """
"""
        
        
        
        node = ContextTreeNode()
        node.start_pos = self.total_tokens - embeddings.shape[1]
        node.end_pos = self.total_tokens
        node.svo_summary = self._extract_svo(text)
        node.embedding = embeddings.mean(dim=1).cpu().numpy().flatten()
        node.text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        
        
        self.context_tree.children.append(node)
        
        
        if len(self.context_tree.children) > self.config.tree_fanout:
            
            self._consolidate_tree_nodes()
    
    def _extract_svo(self, text: str) -> str:
        """
"""
        
        words = text.split()
        if len(words) <= self.config.svo_max_length:
            return text
        
        
        start = " ".join(words[:self.config.svo_max_length//2])
        end = " ".join(words[-self.config.svo_max_length//2:])
        return f"{start} ... {end}"
    
    def _consolidate_tree_nodes(self) -> None:
        """
"""
        
        children = self.context_tree.children
        
        
        if len(children) > self.config.tree_fanout:
            
            parent_node = ContextTreeNode(level=1)
            parent_node.children = children[:self.config.tree_fanout//2]
            parent_node.start_pos = parent_node.children[0].start_pos
            parent_node.end_pos = parent_node.children[-1].end_pos
            
            
            embeddings = [child.embedding for child in parent_node.children if child.embedding is not None]
            if embeddings:
                parent_node.embedding = np.mean(embeddings, axis=0)
            
            
            self.context_tree.children = [parent_node] + children[self.config.tree_fanout//2:]
    
    def query(self, query: torch.Tensor, use_teleport: bool = True) -> torch.Tensor:
        """
"""
        self.stats['teleport_calls'] += 1

        out = self.teleport_attention(
            query=query,
            hot_kv=self.hot_kv,
            context_tree=self.context_tree,
            warm_kv_storage=self.warm_kv_storage,
            use_teleport=use_teleport
        )
        # Update metadata importance/last_access for accessed warm windows (anchor nodes used)
        try:
            anchors = self.teleport_attention.select_anchors(query, self.context_tree, self.config.max_anchors)
            for node, sim in anchors:
                key = f"{node.start_pos}_{node.end_pos}"
                if key in self.warm_kv_storage:
                    w = self.warm_kv_storage[key]
                    meta = w.metadata or {}
                    meta['last_access'] = time.time()
                    # importance: EMA by similarity evidence
                    old = float(meta.get('importance', 0.0))
                    meta['importance'] = 0.9 * old + 0.1 * max(0.0, float(sim))
                    w.metadata = meta
        except Exception:
            pass
        return out
    
    def save_state(self, path: str) -> None:
        """
"""
        state = {
            'config': self.config,
            'warm_kv_storage': self.warm_kv_storage,
            'context_tree': self.context_tree,
            'total_tokens': self.total_tokens,
            'current_window_start': self.current_window_start,
            'stats': self.stats
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, path: str) -> None:
        """
"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.warm_kv_storage = state['warm_kv_storage']
        self.context_tree = state['context_tree']
        self.total_tokens = state['total_tokens']
        self.current_window_start = state['current_window_start']
        self.stats = state['stats']
    
    def get_stats(self) -> Dict[str, Any]:
        """
"""
        return {
            **self.stats,
            'total_tokens': self.total_tokens,
            'warm_windows': len(self.warm_kv_storage),
            'tree_nodes': self._count_tree_nodes(self.context_tree),
            'memory_usage_mb': self._estimate_memory_usage()
        }
    
    def _count_tree_nodes(self, node: ContextTreeNode) -> int:
        """
"""
        count = 1
        for child in node.children:
            count += self._count_tree_nodes(child)
        return count
    
    def _estimate_memory_usage(self) -> float:
        """
"""
        
        hot_size = self.hot_kv.k_cache.numel() * 2 * 2  
        warm_size = len(self.warm_kv_storage) * 1024 * 1024  
        tree_size = self._count_tree_nodes(self.context_tree) * 1024  
        
        return (hot_size + warm_size + tree_size) / (1024 * 1024)

__all__ = [
    'InfinityWindow',
    'MemoryConfig', 
    'HotKVCache',
    'TeleportAttention',
    'ContextTreeNode'
]
