import numpy as np
import torch
from memory.episodic import EpisodicMemory

class SimpleRagProvider:
    def __init__(self, dim: int, topk: int = 8, normalize: bool = True, store_vectors: bool = True):
        self.dim = dim
        self.topk = topk
        self.memory = EpisodicMemory(dim=dim, normalize=normalize, store_vectors=store_vectors)

    @torch.no_grad()
    def update_with_sequence_embed(self, seq_embed: torch.Tensor, text: str | None = None):
        if seq_embed.dim() != 1 or seq_embed.numel() != self.dim:
            raise ValueError("seq_embed must be a 1D tensor with dim")
        v = seq_embed.detach().cpu().numpy()[None, :].astype('float32')
        self.memory.add(v, [text or ""])  

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor | None:
        if len(self.memory.texts) == 0:
            return None
        B, N, C = x.shape
        x_mean = x.mean(dim=1)  
        out_vecs = []
        for b in range(B):
            q = x_mean[b].detach().cpu().numpy().astype('float32')
            res = self.memory.search(q, k=self.topk)
            if not res:
                return None
            # use stored vectors if available, otherwise synthesize zeros
            if hasattr(self.memory, "_vectors") and self.memory._vectors:
                idxs = []
                D, I = self.memory.index.search(q[None, :], self.topk)
                for i in I[0]:
                    if i < 0:
                        continue
                    idxs.append(i)
                if not idxs:
                    return None
                vecs = [self.memory._vectors[i] for i in idxs]
                if len(vecs) < self.topk:
                    pad = [np.zeros((self.dim,), dtype='float32') for _ in range(self.topk - len(vecs))]
                    vecs.extend(pad)
                arr = np.stack(vecs[: self.topk], axis=0).astype('float32')
            else:
                arr = np.zeros((self.topk, self.dim), dtype='float32')
            out_vecs.append(torch.tensor(arr, device=x.device))
        mem = torch.stack(out_vecs, dim=0)  
        return mem
