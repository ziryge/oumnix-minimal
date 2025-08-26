import numpy as np
import torch
from memory.episodic import EpisodicMemory


def test_rag_provider_vectors_roundtrip(tmp_path):
    mem = EpisodicMemory(dim=8, normalize=True, store_vectors=True)
    v = np.random.randn(5, 8).astype('float32')
    mem.add(v, [f"t{i}" for i in range(5)])
    save_dir = tmp_path / "mem"
    mem.save(str(save_dir))
    mem2 = EpisodicMemory(dim=8, normalize=True, store_vectors=True)
    mem2.load(str(save_dir))
    q = np.random.randn(8).astype('float32')
    res = mem2.search(q, k=3)
    assert len(res) <= 3
