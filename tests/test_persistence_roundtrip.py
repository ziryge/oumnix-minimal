import os
import numpy as np
from memory.episodic import EpisodicMemory

def test_episodic_memory_save_load_roundtrip(tmp_path):
    dim = 8
    mem = EpisodicMemory(dim=dim, normalize=True, store_vectors=True, metric='l2')
    v1 = np.ones((1, dim), dtype='float32')
    v2 = np.zeros((1, dim), dtype='float32')
    mem.add(v1, ["vec1"]) 
    mem.add(v2, ["vec2"]) 
    save_dir = tmp_path / "episodic"
    mem.save(str(save_dir))
    mem2 = EpisodicMemory(dim=dim, normalize=True, store_vectors=True, metric='l2')
    mem2.load(str(save_dir))
    res = mem2.search(v1.flatten(), k=2)
    assert isinstance(res, list) and len(res) > 0
    texts = [t for t, _ in res]
    assert "vec1" in texts or "vec2" in texts
