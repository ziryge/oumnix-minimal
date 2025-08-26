import torch
from memory.infinity_window import InfinityWindow, MemoryConfig


def test_infinity_window_smoke(tmp_path):
    cfg = MemoryConfig(hot_kv_size=16, warm_window_size=4, memory_dir=str(tmp_path))
    win = InfinityWindow(cfg, dim=8, n_heads=2, head_dim=4)
    # create dummy q, k, v
    q = torch.randn(1, 6, 8)
    k = torch.randn(1, 6, 2, 4)
    v = torch.randn(1, 6, 2, 4)
    win.add_tokens(k, v, text="hello world", embeddings=q)
    out = win.query(q)
    assert out.shape == q.shape
