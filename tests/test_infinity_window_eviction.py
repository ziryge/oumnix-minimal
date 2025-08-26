import torch
from memory.infinity_window import InfinityWindow, MemoryConfig


def test_warm_window_eviction_policy_smoke():
    cfg = MemoryConfig(hot_kv_size=128, warm_window_size=8, warm_max_windows=2)
    iw = InfinityWindow(cfg, dim=32, n_heads=4, head_dim=8)
    # simulate add_tokens many times to trigger compression into warm windows
    for _ in range(5):
        k = torch.randn(1, 8, 4, 8)
        v = torch.randn(1, 8, 4, 8)
        text = "lorem ipsum"
        emb = torch.randn(1, 8, 32)
        iw.add_tokens(k, v, text, emb)
    # should have at most warm_max_windows after eviction
    assert len(iw.warm_kv_storage) <= cfg.warm_max_windows
