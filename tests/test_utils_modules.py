import torch
import os
import json
from utils import metrics, rag_provider, seeds


def test_perftracker_snapshot():
    p = metrics.PerfTracker(window=10)
    p.update(tokens=1000, seconds=0.5)
    snap = p.snapshot()
    assert 'tokens_per_sec' in snap and 'ms_per_token' in snap and 'vram_gb' in snap


def test_rag_provider_call_shape():
    rp = rag_provider.SimpleRagProvider(dim=64, topk=4)
    x = torch.randn(2, 8, 64)
    out = rp(x)
    assert out is None or (out.dim() == 3 and out.size(0) == 2 and out.size(2) == 64)


def test_seeds_deterministic():
    os.environ.pop("PYTHONHASHSEED", None)
    seeds.set_seed(42, deterministic=True)
    assert os.environ.get("PYTHONHASHSEED") == "42"
