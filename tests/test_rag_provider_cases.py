import torch
import numpy as np
import pytest
from utils.rag_provider import SimpleRagProvider

def test_rag_provider_empty_returns_none():
    rp = SimpleRagProvider(dim=16, topk=2)
    x = torch.randn(1, 4, 16)
    assert rp(x) is None

def test_rag_provider_update_and_call():
    rp = SimpleRagProvider(dim=8, topk=2)
    v = torch.arange(8, dtype=torch.float32)
    rp.update_with_sequence_embed(v)
    x = torch.randn(1, 3, 8)
    out = rp(x)
    assert out is not None and out.shape == (1, 2, 8)

@pytest.mark.parametrize("bad", [torch.randn(2, 8), torch.randn(8, 1)])
def test_rag_provider_update_bad_shape_raises(bad):
    rp = SimpleRagProvider(dim=8)
    with pytest.raises(ValueError):
        rp.update_with_sequence_embed(bad)
