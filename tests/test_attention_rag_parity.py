import torch
import pytest
from core.model import LocalGlobalAttention

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_rag_none_provider_parity():
    x = torch.randn(2, 8, 64, device=DEVICE)
    att_base = LocalGlobalAttention(dim=64, heads=4, local_window=4, use_rag=False).to(DEVICE)
    y_base = att_base(x)
    att_rag = LocalGlobalAttention(dim=64, heads=4, local_window=4, use_rag=True).to(DEVICE)
    att_rag.set_rag_provider(lambda t: None)
    # Copy weights to ensure identical parameters
    att_rag.load_state_dict(att_base.state_dict(), strict=True)
    y_rag = att_rag(x)
    assert torch.allclose(y_base, y_rag, atol=1e-5, rtol=1e-5)
