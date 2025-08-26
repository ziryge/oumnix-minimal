import torch
from core.model import LocalGlobalAttention

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_token_flow_gating_effect():
    x = torch.randn(2, 12, 64, device=DEVICE)
    assert x.is_cuda == (DEVICE.type == 'cuda')
    att = LocalGlobalAttention(dim=64, heads=4, local_window=4, enable_token_flow=True, token_flow_thresh=0.0).to(DEVICE)
    y = att(x)
    assert y.shape == x.shape
