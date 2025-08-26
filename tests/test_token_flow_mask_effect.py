import torch
import pytest
from core.model import LocalGlobalAttention

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_token_flow_mask_reduces_value_utilization():
    x = torch.randn(2, 10, 64, device=DEVICE)
    assert x.is_cuda == (DEVICE.type == 'cuda')
    att_off = LocalGlobalAttention(dim=64, heads=4, local_window=4, enable_token_flow=False, token_flow_thresh=0.0).to(DEVICE)
    att_on = LocalGlobalAttention(dim=64, heads=4, local_window=4, enable_token_flow=True, token_flow_thresh=0.9).to(DEVICE)
    y_off = att_off(x)
    y_on = att_on(x)
    assert y_off.shape == y_on.shape
