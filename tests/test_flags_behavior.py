import torch
import pytest
from utils.tokenizer import tokenizer
from core.model import OumnixSimpleAI, LocalGlobalAttention

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.mark.parametrize('early_exit,exit_threshold', [(False, 0.0), (True, 0.5)])
def test_early_exit_flag_shapes(early_exit, exit_threshold):
    x = torch.randint(0, tokenizer.vocab_size, (2, 9), device=DEVICE)
    m = OumnixSimpleAI(
        vocab_size=tokenizer.vocab_size,
        dim=64,
        n_layers=2,
        use_moop=False,
        early_exit=early_exit,
        exit_threshold=exit_threshold
    ).to(DEVICE)
    y = m(x)
    assert y.shape == (2, 9, tokenizer.vocab_size)

@pytest.mark.parametrize('enable_token_flow,thresh', [(False, 0.0), (True, 0.5)])
def test_token_flow_flag_shapes(enable_token_flow, thresh):
    x = torch.randn(1, 7, 64, device=DEVICE)
    attn = LocalGlobalAttention(dim=64, heads=4, local_window=4, enable_token_flow=enable_token_flow, token_flow_thresh=thresh).to(DEVICE)
    y = attn(x)
    assert y.shape == (1, 7, 64)
