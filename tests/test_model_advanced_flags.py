import torch
import pytest
from core.model import OumnixSimpleAI
from utils.tokenizer import tokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_weave": True},
        {"use_islet_injection": True},
        {"use_bayesian_residuals": True, "residual_std": 0.01},
        {"early_exit": True, "exit_threshold": 10.0},  # extremely high to trigger early exit immediately
    ],
)
def test_advanced_flags_forward(kwargs):
    ids = torch.randint(0, tokenizer.vocab_size, (2, 5), device=DEVICE)
    m = OumnixSimpleAI(vocab_size=tokenizer.vocab_size, dim=64, n_layers=1, **kwargs).to(DEVICE)
    with torch.inference_mode():
        y = m(ids)
    assert y.dim() == 3 and y.size(0) == 2 and y.size(2) == tokenizer.vocab_size
