import torch
import pytest
from utils.tokenizer import tokenizer
from core.model import OumnixSimpleAI

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_early_exit_disabled_parity():
    ids = torch.randint(0, tokenizer.vocab_size, (1, 6), device=DEVICE)
    m = OumnixSimpleAI(vocab_size=tokenizer.vocab_size, dim=64, n_layers=2, early_exit=False, exit_threshold=0.5).to(DEVICE)
    y = m(ids)
    assert y.shape == (1, 6, tokenizer.vocab_size)
