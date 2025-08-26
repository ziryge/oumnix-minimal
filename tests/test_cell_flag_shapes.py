import torch
import pytest
from utils.tokenizer import tokenizer
from core.model import OumnixSimpleAI

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_cell_flag_shapes():
    ids = torch.randint(0, tokenizer.vocab_size, (2, 7), device=DEVICE)
    m = OumnixSimpleAI(
        vocab_size=tokenizer.vocab_size,
        dim=64,
        n_layers=2,
        use_moop=True,
        use_cell=True,
        cell_threshold=0.5
    ).to(DEVICE)
    with torch.inference_mode():
        y = m(ids)
    assert y.shape == (2, 7, tokenizer.vocab_size)
