import torch
import pytest
from utils.tokenizer import tokenizer
from core.model import OumnixSimpleAI

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_moop_on_off_shape_and_forward():
    ids = torch.randint(0, tokenizer.vocab_size - 1, (2, 7), device=DEVICE)
    m_off = OumnixSimpleAI(vocab_size=tokenizer.vocab_size, dim=64, n_layers=2, use_moop=False).to(DEVICE)
    m_on = OumnixSimpleAI(vocab_size=tokenizer.vocab_size, dim=64, n_layers=2, use_moop=True).to(DEVICE)
    with torch.inference_mode():
        y_off = m_off(ids)
        y_on = m_on(ids)
    assert y_off.shape == y_on.shape
