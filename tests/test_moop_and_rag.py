import torch
import pytest
from utils.tokenizer import tokenizer
from core.model import OumnixSimpleAI

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_moop_shapes_and_default_parity():
    x = torch.randint(0, tokenizer.vocab_size - 1, (2, 7), device=DEVICE)
    m_off = OumnixSimpleAI(vocab_size=tokenizer.vocab_size, dim=64, n_layers=2, use_moop=False).to(DEVICE)
    m_on = OumnixSimpleAI(vocab_size=tokenizer.vocab_size, dim=64, n_layers=2, use_moop=True).to(DEVICE)
    y_off = m_off(x)
    y_on = m_on(x)
    assert y_off.shape == y_on.shape

@pytest.mark.parametrize('rag_on', [False, True])
def test_rag_disabled_parity(rag_on):
    x = torch.randint(0, tokenizer.vocab_size - 1, (1, 6), device=DEVICE)
    m = OumnixSimpleAI(vocab_size=tokenizer.vocab_size, dim=64, n_layers=1, use_moop=False).to(DEVICE)
    if rag_on:
        for layer in m.layers:
            att = layer[0]
            if hasattr(att, 'set_rag_provider'):
                att.use_rag = True
                att.set_rag_provider(lambda x: torch.randn(x.size(0), 2, x.size(-1), device=x.device))
    y = m(x)
    assert y.shape == (1, 6, tokenizer.vocab_size)
