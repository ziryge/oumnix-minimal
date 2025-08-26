import torch
from core.model import OumnixSimpleAI, LocalGlobalAttention
from utils.tokenizer import tokenizer

def test_defaults_flags():
    m = OumnixSimpleAI(vocab_size=tokenizer.vocab_size, dim=64, n_layers=1)
    assert m.use_moop is False
    for layer in m.layers:
        att = layer[0]
        assert isinstance(att, LocalGlobalAttention)
        assert att.use_rag is False
