import torch
import time
import pytest
from utils.tokenizer import tokenizer
from core.model import OumnixSimpleAI

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.mark.timeout(20)
def test_performance_smoke_tokens_per_second():
    ids = torch.randint(0, tokenizer.vocab_size - 1, (1, 64), device=DEVICE)
    m = OumnixSimpleAI(vocab_size=tokenizer.vocab_size, dim=64, n_layers=2).to(DEVICE)
    t0 = time.perf_counter()
    with torch.inference_mode():
        _ = m(ids)
    dt = time.perf_counter() - t0
    assert dt < 5.0
