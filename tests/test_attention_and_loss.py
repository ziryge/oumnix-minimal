import os
import torch
import pytest

from core.model import OumnixSimpleAI
from core.loss import free_energy_loss
from utils.tokenizer import tokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_causal_attention_no_future_leakage():
    model = OumnixSimpleAI(vocab_size=tokenizer.vocab_size, dim=64, n_layers=2).to(DEVICE)
    ids = torch.randint(0, tokenizer.vocab_size, (1, 8), device=DEVICE)
    logits_full = model(ids)
    ids_masked = ids.clone()
    ids_masked[0, -1] = tokenizer.token2id.get(tokenizer.pad_token, 0)
    logits_truncated = model(ids_masked)
    assert torch.allclose(logits_truncated[0, :-1], logits_full[0, :-1], atol=1e-4, rtol=1e-4)

def test_free_energy_loss_finite_cpu():
    vocab = tokenizer.vocab_size
    logits = torch.randn(2, 5, vocab)
    target = torch.randint(0, vocab, (2, 5))
    sigma = torch.full((2, 5, 1), 0.1)
    loss = free_energy_loss(logits, target, sigma)
    assert torch.isfinite(loss).item()

def test_free_energy_loss_finite_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    vocab = tokenizer.vocab_size
    logits = torch.randn(2, 5, vocab, device='cuda')
    target = torch.randint(0, vocab, (2, 5), device='cuda')
    sigma = torch.full((2, 5, 1), 0.1, device='cuda')
    loss = free_energy_loss(logits, target, sigma)
    assert torch.isfinite(loss).item()
