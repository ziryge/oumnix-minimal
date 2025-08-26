import torch
from core.model import OumnixSimpleAI
from utils.tokenizer import tokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_islet_cache_path_and_entropy_track():
    ids = torch.randint(0, tokenizer.vocab_size, (2, 4), device=DEVICE)
    m = OumnixSimpleAI(vocab_size=tokenizer.vocab_size, dim=64, n_layers=1, use_moop=True, use_islet_injection=True, islet_capacity=4).to(DEVICE)
    with torch.inference_mode():
        y1 = m(ids)
        y2 = m(ids)
    assert y1.shape == y2.shape
    # access entropy track from mixer if present
    found_entropy = False
    for layer in m.layers:
        for mod in layer:
            if hasattr(mod, 'last_entropy') and mod.last_entropy is not None:
                found_entropy = True
                break
    assert found_entropy
