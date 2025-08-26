import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def test_imports():
    import torch
    assert torch is not None
    from utils.tokenizer import tokenizer
    assert tokenizer.vocab_size > 0
    from core.oumnix_ai import create_oumnix_ai, OumnixAIConfig
    assert create_oumnix_ai is not None and OumnixAIConfig is not None
    from utils.dataset import TextLineDataset
    assert TextLineDataset is not None

def test_model_creation():
    from core.oumnix_ai import create_oumnix_ai, OumnixAIConfig
    from utils.tokenizer import tokenizer
    config = OumnixAIConfig(
        vocab_size=tokenizer.vocab_size,
        model_dim=256,
        n_layers=2,
        n_heads=4,
        use_neurochemistry=False
    )
    model = create_oumnix_ai(config)
    assert sum(p.numel() for p in model.parameters()) > 0

def test_dataset():
    from utils.dataset import TextLineDataset
    ds = TextLineDataset(dataset_dir="datasets")
    assert isinstance(len(ds), int)
    if len(ds) > 0:
        sample = ds[0]
        assert hasattr(sample, 'shape')

def test_fp8():
    import torch
    has_fp8 = hasattr(torch, 'float8_e4m3fn')
    has_fp8_e5m2 = hasattr(torch, 'float8_e5m2')
    assert isinstance(has_fp8, bool) and isinstance(has_fp8_e5m2, bool)
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        assert isinstance(name, str) and len(name) > 0

def main():
    # manual runner kept for compatibility; CI uses pytest directly
    test_imports()
    test_model_creation()
    test_dataset()
    test_fp8()

if __name__ == "__main__":
    main()
