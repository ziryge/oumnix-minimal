import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def test_imports():
    print("Testing imports...")
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"[FAIL] PyTorch: {e}")
        return False
    
    try:
        from utils.tokenizer import tokenizer
        print(f"Tokenizer: {tokenizer.vocab_size} tokens")
    except ImportError as e:
        print(f"[FAIL] Tokenizer: {e}")
        return False
    
    try:
        from core.oumnix_ai import create_oumnix_ai, OumnixAIConfig
        print("OumnixAI imports successful")
    except ImportError as e:
        print(f"[FAIL] OumnixAI import error: {e}")
        return False
    
    try:
        from utils.dataset import TextLineDataset
        print("Dataset imports successful")
    except ImportError as e:
        print(f"[FAIL] Dataset: {e}")
        return False
    
    return True

def test_model_creation():
    print("\nTesting model creation...")
    
    try:
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
        print(f"[OK] Model created: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
        return True
        
    except Exception as e:
        print(f"[FAIL] Model creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset():
    print("\nTesting dataset...")
    
    try:
        from utils.dataset import TextLineDataset
        
        dataset = TextLineDataset(dataset_dir="datasets")
        print(f"[OK] Dataset loaded: {len(dataset)} samples")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"[OK] Sample: {sample.shape} tokens")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Dataset error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fp8():
    print("\nTesting FP8 support...")
    
    try:
        import torch
        
        has_fp8 = hasattr(torch, 'float8_e4m3fn')
        print(f"FP8 E4M3: {'Available' if has_fp8 else 'Not available'}")
        
        has_fp8_e5m2 = hasattr(torch, 'float8_e5m2')
        print(f"FP8 E5M2: {'Available' if has_fp8_e5m2 else 'Not available'}")
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"GPU: {device_name}")
            
            is_rtx_4000 = "RTX 40" in device_name
            print(f"RTX 4000 Series: {'Yes' if is_rtx_4000 else 'No'}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] FP8 test error: {e}")
        return False

def main():
    print("System Test – Training Environment")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Dataset", test_dataset),
        ("FP8 Support", test_fp8)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"[FAIL] Test {name} failed with error: {e}")
            results.append((name, False))
    
    print("\nTest Results:")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\nAll tests passed successfully!")
        print("You can now run training with for example:")
        print("  python train.py --use_amp --batch_size 4")
    else:
        print("\nSome tests failed. Check errors above.")

if __name__ == "__main__":
    main()
