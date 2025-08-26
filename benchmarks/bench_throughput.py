import argparse
import time
import torch
from utils.tokenizer import tokenizer
from core.model import OumnixSimpleAI
from utils.metrics import PerfTracker


def bench(dim: int, layers: int, heads: int, seq: int, batch: int, steps: int, device: str):
    dev = torch.device(device)
    model = OumnixSimpleAI(vocab_size=tokenizer.vocab_size, dim=dim, n_layers=layers).to(dev)
    ids = torch.randint(0, tokenizer.vocab_size - 1, (batch, seq), device=dev)
    perf = PerfTracker(window=steps)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() and dev.type == 'cuda' else None
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(steps):
            _ = model(ids)
            tokens = batch * seq
            perf.update(tokens, max(time.perf_counter() - t0, 1e-6))
            t0 = time.perf_counter()
    snap = perf.snapshot()
    print(f"tokens_per_sec={snap['tokens_per_sec']:.1f}, ms_per_token={snap['ms_per_token']:.3f}, vram_gb={snap['vram_gb']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--seq', type=int, default=256)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    bench(args.dim, args.layers, args.heads, args.seq, args.batch, args.steps, args.device)
