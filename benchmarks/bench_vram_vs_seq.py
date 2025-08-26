import argparse
import time
import torch
from utils.tokenizer import tokenizer
from core.model import OumnixSimpleAI


def bench_vram(dim: int, layers: int, seq_values: str, batch: int, device: str):
    dev = torch.device(device)
    seqs = [int(s) for s in seq_values.split(',')]
    model = OumnixSimpleAI(vocab_size=tokenizer.vocab_size, dim=dim, n_layers=layers).to(dev)
    rows = []
    with torch.inference_mode():
        for seq in seqs:
            if torch.cuda.is_available() and dev.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            ids = torch.randint(0, tokenizer.vocab_size - 1, (batch, seq), device=dev)
            t0 = time.perf_counter()
            _ = model(ids)
            dt = time.perf_counter() - t0
            vram = 0.0
            if torch.cuda.is_available() and dev.type == 'cuda':
                vram = torch.cuda.max_memory_allocated() / 1e9
            rows.append((seq, dt, vram))
    print("seq,ms,gb")
    for seq, dt, vram in rows:
        print(f"{seq},{dt*1000:.3f},{vram:.2f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--seqs', type=str, default='64,128,256,512')
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    bench_vram(args.dim, args.layers, args.seqs, args.batch, args.device)
