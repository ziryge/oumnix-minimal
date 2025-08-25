import argparse
import os
import sys
from pathlib import Path
import time
import json
import gc

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import re
from torch.amp import autocast, GradScaler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Tokenizer
try:
    from utils.tokenizer import tokenizer
    print("[OK] Tokenizer loaded")
except ImportError as e:
    print(f"[FAIL] Tokenizer error: {e}")
    sys.exit(1)

# Model
try:
    from core.oumnix_ai import create_oumnix_ai, OumnixAIConfig
    USE_ADVANCED_MODEL = True
    print("[OK] Advanced model available")
except ImportError as e:
    print(f"[WARN] Advanced model not available: {e}")
    try:
        from core.model import OumnixSimpleAI
        USE_ADVANCED_MODEL = False
        print("[OK] Using simple model")
    except ImportError as e2:
        print(f"[FAIL] No model available: {e2}")
        sys.exit(1)


class StreamingDataset(Dataset):
    def __init__(self, dataset_dir: str, max_length: int = 512, chunk_size: int = 1000,
                 max_chunks: int = 100, strip_tags: bool = False):
        self.dataset_dir = Path(dataset_dir)
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self.strip_tags = strip_tags

        self.files = []
        for ext in ['*.json', '*.jsonl', '*.txt']:
            self.files.extend(self.dataset_dir.rglob(ext))
        print(f"[INFO] Found {len(self.files)} files (streaming all)")

        self.total_lines = 0
        self.file_line_counts = []
        for file_path in self.files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                    self.file_line_counts.append(line_count)
                    self.total_lines += line_count
            except Exception:
                self.file_line_counts.append(0)

        print(f"[INFO] Approx. total: {self.total_lines} lines")

        self.current_chunk = []
        self.current_chunk_idx = 0
        self.chunk_start_idx = 0

    def __len__(self):
        if self.max_chunks is not None and self.max_chunks > 0:
            return min(self.total_lines, self.chunk_size * self.max_chunks)
        return self.total_lines

    def _load_chunk(self, start_idx: int):
        self.current_chunk = []
        lines_loaded = 0
        current_line = 0

        for file_idx, file_path in enumerate(self.files):
            if lines_loaded >= self.chunk_size:
                break

            file_line_count = self.file_line_counts[file_idx]
            if current_line + file_line_count <= start_idx:
                current_line += file_line_count
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_idx, line in enumerate(f):
                        if current_line < start_idx:
                            current_line += 1
                            continue
                        if lines_loaded >= self.chunk_size:
                            break

                        texts = self._extract_texts(line.strip())
                        for text in texts:
                            if self.strip_tags:
                                text = self._clean_text(text)
                            tokens = self._tokenize_text(text)
                            if len(tokens) > 0:
                                self.current_chunk.append(tokens)
                                lines_loaded += 1
                                if lines_loaded >= self.chunk_size:
                                    break
                        current_line += 1
            except Exception as e:
                print(f"[WARN] Failed to read {file_path}: {e}")
                continue

        self.chunk_start_idx = start_idx
        print(f"[INFO] Chunk loaded: {len(self.current_chunk)} samples (start: {start_idx})")
        gc.collect()

    def _extract_texts(self, line: str):
        if not line:
            return []
        try:
            obj = json.loads(line)
            texts = []
            if isinstance(obj, dict):
                if 'conversations' in obj and isinstance(obj['conversations'], list):
                    for turn in obj['conversations']:
                        if isinstance(turn, dict) and isinstance(turn.get('content'), str):
                            texts.append(turn['content'])
                val = obj.get('text') or obj.get('content')
                if isinstance(val, str):
                    texts.append(val)
                return texts
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        val = item.get('text') or item.get('content')
                        if isinstance(val, str):
                            texts.append(val)
                    elif isinstance(item, str):
                        texts.append(item)
                return texts or [line]
        except json.JSONDecodeError:
            pass

        texts = []
        for key in ("content", "text"):
            for m in re.finditer(rf'"{key}"\s*:\s*"(.*?)"', line):
                s = m.group(1)
                s = s.encode('utf-8').decode('unicode_escape')
                texts.append(s)
        return texts or [line]

    def _tokenize_text(self, text: str) -> torch.Tensor:
        if len(text.strip()) == 0:
            return torch.empty(0, dtype=torch.long)
        token_ids = tokenizer.encode(text)[:self.max_length]
        if len(token_ids) == 0:
            return torch.empty(0, dtype=torch.long)
        return torch.tensor(token_ids, dtype=torch.long)

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"</?[^>]{1,20}>", "", text)
        text = text.replace('\n', ' ').replace('\t', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def __getitem__(self, idx):
        chunk_idx = idx - self.chunk_start_idx
        if (not self.current_chunk or 
            chunk_idx < 0 or 
            chunk_idx >= len(self.current_chunk)):
            chunk_start = (idx // self.chunk_size) * self.chunk_size
            self._load_chunk(chunk_start)
            chunk_idx = idx - self.chunk_start_idx
        if 0 <= chunk_idx < len(self.current_chunk):
            return self.current_chunk[chunk_idx]
        return torch.empty(0, dtype=torch.long)


def collate_fn_streaming(batch):
    batch = [t for t in batch if t.numel() > 0]
    if not batch:
        dummy = torch.tensor([tokenizer.token2id.get(tokenizer.unk_token, 1)], dtype=torch.long)
        return dummy.unsqueeze(0), dummy.unsqueeze(0)

    max_len = min(max(t.size(0) for t in batch), 512)
    padded = []
    pad_token_id = tokenizer.token2id.get(tokenizer.pad_token, 0)

    for t in batch:
        if t.size(0) > max_len:
            t = t[:max_len]
        pad_len = max_len - t.size(0)
        if pad_len > 0:
            t = torch.cat([t, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        padded.append(t)

    ids = torch.stack(padded)
    return ids, ids


def save_checkpoint_step(model, optimizer, scaler, args, epoch, global_step, avg_loss=None):
    ckpt = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'avg_loss': avg_loss,
        'args': vars(args)
    }
    path = os.path.join(args.out_dir, f"ckpt_step_{global_step}.pt")
    torch.save(ckpt, path)
    print(f"[INFO] Step checkpoint saved: {path}")


def create_model_streaming(args):
    if USE_ADVANCED_MODEL:
        print("[INFO] Creating advanced model (streaming mode)...")
        config = OumnixAIConfig(
            vocab_size=tokenizer.vocab_size,
            model_dim=args.dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            max_sequence_length=512,
            use_neurochemistry=args.enable_neurochem,
            hot_kv_size=512,
            consolidation_interval=7200,
            batch_size=args.batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        model = create_oumnix_ai(config)
    else:
        print("[INFO] Creating simple model (streaming mode)...")
        model = OumnixSimpleAI(
            vocab_size=tokenizer.vocab_size,
            dim=args.dim,
            n_layers=args.n_layers
        )
    return model


def train_epoch_streaming(model, dataset, optimizer, scaler, device, epoch, args, global_step=0):
    model.train()
    total_loss = 0.0
    num_samples = 0

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_streaming,
        num_workers=0,
        pin_memory=False
    )

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for step, (ids, targets) in enumerate(progress_bar):
        if ids.numel() == 0:
            continue

        ids = ids.to(device, non_blocking=False)
        targets = targets.to(device, non_blocking=False)

        optimizer.zero_grad()

        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        with autocast(device_type=device_type, enabled=(args.use_amp and device_type == 'cuda')):
            if USE_ADVANCED_MODEL:
                try:
                    _out = model.core_model(input_ids=ids, attention_mask=None, use_cache=False)
                    logits = _out['logits'] if isinstance(_out, dict) else _out
                    del _out
                except Exception as e:
                    print(f"[WARN] Advanced model failed: {e}")
                    logits = model.core_model.lm_head(model.core_model.token_embed(ids))
            else:
                logits = model(ids)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=tokenizer.token2id.get(tokenizer.pad_token, 0)
            )

        if args.use_amp:
            scaler.scale(loss).backward(retain_graph=True)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward(retain_graph=True)
            optimizer.step()

        total_loss += loss.item()
        num_samples += 1
        avg_loss = total_loss / num_samples

        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'samples': num_samples,
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

        if (step + 1) % args.log_interval == 0:
            print(f"[INFO] Epoch {epoch} | Step {step+1} | Loss: {avg_loss:.4f} | Samples: {num_samples}")

        global_step += 1
        if args.save_every_steps and args.save_every_steps > 0 and (global_step % args.save_every_steps == 0):
            save_checkpoint_step(model, optimizer, scaler, args, epoch, global_step, avg_loss)

        if (step + 1) % 100 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    return total_loss / max(num_samples, 1), global_step


def main():
    parser = argparse.ArgumentParser(description="Streaming Training")
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (small for streaming)')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--dim', type=int, default=256, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--chunk_size', type=int, default=500, help='Chunk size')
    parser.add_argument('--max_chunks', type=int, default=100, help='Chunks per epoch (<=0 use all)')
    parser.add_argument('--max_length', type=int, default=256, help='Max sequence length')
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--strip_tags', action='store_true', help='Strip short tags <...> and normalize spaces during training')
    parser.add_argument('--enable_neurochem', action='store_true', help='Enable neurochemistry system')
    parser.add_argument('--use_fp8', action='store_true', help='Use FP8 (RTX 4000)')
    parser.add_argument('--log_interval', type=int, default=25, help='Log interval')
    parser.add_argument('--save_every_steps', type=int, default=10000, help='Save checkpoint every N steps')
    parser.add_argument('--out_dir', type=str, default='checkpoints_streaming', help='Output directory')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')

    args = parser.parse_args()

    print("Streaming Training")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[INFO] GPU: {gpu_name}")
        print(f"[INFO] VRAM: {vram_gb:.1f}GB")
        if "RTX 40" in gpu_name:
            print("[INFO] RTX 4000 detected - enabling FP8 optimizations")
            args.use_fp8 = True

    try:
        dataset = StreamingDataset(
            dataset_dir="datasets",
            max_length=args.max_length,
            chunk_size=args.chunk_size,
            max_chunks=args.max_chunks,
            strip_tags=args.strip_tags
        )
        if len(dataset) == 0:
            print("[FAIL] Empty dataset. Add files to datasets/")
            return
    except Exception as e:
        print(f"[FAIL] Dataset error: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        model = create_model_streaming(args)
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Model parameters: {total_params/1e6:.1f}M")
    except Exception as e:
        print(f"[FAIL] Model creation error: {e}")
        import traceback
        traceback.print_exc()
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = GradScaler() if args.use_amp else None
    os.makedirs(args.out_dir, exist_ok=True)

    print("\nTraining configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Max length: {args.max_length}")
    print(f"  Mixed precision: {args.use_amp}")
    print(f"  FP8: {args.use_fp8}")
    print("-" * 50)

    best_loss = float('inf')
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n[INFO] Epoch {epoch}/{args.epochs}")
        start_time = time.time()
        try:
            avg_loss, global_step = train_epoch_streaming(model, dataset, optimizer, scaler, device, epoch, args, global_step)
            epoch_time = time.time() - start_time
            print(f"[OK] Epoch {epoch} completed")
            print(f"  Average loss: {avg_loss:.4f}")
            print(f"  Time: {epoch_time:.1f}s")

            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                'loss': avg_loss,
                'args': vars(args)
            }
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"[INFO] Epoch checkpoint saved: {ckpt_path}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(args.out_dir, "best_model.pt")
                torch.save(checkpoint, best_path)
                print(f"[OK] New best model: {best_path}")

        except Exception as e:
            print(f"[FAIL] Error in epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            break

    print("\nTraining finished")
    print(f"[RESULT] Best loss: {best_loss:.4f}")
    print(f"[RESULT] Checkpoints saved in: {args.out_dir}")
    print(f"\nTo run the trained model:")
    print(f"  python main.py --load-state --state-dir {args.out_dir}")


if __name__ == "__main__":
    main()
