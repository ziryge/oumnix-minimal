import argparse
import os
import sys
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.oumnix_ai import create_oumnix_ai, OumnixAIConfig
from utils.dataset import TextLineDataset
from utils.tokenizer import tokenizer


def collate_fn(batch):
    max_len = min(max(t.size(0) for t in batch), 2048)

    padded = []
    attention_masks = []

    for t in batch:
        if t.size(0) > max_len:
            t = t[:max_len]

        pad_len = max_len - t.size(0)
        if pad_len > 0:
            padded_t = torch.cat([
                t,
                torch.full((pad_len,), tokenizer.token2id[tokenizer.pad_token], dtype=torch.long)
            ])
            mask = torch.cat([torch.ones(t.size(0)), torch.zeros(pad_len)])
        else:
            padded_t = t
            mask = torch.ones(t.size(0))

        padded.append(padded_t)
        attention_masks.append(mask)

    ids = torch.stack(padded)
    masks = torch.stack(attention_masks).bool()

    targets = ids.clone()

    return ids, targets, masks


def compute_loss(model, batch, device, use_amp=False):
    ids, targets, masks = batch
    ids = ids.to(device)
    targets = targets.to(device)
    masks = masks.to(device)

    with autocast(enabled=use_amp):
        outputs = model.core_model(
            input_ids=ids,
            attention_mask=masks,
            use_cache=False
        )
        logits = outputs['logits']

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        shift_masks = masks[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none'
        )
        loss = loss.view(shift_labels.shape)
        loss = loss * shift_masks.float()

        valid_tokens = shift_masks.sum()
        if valid_tokens.item() == 0:
            return (loss.mean(), logits)
        loss = loss.sum() / valid_tokens

        if hasattr(model, 'consolidator') and hasattr(model.consolidator, 'ewc'):
            ewc_loss = model.consolidator.ewc.compute_ewc_loss()
            loss = loss + 0.01 * ewc_loss

    return loss, logits


def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, epoch, args):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for step, batch in enumerate(progress_bar):
        loss, logits = compute_loss(model, batch, device, args.use_amp)

        loss = loss / args.gradient_accumulation_steps

        if args.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            if scheduler:
                scheduler.step()

            optimizer.zero_grad()

        total_loss += loss.item() * args.gradient_accumulation_steps
        avg_loss = total_loss / (step + 1)

        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

        if (step + 1) % args.log_interval == 0:
            print(f"Epoch {epoch} | Step {step+1}/{num_batches} | "
                  f"Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    return total_loss / num_batches


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Starting Oumnix Agent Training")
    print(f"   Device: {device}")
    print(f"   Vocabulary: {tokenizer.vocab_size} tokens")
    print(f"   Model: {args.dim}d, {args.n_layers} layers")
    print(f"   Mixed Precision: {args.use_amp}")

    dataset = TextLineDataset(dataset_dir=os.path.join(PROJECT_ROOT, 'datasets'))
    print(f"   Dataset: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    config = OumnixAIConfig(
        vocab_size=tokenizer.vocab_size,
        model_dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_sequence_length=args.max_seq_len,
        use_neurochemistry=False,
        state_dir=args.out_dir
    )

    if hasattr(torch, 'float8_e4m3fn'):
        print("[INFO] FP8 detected - optimizing for RTX 4000 series")
        config.use_fp8 = True
        config.fp8_e4m3_forward = True
        config.fp8_e5m2_backward = True
        config.fp8_dynamic_scaling = True
    else:
        print("[INFO] FP8 not available - using FP16")
        config.use_fp8 = False

    model = create_oumnix_ai(config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable parameters: {total_params/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        eps=1e-8
    )

    total_steps = max(1, len(dataloader) * args.epochs // max(1, args.gradient_accumulation_steps))
    warmup_steps = int(0.1 * total_steps)

    scheduler = None
    try:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
    except Exception as e:
        print(f"[WARN] OneCycleLR scheduler not available ({e}); continuing without scheduler")

    scaler = GradScaler() if args.use_amp else None

    print(f"   Total steps: {total_steps}")
    print(f"   Warmup steps: {warmup_steps}")
    print("=" * 60)

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        avg_loss = train_epoch(
            model, dataloader, optimizer, scheduler, scaler, device, epoch, args
        )

        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch} completed")
        print(f"   Average loss: {avg_loss:.4f}")
        print(f"   Time: {epoch_time:.1f}s")
        print(f"   Tokens/s: {len(dataset) * args.max_seq_len / epoch_time:.0f}")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'config': config,
            'loss': avg_loss,
            'tokenizer_vocab': tokenizer.id2token,
        }

        ckpt_path = os.path.join(args.out_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, ckpt_path)
        print(f"   Checkpoint saved: {ckpt_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.out_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"   New best model: {best_path}")

        print("-" * 60)

    print("Training finished")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   Checkpoints saved in: {args.out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Oumnix Agent Training")

    parser.add_argument('--dim', type=int, default=768, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--max_seq_len', type=int, default=2048, help='Max sequence length')

    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping')

    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--num_workers', type=int, default=2, help='Dataloader workers')

    parser.add_argument('--log_interval', type=int, default=100, help='Log interval')
    parser.add_argument('--out_dir', type=str, default='checkpoints', help='Output directory')

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, 'train_config.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    train(args)


if __name__ == '__main__':
    main()
