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
from utils.logging_utils import get_logger
from utils.metrics import MovingAverage, Timer, PerfTracker

# Global perf tracker for OMNX exporter
PERF_EXPORT = PerfTracker(window=100)
from utils.omnx_exporter import start_omnx_exporter
from memory.persistence import PersistenceManager

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.oumnix_ai import create_oumnix_ai, OumnixAIConfig
from utils.dataset import TextLineDataset
from utils.tokenizer import tokenizer
from utils.seeds import set_seed

def _init_weights(m):
    import torch
    import torch.nn as nn
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)

def _param_groups(model, weight_decay: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith('.bias') or 'norm' in n.lower() or 'layernorm' in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


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
    logger = get_logger("train")
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    step_timer = Timer()
    step_timer.start()
    perf = PerfTracker(window=100)

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
            dt = step_timer.stop()
            step_timer.start()
            tokens = args.batch_size * args.max_seq_len * args.log_interval
            # Aggregate attention metrics if present on model
            kv_hit = 0.0
            head_drop = 0.0
            n = 0
            try:
                core = getattr(model, 'core_model', None)
                layers_src = None
                if core is not None and hasattr(core, 'layers'):
                    layers_src = core.layers
                elif hasattr(model, 'layers'):
                    layers_src = model.layers
                if layers_src is not None:
                    for layer in layers_src:
                        try:
                            att = layer[0]
                            if hasattr(att, 'last_kv_hit') and hasattr(att, 'last_head_drop'):
                                kv_hit += float(att.last_kv_hit)
                                head_drop += float(att.last_head_drop)
                                n += 1
                        except Exception:
                            continue
                if n > 0:
                    kv_hit /= n
                    head_drop /= n
            except Exception:
                pass
            perf.update(tokens=tokens, seconds=dt, kv_hit=kv_hit, head_drop=head_drop)
            PERF_EXPORT.update(tokens=tokens, seconds=dt, kv_hit=kv_hit, head_drop=head_drop)
            snap = perf.snapshot()
            logger.info(f"Epoch {epoch} | Step {step+1}/{num_batches} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | ms/token: {snap['ms_per_token']:.3f} | tokens/s: {snap['tokens_per_sec']:.1f} | vram_gb: {snap['vram_gb']:.2f}")

    return total_loss / num_batches


def evaluate(model, dataloader, device, args):
    model.eval()
    losses = []
    with torch.inference_mode():
        for batch in dataloader:
            loss, _ = compute_loss(model, batch, device, args.use_amp)
            losses.append(loss.item())
    return sum(losses) / max(len(losses), 1)


def train(args):
    logger = get_logger("train")
    # OMNX exporter
    omnx_thread = start_omnx_exporter(lambda: PERF_EXPORT.snapshot(), port=None)
    # Persistence manager
    pm = None
    if args.save_full_state or args.resume_full_state:
        pm = PersistenceManager(base_dir=args.state_dir, password=args.state_password)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Starting Oumnix Agent Training")
    logger.info(f"Device: {device}")
    logger.info(f"Vocabulary: {tokenizer.vocab_size} tokens")
    logger.info(f"Model: {args.dim}d, {args.n_layers} layers")
    logger.info(f"Mixed Precision: {args.use_amp}")

    dataset = TextLineDataset(dataset_dir=os.path.join(PROJECT_ROOT, 'datasets'))
    logger.info(f"Dataset: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
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
    try:
        model.apply(_init_weights)
    except Exception:
        pass

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {total_params/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        _param_groups(model, args.weight_decay),
        lr=args.lr,
        betas=(0.9, 0.95),
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
        logger.warning(f"OneCycleLR scheduler not available ({e}); using linear warmup then constant LR")
        warmup_steps = max(1, int(0.1 * total_steps))
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler() if args.use_amp else None

    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info("=" * 60)

    best_loss = float('inf')
    patience = getattr(args, 'early_stop_patience', 0)
    bad_epochs = 0

    # Resume full state if requested
    if args.resume_full_state and pm is not None:
        try:
            state = pm.load_complete_state()
            model.load_state_dict(state['model_weights'])
            # Restore memory systems if present
            try:
                mem_state = state.get('memory_state') or {}
                if 'infinity_window' in mem_state and hasattr(model, 'memory_system'):
                    model.memory_system = mem_state['infinity_window']
            except Exception:
                logger.warning("Failed to restore memory state; continuing")
            if state.get('optimizer_state'):
                optimizer.load_state_dict(state['optimizer_state'])
            if state.get('scheduler_state') and scheduler is not None:
                scheduler.load_state_dict(state['scheduler_state'])
            if state.get('scaler_state') and scaler is not None:
                scaler.load_state_dict(state['scaler_state'])
            logger.info("Resumed full state from LifeFile")
        except Exception as e:
            logger.warning(f"Failed to resume full state: {e}")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        avg_loss = train_epoch(
            model, dataloader, optimizer, scheduler, scaler, device, epoch, args
        )

        epoch_time = time.time() - start_time

        logger.info(f"Epoch {epoch} completed")
        logger.info(f"Average loss: {avg_loss:.4f}")
        logger.info(f"Time: {epoch_time:.1f}s")
        logger.info(f"Tokens/s: {len(dataset) * args.max_seq_len / max(epoch_time,1e-6):.0f}")

        val_loss = evaluate(model, val_loader, device, args)
        logger.info(f"Validation loss: {val_loss:.4f}")

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
        tmp_ckpt = ckpt_path + ".tmp"
        torch.save(checkpoint, tmp_ckpt)
        os.replace(tmp_ckpt, ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")

        # Save full state via PersistenceManager
        if args.save_full_state and pm is not None:
            try:
                mem_state = {}
                try:
                    if hasattr(model, 'memory_system'):
                        mem_state['infinity_window'] = model.memory_system
                except Exception:
                    pass
                pm.save_complete_state(
                    model_state=model.state_dict(),
                    memory_state=mem_state,
                    neuro_state={},
                    metacognition_state={},
                    config={
                        'dim': args.dim,
                        'n_layers': args.n_layers,
                        'n_heads': args.n_heads,
                        'max_seq_len': args.max_seq_len,
                        'lr': args.lr,
                    },
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict() if scheduler else None,
                    scaler_state=scaler.state_dict() if scaler else None,
                )
                logger.info("Full agent state saved to LifeFile")
            except Exception as e:
                logger.warning(f"Failed to save full state: {e}")

        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss
            best_path = os.path.join(args.out_dir, "best_model.pt")
            tmp_best = best_path + ".tmp"
            torch.save(checkpoint, tmp_best)
            os.replace(tmp_best, best_path)
            logger.info(f"New best model: {best_path}")
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience > 0 and bad_epochs >= patience:
                logger.info("Early stopping triggered")
                break

        print("-" * 60)

    print("Training finished")
    print(f"   Best val loss: {best_loss:.4f}")
    print(f"   Checkpoints saved in: {args.out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Oumnix Agent Training")
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic mode')

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
    parser.add_argument('--early_stop_patience', type=int, default=0, help='Early stopping patience (0 disables)')

    # Persistence flags
    parser.add_argument('--save-full-state', dest='save_full_state', action='store_true', help='Save full agent state to LifeFile after each epoch')
    parser.add_argument('--resume-full-state', dest='resume_full_state', action='store_true', help='Resume training from LifeFile state')
    parser.add_argument('--state-dir', dest='state_dir', type=str, default='.ai_state', help='Directory for LifeFile persistence')
    parser.add_argument('--state-password', dest='state_password', type=str, default=os.environ.get('OUMNIX_STATE_PASSWORD', ''), help='Password for LifeFile encryption')

    args = parser.parse_args()

    set_seed(1337, deterministic=args.deterministic)

    os.makedirs(args.out_dir, exist_ok=True)

    tmp_path = os.path.join(args.out_dir, 'train_config.txt.tmp')
    final_path = os.path.join(args.out_dir, 'train_config.txt')
    with open(tmp_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    os.replace(tmp_path, final_path)

    train(args)


if __name__ == '__main__':
    main()
