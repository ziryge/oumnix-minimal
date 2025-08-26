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
from utils.seeds import set_seed
from utils.logging_utils import get_logger

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Load tokenizer
try:
    from utils.tokenizer import tokenizer
    print("[OK] Tokenizer loaded")
except ImportError as e:
    print(f"[FAIL] Tokenizer error: {e}")
    sys.exit(1)

# Load dataset
try:
    from utils.dataset import TextLineDataset
    print("[OK] Dataset module loaded")
except ImportError as e:
    print(f"[FAIL] Dataset error: {e}")
    sys.exit(1)

# Load model
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

def collate_fn(batch):
    max_len = min(max(t.size(0) for t in batch), 1024)  
    
    padded = []
    for t in batch:
        if t.size(0) > max_len:
            t = t[:max_len]
        
        pad_len = max_len - t.size(0)
        if pad_len > 0:
            t = torch.cat([
                t, 
                torch.full((pad_len,), tokenizer.token2id.get(tokenizer.pad_token, 0), dtype=torch.long)
            ])
        padded.append(t)
    
    ids = torch.stack(padded)
    return ids, ids  

def create_model(args):
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
    if USE_ADVANCED_MODEL:
        print("[INFO] Creating advanced model...")
        config = OumnixAIConfig(
            vocab_size=tokenizer.vocab_size,
            model_dim=args.dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            use_neurochemistry=False  
        )
        model = create_oumnix_ai(config)
        print("[OK] Advanced model created")
    else:
        print("[INFO] Creating simple model...")
        model = OumnixSimpleAI(
            vocab_size=tokenizer.vocab_size, 
            dim=args.dim, 
            n_layers=args.n_layers
        )
        print("[OK] Simple model created")
    
    return model

def train_epoch(model, dataloader, optimizer, scaler, device, epoch, args):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for step, (ids, targets) in enumerate(progress_bar):
        ids = ids.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=args.use_amp):
            if USE_ADVANCED_MODEL:
                _out = model.core_model(ids)
                logits = _out['logits']
                del _out  
            else:
                logits = model(ids)
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=tokenizer.token2id.get(tokenizer.pad_token, 0)
            )
        
        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        progress_bar.set_postfix({
            'loss': f'{total_loss/(step+1):.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
        
        if (step + 1) % args.log_interval == 0:
            avg_loss = total_loss / (step + 1)
            print(f"Epoch {epoch} | Step {step+1}/{num_batches} | Loss: {avg_loss:.4f}")
    
    return total_loss / num_batches

def main():
    parser = argparse.ArgumentParser(description="Simplified Oumnix AI Training")
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic mode')
    
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--dim', type=int, default=256, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--log_interval', type=int, default=50, help='Log interval')
    parser.add_argument('--out_dir', type=str, default='checkpoints_simple', help='Output directory')
    parser.add_argument('--max_samples', type=int, default=2000, help='Limit dataset samples for quick training')
    
    args = parser.parse_args()
    
    logger = get_logger("train_simple")
    set_seed(1337, deterministic=args.deterministic)
    logger.info("Simplified Oumnix Agent Training")
    logger.info("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Load dataset
    try:
        dataset = TextLineDataset(dataset_dir="datasets")
        logger.info(f"Dataset: {len(dataset)} samples")
        
        if args.max_samples and len(dataset) > args.max_samples:
            dataset.samples = dataset.samples[:args.max_samples]
            logger.info(f"Dataset limited to {len(dataset)} samples (max_samples)")
        
        if len(dataset) == 0:
            logger.error("Empty dataset! Add files to datasets/")
            return
        
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=0  
        )
        
    except Exception as e:
        print(f"[FAIL] Dataset error: {e}")
        return
    
    # Create model
    try:
        model = create_model(args)
        model = model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Parameters: {total_params/1e6:.1f}M")
        
    except Exception as e:
        print(f"[FAIL] Model creation error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith('.bias') or 'norm' in n.lower() or 'layernorm' in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    param_groups = [
        {"params": decay, "weight_decay": 0.01},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    scaler = GradScaler() if args.use_amp else None
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    logger.info("Starting training...")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Mixed precision: {args.use_amp}")
    logger.info("-" * 50)
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        try:
            avg_loss = train_epoch(model, dataloader, optimizer, scaler, device, epoch, args)
            
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch} completed:")
            logger.info(f"  Loss: {avg_loss:.4f}")
            logger.info(f"  Time: {epoch_time:.1f}s")
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': vars(args)
            }
            
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_epoch_{epoch}.pt")
            tmp_ckpt = ckpt_path + ".tmp"
            torch.save(checkpoint, tmp_ckpt)
            os.replace(tmp_ckpt, ckpt_path)
            logger.info(f"  Checkpoint saved: {ckpt_path}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(args.out_dir, "best_model.pt")
                tmp_best = best_path + ".tmp"
                torch.save(checkpoint, tmp_best)
                os.replace(tmp_best, best_path)
                logger.info(f"  New best model: {best_path}")
            
            logger.info("-" * 50)
            
        except Exception as e:
            print(f"[FAIL] Error in epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    logger.info("Training finished")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Checkpoints saved in: {args.out_dir}")

if __name__ == "__main__":
    main()
