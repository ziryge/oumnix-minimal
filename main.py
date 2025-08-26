import argparse
import os
from utils.seeds import set_seed
import sys
import os
import signal
# from pathlib import Path
import torch
from utils.logging_utils import get_logger

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.oumnix_ai import create_oumnix_ai, OumnixAIConfig
from utils.tokenizer import tokenizer

ai_instance = None

def signal_handler(signum, frame):
    logger = get_logger("main")
    logger.info("Interrupt signal received...")
    if ai_instance:
        ai_instance.deactivate()
    sys.exit(0)

def main():
    global ai_instance
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description="oumnix agent – full offline agent")
    parser.add_argument(
        "--ui",
        choices=["cli", "web"],
        default="cli",
        help="Interface to launch (default: cli)",
    )
    parser.add_argument(
        "--model-dim",
        type=int,
        default=768,
        help="Model dimensionality (default: 768)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=12,
        help="Number of layers (default: 12)",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=12,
        help="Number of attention heads (default: 12)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="Path to training checkpoint (.pt) to load weights",
    )
    parser.add_argument(
        "--align-config-from-checkpoint",
        action="store_true",
        help="Align model dimensions to checkpoint when loading",
    )
    parser.add_argument(
        "--load-state",
        action="store_true",
        help="Load previously saved state",
    )
    parser.add_argument(
        "--state-dir",
        type=str,
        default=".ai_state",
        help="Directory to save state (default: .ai_state)",
    )
    parser.add_argument(
        "--password",
        type=str,
        help="Password for encryption (if not provided, a default is used)",
    )
    parser.add_argument(
        "--no-neurochemistry",
        action="store_true",
        help="Disable neurochemistry system",
    )
    parser.add_argument(
        "--consolidation-interval",
        type=int,
        default=3600,
        help="Consolidation interval in seconds (default: 3600)",
    )
    
    args = parser.parse_args()

    # Deterministic seeding (env toggle)
    set_seed(1337, deterministic=(os.environ.get("OUMNIX_DETERMINISTIC", "0") == "1"))
    
    logger = get_logger("main")
    logger.info("Initializing oumnix agent...")
    logger.info(f"Vocabulary: {tokenizer.vocab_size} tokens")
    logger.info(f"Model: {args.model_dim}d, {args.layers} layers")
    logger.info(f"State directory: {args.state_dir}")
    
    try:
        ckpt_to_load = None
        if args.checkpoint_path and os.path.exists(args.checkpoint_path):
            try:
                ckpt_to_load = torch.load(args.checkpoint_path, map_location="cpu")
                if args.align_config_from_checkpoint and isinstance(ckpt_to_load, dict):
                    ckpt_args = ckpt_to_load.get('args') or {}
                    args.model_dim = int(ckpt_args.get('dim', args.model_dim))
                    args.layers = int(ckpt_args.get('n_layers', args.layers))
                    args.n_heads = int(ckpt_args.get('n_heads', args.n_heads))
                    print(f"[CONFIG] Aligning config to checkpoint: {args.model_dim}d, {args.layers} layers, {args.n_heads} heads")
            except Exception as e:
                logger.warning(f"Could not inspect checkpoint: {e}")
        else:
            candidate_dirs = [
                os.path.join(PROJECT_ROOT, 'checkpo'),
                os.path.join(PROJECT_ROOT, 'checkpoints_streaming'),
                os.path.join(PROJECT_ROOT, 'checkpoints')
            ]
            found = []
            for d in candidate_dirs:
                if not os.path.isdir(d):
                    continue
                try:
                    for name in os.listdir(d):
                        path = os.path.join(d, name)
                        if not os.path.isfile(path):
                            continue
                        if name == 'best_model.pt':
                            found.append((10**12, path))  
                        elif name.startswith('ckpt_step_') and name.endswith('.pt'):
                            try:
                                step = int(name[len('ckpt_step_'):-3])
                                found.append((step, path))
                            except ValueError:
                                pass
                        elif name.startswith('checkpoint_epoch_') and name.endswith('.pt'):
                            try:
                                ep = int(name[len('checkpoint_epoch_'):-3])
                                found.append((ep * 10**6, path))
                            except ValueError:
                                pass
                except Exception:
                    continue
            if found:
                found.sort(key=lambda x: x[0], reverse=True)
                auto_path = found[0][1]
                try:
                    ckpt_to_load = torch.load(auto_path, map_location='cpu')
                    args.checkpoint_path = auto_path
                    logger.info(f"Automatically detected checkpoint: {auto_path}")
                    if args.align_config_from_checkpoint and isinstance(ckpt_to_load, dict):
                        ckpt_args = ckpt_to_load.get('args') or {}
                        args.model_dim = int(ckpt_args.get('dim', args.model_dim))
                        args.layers = int(ckpt_args.get('n_layers', args.layers))
                        args.n_heads = int(ckpt_args.get('n_heads', args.n_heads))
                        logger.info(f"Aligning config to checkpoint: {args.model_dim}d, {args.layers} layers, {args.n_heads} heads")
                except Exception as e:
                    logger.warning(f"Failed to load detected checkpoint: {e}")

        config = OumnixAIConfig(
            vocab_size=tokenizer.vocab_size,
            model_dim=args.model_dim,
            n_layers=args.layers,
            n_heads=args.n_heads,
            state_dir=args.state_dir,
            encryption_password=args.password,
            use_neurochemistry=not args.no_neurochemistry,
            consolidation_interval=args.consolidation_interval
        )
        
        ai_instance = create_oumnix_ai(config)
        
        if ckpt_to_load is not None:
            try:
                state = ckpt_to_load.get('model_state_dict', ckpt_to_load)
                missing, unexpected = ai_instance.core_model.load_state_dict(state, strict=False)
                logger.info(f"Checkpoint loaded: {args.checkpoint_path}")
                if missing:
                    logger.warning(f"Missing {len(missing)} keys")
                if unexpected:
                    logger.warning(f"{len(unexpected)} unexpected keys")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        
        if args.load_state:
            logger.info("Trying to load previous state...")
            if ai_instance.load_state():
                logger.info("State loaded successfully")
            else:
                logger.warning("Could not load state, starting from scratch")
        
        ai_instance.activate()
        
        status = ai_instance.get_system_status()
        logger.info(f"Status: {status['model_params']/1e6:.1f}M parameters, {status['background_threads']} active threads")
        
        if args.ui == "cli":
            try:
                from ui.advanced_cli import start_advanced_cli
                start_advanced_cli(ai_instance)
            except ImportError:
                from ui.cli import chat
                logger.warning("Using simple CLI (advanced_cli not available)")
                chat()
        else:
            try:
                from ui.advanced_web import start_advanced_web
                start_advanced_web(ai_instance)
            except ImportError:
                from ui.web import chat_interface
                logger.warning("Using simple web interface (advanced_web not available)")
                chat_interface.launch()
            
    except KeyboardInterrupt:
        logger.info("User interrupted")
    except Exception as e:
        logger.error(f"Fatal: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ai_instance:
            ai_instance.deactivate()
        logger.info("oumnix agent terminated")

if __name__ == "__main__":
    main()
