import sys
import os
import time
import torch
from core.model import OumnixSimpleAI
from neuro.chemistry import NeuroState
from memory.short_term import ShortTermBuffer
from utils.tokenizer import tokenizer
from utils.logging_utils import get_logger
from utils.seeds import set_seed

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')

set_seed(1337, deterministic=(os.environ.get("OUMNIX_DETERMINISTIC","0") == "1"))
model = OumnixSimpleAI(vocab_size=tokenizer.vocab_size).to(DEVICE)
state = NeuroState()
buffer = ShortTermBuffer()
_USE_RAG = os.environ.get("OUMNIX_USE_RAG", "0") == "1"
if _USE_RAG:
    from utils.rag_provider import SimpleRagProvider
    rag = SimpleRagProvider(dim=model.embed.embedding_dim, topk=8)
    for layer in model.layers:
        att = layer[0]
        if hasattr(att, 'set_rag_provider'):
            att.use_rag = True
            att.set_rag_provider(lambda x: rag(x))

def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default

def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default

def chat():  # -> None
    logger = get_logger("cli")
    logger.info("OumnixSimpleAI – terminal chat (Ctrl-C to exit)")
    while True:
        try:
            user = input("You: ")
        except KeyboardInterrupt:
            logger.info("Shutting down")
            break
        ctx = tokenizer.encode(user)
        ids = torch.tensor([ctx], device=DEVICE)
        generated = []
        max_new_tokens = _env_int("OUMNIX_MAX_NEW_TOKENS", 16)
        top_k = _env_int("OUMNIX_TOPK", 50)
        temperature_env = _env_float("OUMNIX_TEMPERATURE", state.temperature())
        stops_raw = os.environ.get("OUMNIX_STOP_SEQUENCES", "")
        stop_seqs = [s for s in (t.strip() for t in stops_raw.split(",")) if s]
        metrics_interval = _env_int("OUMNIX_METRICS_INTERVAL", 0)
        t0 = time.perf_counter()
        with torch.inference_mode():
            for step in range(max_new_tokens):
                logits = model(ids)
                last_logits = logits[:, -1] / temperature_env
                probs = torch.softmax(last_logits, dim=-1)
                if top_k > 0 and top_k < probs.size(-1):
                    topv, topi = torch.topk(probs, top_k)
                    idx = topi[0, torch.multinomial(topv[0], num_samples=1)]
                else:
                    idx = torch.multinomial(probs[0], num_samples=1)
                token_id = int(idx.item())
                generated.append(token_id)
                ids = torch.cat([ids, idx.view(1, 1)], dim=1)
                if metrics_interval > 0 and (step + 1) % metrics_interval == 0:
                    dt = time.perf_counter() - t0
                    tps = (step + 1) / max(dt, 1e-6)
                    logger.info(f"gen: {step+1} tokens in {dt:.2f}s ({tps:.1f} toks/s)")
                if stop_seqs:
                    partial = tokenizer.decode(generated)
                    if any(s in partial for s in stop_seqs):
                        break
        reply = tokenizer.decode(generated)
        logger.info(f"AI: {reply}")
        with torch.inference_mode():
            all_ids = ids.squeeze(0)
            logits = model(ids)
            last_probs = torch.softmax(logits[:, -1] / temperature_env, dim=-1)
            last_idx = int(ids[0, -1].item())
            surprise = -torch.log(last_probs[0, last_idx]).item()
            reward = 1.0 if "thank you" in reply.lower() else 0.0
            state.update(reward, surprise)
            embed_vec = model.embed(all_ids.unsqueeze(0)).squeeze(0)
            if _USE_RAG:
                seq_emb = embed_vec.mean(dim=0)
                rag.update_with_sequence_embed(seq_emb)
        buffer.add(all_ids, embed_vec)

if __name__ == "__main__":
    chat()
