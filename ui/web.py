import os
import json
import torch
import gradio as gr
from core.model import OumnixSimpleAI
from neuro.chemistry import NeuroState
from utils.tokenizer import tokenizer
from utils.logging_utils import get_logger
from utils.seeds import set_seed

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')

set_seed(1337, deterministic=(os.environ.get("OUMNIX_DETERMINISTIC","0") == "1"))
model = OumnixSimpleAI(vocab_size=tokenizer.vocab_size).to(DEVICE)
state = NeuroState()
_USE_RAG = os.environ.get("OUMNIX_USE_RAG", "0") == "1"
if _USE_RAG:
    from utils.rag_provider import SimpleRagProvider
    rag = SimpleRagProvider(dim=model.embed.embedding_dim, topk=8)
    for layer in model.layers:
        att = layer[0]
        if hasattr(att, 'set_rag_provider'):
            att.use_rag = True
            att.set_rag_provider(lambda x: rag(x))

def make_interface(temperature: float | None = None, top_k: int = 50, max_new_tokens: int = 16):
    logger = get_logger("web")
    def respond(message, history):
        temp = temperature if temperature is not None else state.temperature()
        ctx = tokenizer.encode(message)
        ids = torch.tensor([ctx], device=DEVICE)
        generated = []
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                logits = model(ids)
                last_logits = logits[:, -1] / temp
                probs = torch.softmax(last_logits, dim=-1)
                if top_k > 0 and top_k < probs.size(-1):
                    topv, topi = torch.topk(probs, top_k)
                    idx = topi[0, torch.multinomial(topv[0], num_samples=1)]
                else:
                    idx = torch.multinomial(probs[0], num_samples=1)
                token_id = int(idx.item())
                generated.append(token_id)
                ids = torch.cat([ids, idx.view(1, 1)], dim=1)
        reply = tokenizer.decode(generated)
        with torch.inference_mode():
            logits = model(ids)
            last_probs = torch.softmax(logits[:, -1] / temp, dim=-1)
            last_idx = int(ids[0, -1].item())
            surprise = -torch.log(last_probs[0, last_idx]).item()
            reward = 1.0 if "thanks" in reply.lower() else 0.0
            state.update(reward, surprise)
            if _USE_RAG:
                embed_vec = model.embed(ids).squeeze(0)
                seq_emb = embed_vec.mean(dim=0)
                rag.update_with_sequence_embed(seq_emb)
        return reply, history + [(message, reply)]
    return gr.ChatInterface(fn=respond, title="OumnixSimpleAI – Chat Web", type='messages')

def make_interface_with_controls():  # -> gr.Blocks
    settings_path = os.path.join(os.getcwd(), ".oumnix_web_settings.json")
    def load_settings():
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"temperature": 1.0, "top_k": 50, "max_new_tokens": 16}
    def save_settings(d):
        try:
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(d, f)
        except Exception:
            pass
    init = load_settings()
    def respond_ctrl(message, history, temperature, top_k, max_new_tokens):
        cfg = {"temperature": float(temperature), "top_k": int(top_k), "max_new_tokens": int(max_new_tokens)}
        save_settings(cfg)
        return make_interface(temperature=cfg["temperature"], top_k=cfg["top_k"], max_new_tokens=cfg["max_new_tokens"]).fn(message, history)
    with gr.Blocks() as demo:
        temperature = gr.Slider(0.1, 2.0, value=float(init.get("temperature", 1.0)), step=0.05, label="Temperature")
        top_k = gr.Slider(1, 200, value=int(init.get("top_k", 50)), step=1, label="Top-k")
        max_new_tokens = gr.Slider(1, 128, value=int(init.get("max_new_tokens", 16)), step=1, label="Max new tokens")
        chat = gr.ChatInterface(fn=respond_ctrl, additional_inputs=[temperature, top_k, max_new_tokens], title="OumnixSimpleAI – Chat Web", type='messages')
    return demo

chat_interface = make_interface()

if __name__ == "__main__":
    if os.environ.get("OUMNIX_WEB_CONTROLS", "0") == "1":
        make_interface_with_controls().launch()
    else:
        chat_interface.launch()
