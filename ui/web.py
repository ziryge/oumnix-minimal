"""
"""
import torch
import gradio as gr
from core.model import OumnixSimpleAI
from neuro.chemistry import NeuroState
from utils.tokenizer import tokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OumnixSimpleAI(vocab_size=tokenizer.vocab_size).to(DEVICE)
state = NeuroState()

def respond(message, history):
    ids = torch.tensor([tokenizer.encode(message)], device=DEVICE)
    logits = model(ids)
    probs = torch.softmax(logits / state.temperature(), dim=-1)
    next_token = torch.multinomial(probs[0, -1], 1)
    reply = tokenizer.decode(next_token.tolist())
    
    reward = 1.0 if "thanks" in reply.lower() else 0.0
    surprise = -torch.log(probs[0, -1, next_token]).item()
    state.update(reward, surprise)
    return reply, history + [(message, reply)]

chat_interface = gr.ChatInterface(fn=respond, title="OumnixSimpleAI – Chat Web")

if __name__ == "__main__":
    chat_interface.launch()
