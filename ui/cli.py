import sys
import torch
from core.model import OumnixSimpleAI
from core.loss import free_energy_loss
from neuro.chemistry import NeuroState
from memory.short_term import ShortTermBuffer
from utils.tokenizer import tokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = OumnixSimpleAI(vocab_size=tokenizer.vocab_size).to(DEVICE)
state = NeuroState()
buffer = ShortTermBuffer()

def chat():
    print("=== OumnixSimpleAI – terminal chat (Ctrl-C to exit) ===")
    while True:
        try:
            user = input("You: ")
        except KeyboardInterrupt:
            print("\nShutting down.")
            break
        
        ids = torch.tensor([tokenizer.encode(user)], device=DEVICE)
        
        sigma = torch.full_like(ids.unsqueeze(-1).float(), 0.1)
        
        logits = model(ids)
        
        _ = free_energy_loss(logits, ids, sigma)
        
        probs = torch.softmax(logits / state.temperature(), dim=-1)
        next_token = torch.multinomial(probs[0, -1], num_samples=1)
        reply = tokenizer.decode(next_token.tolist())
        print(f"AI: {reply}")
        
        reward = 1.0 if "thank you" in reply.lower() else 0.0
        surprise = -torch.log(probs[0, -1, next_token]).item()
        state.update(reward, surprise)
        
        with torch.no_grad():
            embed_vec = model.embed(ids).squeeze(0)
        buffer.add(ids.squeeze(0), embed_vec)

if __name__ == "__main__":
    chat()
