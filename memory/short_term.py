"""
"""
import torch

class ShortTermBuffer:
    def __init__(self, capacity_tokens: int = 120_000, max_length: int | None = None):
        self.capacity = capacity_tokens if max_length is None else int(max_length)
        self.tokens = []  
        self.embeds = []  

    def add(self, ids: torch.Tensor, embeds: torch.Tensor):
        """
"""
        self.tokens.extend(ids.tolist())
        self.embeds.append(embeds.detach().cpu())
        
        excess = len(self.tokens) - self.capacity
        if excess > 0:
            del self.tokens[:excess]
            
            
            while self.embeds and self.embeds[0].shape[0] <= excess:
                excess -= self.embeds[0].shape[0]
                self.embeds.pop(0)
            if excess > 0 and self.embeds:
                self.embeds[0] = self.embeds[0][excess:]

    def get(self):
        """
"""
        if not self.tokens:
            return torch.empty(0, dtype=torch.long), torch.empty(0)
        ids = torch.tensor(self.tokens, dtype=torch.long)
        embeds = torch.cat(self.embeds, dim=0)
        return ids, embeds
