"""
"""
import torch
from torch import nn
from collections import defaultdict

class Consolidator:
    def __init__(self, model: nn.Module, ewc_lambda: float = 0.4):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self._old_params = {}
        self._fisher = defaultdict(lambda: torch.tensor(0.0))

    def compute_fisher(self, dataloader):
        """
"""
        self.model.eval()
        for ids, targets in dataloader:
            self.model.zero_grad()
            logits = self.model(ids)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self._fisher[n] += p.grad.detach() ** 2
        
        for n in self._fisher:
            self._fisher[n] = self._fisher[n] / len(dataloader)
        self._old_params = {n: p.clone().detach() for n, p in self.model.named_parameters()}

    def ewc_loss(self):
        """
"""
        loss = 0.0
        for n, p in self.model.named_parameters():
            loss += (self._fisher[n] * (p - self._old_params[n]) ** 2).sum()
        return self.ewc_lambda * loss
