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
        self._fisher = defaultdict(lambda: None)

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
                    g2 = p.grad.detach() ** 2
                    if self._fisher[n] is None:
                        self._fisher[n] = g2
                    else:
                        self._fisher[n] = self._fisher[n] + g2
        for n, v in list(self._fisher.items()):
            if v is None:
                self._fisher[n] = torch.zeros_like(self.model.state_dict()[n])
            else:
                self._fisher[n] = v / max(len(dataloader), 1)
        self._old_params = {n: p.clone().detach() for n, p in self.model.named_parameters()}

    def ewc_loss(self):
        """
"""
        loss = 0.0
        for n, p in self.model.named_parameters():
            fisher = self._fisher.get(n, None)
            old = self._old_params.get(n, None)
            if fisher is None or old is None:
                continue
            loss = loss + (fisher * (p - old) ** 2).sum()
        return self.ewc_lambda * loss
