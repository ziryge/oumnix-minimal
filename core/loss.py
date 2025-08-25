"""
"""
import torch
import torch.nn.functional as F

def free_energy_loss(logits: torch.Tensor, target: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
"""
    
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction="mean")
    
    
    entropy = 0.5 * torch.log(2 * torch.pi * torch.exp(torch.tensor(1.0)) * sigma).sum(-1).mean()
    
    return ce - 0.01 * entropy
