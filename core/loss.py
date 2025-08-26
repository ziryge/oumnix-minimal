import torch
import torch.nn.functional as F

def free_energy_loss(logits: torch.Tensor, target: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction="mean")
    eps = 1e-8
    sigma = torch.clamp(sigma, min=eps)
    device = logits.device
    const = torch.log(2 * torch.pi * torch.exp(torch.tensor(1.0, device=device)))
    entropy = 0.5 * (const + torch.log(sigma)).sum(-1).mean()
    return ce - 0.01 * entropy
