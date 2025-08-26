import torch
from torch.utils.data import DataLoader
from utils.tokenizer import tokenizer
from core.model import OumnixSimpleAI
from memory.consolidator import Consolidator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def simple_collate(batch):
    ids = [torch.tensor(tokenizer.encode(x)) for x in batch]
    max_len = max(len(x) for x in ids)
    pad_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
    ids = [torch.nn.functional.pad(x, (0, max_len - len(x)), value=pad_id) for x in ids]
    ids = torch.stack(ids, dim=0)
    targets = ids.clone()
    return ids.to(DEVICE), targets.to(DEVICE)

def test_consolidator_compute_fisher_and_ewc():
    data = ["alpha beta", "gamma delta", "epsilon zeta", "eta theta"]
    dl = DataLoader(data, batch_size=2, shuffle=False, collate_fn=simple_collate)
    model = OumnixSimpleAI(vocab_size=tokenizer.vocab_size, dim=32, n_layers=1).to(DEVICE)
    cons = Consolidator(model)
    cons.compute_fisher(dl)
    loss = cons.ewc_loss()
    assert torch.is_tensor(loss) and loss >= 0
