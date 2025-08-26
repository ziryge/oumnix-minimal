import torch
from memory.short_term import ShortTermBuffer


def test_short_term_trim_overflow():
    buf = ShortTermBuffer(max_length=5)
    ids = torch.arange(0, 8)
    embeds = torch.randn(8, 4)
    buf.add(ids, embeds)
    got_ids, got_embeds = buf.get()
    assert got_ids.numel() == 5 and got_embeds.shape == (5, 4)
    assert torch.equal(got_ids, torch.arange(3, 8))
