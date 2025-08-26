import torch
from memory.short_term import ShortTermBuffer


def test_short_term_buffer_add_and_get():
    buf = ShortTermBuffer(max_length=10)
    ids = torch.arange(0, 6)
    embeds = torch.randn(6, 8)
    buf.add(ids, embeds)
    got_ids, got_embeds = buf.get()
    assert got_ids is not None and got_embeds is not None
    assert got_ids.numel() == 6 and got_embeds.shape == (6, 8)
