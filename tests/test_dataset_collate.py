import json
import torch
from utils.dataset import TextLineDataset
from train import collate_fn


def test_textline_dataset_and_collate(tmp_path):
    data_file = tmp_path / "data.jsonl"
    lines = [
        json.dumps({"text": "hello world"}),
        json.dumps({"content": "another line"}),
        json.dumps({"conversations": [{"role": "user", "content": "hi"}]})
    ]
    data_file.write_text("\n".join(lines))
    ds = TextLineDataset(dataset_dir=str(tmp_path))
    assert len(ds) >= 1
    batch = [ds[0], ds[1]]
    ids, targets, masks = collate_fn(batch)
    assert isinstance(ids, torch.Tensor)
    assert ids.shape == targets.shape
