import os
import json
import shutil
import numpy as np
import torch
from pathlib import Path

from utils.dataset import TextLineDataset
from utils.tokenizer import tokenizer
from memory.episodic import EpisodicMemory

TMP_DIR = Path("tmp_rovodev_testdata")

def setup_module(module=None):
    TMP_DIR.mkdir(exist_ok=True)
    with open(TMP_DIR / "a.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "hello world"}) + "\n")
        f.write("not-json line\n")
    with open(TMP_DIR / "b.txt", "w", encoding="utf-8") as f:
        f.write("another line\n")

def teardown_module(module=None):
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)

def test_textline_dataset_tolerance():
    ds = TextLineDataset(dataset_dir=str(TMP_DIR))
    assert len(ds) >= 2
    item = ds[0]
    assert isinstance(item, torch.Tensor)
    assert item.dtype == torch.long


def test_episodic_memory_roundtrip(tmp_path):
    mem = EpisodicMemory(dim=4, normalize=True)
    v = np.random.randn(3, 4).astype('float32')
    mem.add(v, ["a", "b", "c"])
    q = np.random.randn(4).astype('float32')
    res = mem.search(q, k=2)
    assert len(res) <= 2
    save_dir = tmp_path / "mem"
    mem.save(str(save_dir))
    mem2 = EpisodicMemory(dim=4, normalize=True)
    mem2.load(str(save_dir))
    res2 = mem2.search(q, k=2)
    assert len(res2) <= 2
