import json
import os
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

from .tokenizer import tokenizer

class TextLineDataset(Dataset):
    """Tolerant text loader that extracts text/content from .json/.jsonl/.txt files."""
    def __init__(self, dataset_dir: str = "datasets"):
        self.dataset_dir = Path(dataset_dir)
        self.samples: List[List[int]] = []
        self._load_all()

    def _load_all(self):
        files_read = 0
        lines_total = 0
        lines_kept = 0
        for file_path in self.dataset_dir.rglob("*.*"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    files_read += 1
                    for line in f:
                        lines_total += 1
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                text = obj.get("text") or obj.get("content") or json.dumps(obj)
                            else:
                                text = str(obj)
                        except json.JSONDecodeError:
                            text = line
                        token_ids = tokenizer.encode(text)
                        if token_ids:
                            self.samples.append(token_ids)
                            lines_kept += 1
            except Exception:
                continue
        _ = (files_read, lines_total, lines_kept)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):  # -> torch.Tensor
        ids = torch.tensor(self.samples[idx], dtype=torch.long)
        return ids
