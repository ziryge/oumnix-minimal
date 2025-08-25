"""
"""
import json
import os
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

from .tokenizer import tokenizer

class TextLineDataset(Dataset):
    """
"""
    def __init__(self, dataset_dir: str = "datasets"):
        self.dataset_dir = Path(dataset_dir)
        self.samples: List[List[int]] = []
        self._load_all()

    def _load_all(self):
        for file_path in self.dataset_dir.rglob("*.*"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
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
            except Exception as e:
                
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = torch.tensor(self.samples[idx], dtype=torch.long)
        return ids
