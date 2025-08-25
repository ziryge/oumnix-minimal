"""
"""
import os
import re
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import List, Iterable, Optional, Dict, Any

class SimpleTokenizer:
    def __init__(self, dataset_dir: str = "datasets", vocab_path: str | None = None,
                 max_vocab_size: int = 50000, max_files: int | None = 1,
                 max_lines_per_file: int | None = 100000, lowercase: bool = True):
        self.dataset_dir = Path(dataset_dir)
        self.vocab_path = Path(vocab_path) if vocab_path else None
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        
        self.special_tokens = {self.pad_token: 0, self.unk_token: 1}
        self.id2token: List[str] = []
        self.token2id: dict[str, int] = {}
        self.max_vocab_size = max_vocab_size
        self.max_files = max_files
        self.max_lines_per_file = max_lines_per_file
        self.lowercase = lowercase
        
        self._version = "regex_ws_v3"
        if self.vocab_path and self.vocab_path.exists():
            loaded_ok = self.load(self.vocab_path)
            if not loaded_ok:
                self.build_vocab()
        else:
            self.build_vocab()

    
    
    
    def build_vocab(self):
        """
"""
        counter: Counter[str] = Counter()
        files_read = 0
        for file_path in self.dataset_dir.rglob("*.*"):
            if self.max_files is not None and files_read >= self.max_files:
                break
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    line_count = 0
                    for raw in f:
                        if self.max_lines_per_file is not None and line_count >= self.max_lines_per_file:
                            break
                        raw = raw.strip()
                        if not raw:
                            continue
                        line_count += 1
                        for text in self._extract_texts_from_line(raw):
                            if self.lowercase:
                                text = text.lower()
                            for tok in self._basic_tokenize(text):
                                counter[tok] += 1
                files_read += 1
            except Exception:
                continue
        
        most_common = [t for t, _ in counter.most_common(self.max_vocab_size)]
        
        self.id2token = [self.pad_token, self.unk_token] + most_common
        self.token2id = {tok: idx for idx, tok in enumerate(self.id2token)}
        if self.vocab_path:
            self.save(self.vocab_path)

    def save(self, path: Path):
        meta = {
            "id2token": self.id2token,
            "token2id": self.token2id,
            "version": self._version,
            "lowercase": self.lowercase,
            "special_tokens": self.special_tokens,
            "max_vocab_size": self.max_vocab_size,
        }
        with open(path, "wb") as f:
            pickle.dump(meta, f)

    def load(self, path: Path) -> bool:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            
            if not isinstance(data, dict) or "id2token" not in data or "token2id" not in data:
                return False
            if data.get("version") != self._version:
                return False
            if data.get("lowercase") != self.lowercase:
                return False
            if data.get("special_tokens") != self.special_tokens:
                return False
            
            self.id2token = list(data["id2token"])
            self.token2id = dict(data["token2id"])
            
            max_allowed = self.max_vocab_size + len(self.special_tokens)
            if len(self.id2token) > max_allowed:
                self.id2token = self.id2token[:max_allowed]
                self.token2id = {tok: idx for idx, tok in enumerate(self.id2token)}
            return True
        except Exception:
            return False

    
    
    
    @property
    def vocab_size(self) -> int:
        return len(self.id2token)

    def encode(self, text: str) -> List[int]:
        """
"""
        if self.lowercase:
            text = text.lower()
        tokens = self._basic_tokenize(text)
        unk_id = self.token2id.get(self.unk_token, 1)
        return [self.token2id.get(tok, unk_id) for tok in tokens]

    def decode(self, ids: List[int]) -> str:
        """
"""
        if not ids:
            return ""
        
        toks = []
        for i in ids:
            tok = self.id2token[i] if 0 <= i < len(self.id2token) else self.unk_token
            if tok == self.pad_token:
                continue
            toks.append(tok)
        if not toks:
            return ""
        if len(toks) == 1:
            return toks[0]
        
        attach_to_prev = set(list(".,!?;:%)]}…»\u201d\u2019")) | {"..."}
        no_space_after = set(list("([{«\u201c\u2018"))
        out = []
        for t in toks:
            if not out:
                out.append(t)
                continue
            if t in attach_to_prev:
                out[-1] = out[-1] + t
            elif out[-1] in no_space_after:
                out[-1] = out[-1] + t
            else:
                out.append(t)
        return " ".join(out)

    
    
    
    def _basic_tokenize(self, text: str) -> List[str]:
        """
"""
        
        return re.findall(r"\.\.\.|…|\w+|[^\w\s]", text, flags=re.UNICODE)

    def _extract_texts_from_line(self, raw: str) -> Iterable[str]:
        """
"""
        try:
            obj: Any = json.loads(raw)
        except json.JSONDecodeError:
            return [raw]
        texts: List[str] = []
        if isinstance(obj, dict):
            if "conversations" in obj and isinstance(obj["conversations"], list):
                for turn in obj["conversations"]:
                    if isinstance(turn, dict) and isinstance(turn.get("content"), str):
                        texts.append(turn["content"])
            
            val = obj.get("text") or obj.get("content")
            if isinstance(val, str):
                texts.append(val)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    val = item.get("text") or item.get("content")
                    if isinstance(val, str):
                        texts.append(val)
        else:
            texts.append(str(obj))
        return texts or [raw]


tokenizer = SimpleTokenizer(
    dataset_dir=os.path.join(os.path.dirname(__file__), "..", "datasets"),
    vocab_path=os.path.join(os.path.dirname(__file__), "vocab.pkl"),
    max_vocab_size=30000,
    max_files=None,
    max_lines_per_file=200000,
    lowercase=True
)
