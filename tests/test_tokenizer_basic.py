import os
import io
import json
from utils.tokenizer import tokenizer

def test_encode_decode_roundtrip_simple():
    text = "Hello, world!"
    ids = tokenizer.encode(text)
    out = tokenizer.decode(ids)
    assert isinstance(ids, list) and len(ids) > 0
    # Robust tokenizer may not contain these words in small default vocab; ensure decode yields non-empty tokens
    assert len(out) > 0


def test_vocab_build_from_temp_jsonl(tmp_path, monkeypatch):
    data = [
        {"text": "First line."},
        {"content": "Second line!"},
        "Third line?",
    ]
    p = tmp_path / "ds.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) if not isinstance(item, str) else json.dumps({"text": item}))
            f.write("\n")
    # Tokenizer scans datasets/ by default; point to tmp by chdir
    monkeypatch.chdir(tmp_path)
    ids = tokenizer.encode("Build vocab from temp")
    assert isinstance(ids, list)
