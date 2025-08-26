import json
from pathlib import Path
from utils.tokenizer import SimpleTokenizer


def test_tokenizer_build_small_vocab(tmp_path, monkeypatch):
    ds = tmp_path / "datasets"
    ds.mkdir()
    # create a small jsonl file to drive vocab
    data = [
        {"text": "wait"},
        {"text": "yes"},
        {"text": "really"}
    ]
    f = ds / "a.jsonl"
    with f.open("w", encoding="utf-8") as fh:
        for item in data:
            fh.write(json.dumps(item) + "\n")
    tok = SimpleTokenizer(dataset_dir=str(ds), vocab_path=str(tmp_path / "utils" / "vocab.pkl"), max_files=None)
    ids = tok.encode("wait yes really")
    out = tok.decode(ids)
    assert "wait" in out and "yes" in out and "really" in out
