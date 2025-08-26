import json
import io
import os
from utils.tokenizer import tokenizer, SimpleTokenizer


def test_tokenizer_export_import_roundtrip(tmp_path):
    tk = SimpleTokenizer(dataset_dir="datasets")
    tk.unfreeze()
    tk.add_token("Ωmega")
    path = tmp_path / "vocab.json"
    tk.export_json(str(path))
    tk2 = SimpleTokenizer(dataset_dir="datasets")
    tk2.import_json(str(path))
    assert tk2.vocab_size >= tk.vocab_size


def test_tokenizer_freeze_and_add_raises():
    tk = SimpleTokenizer(dataset_dir="datasets")
    tk.freeze()
    try:
        tk.add_token("newtoken")
        ok = True
    except Exception:
        ok = False
    assert not ok


def test_tokenizer_unicode_and_unk():
    text = "Wait… really?! – Ωmega"
    ids = tokenizer.encode(text)
    out = tokenizer.decode(ids)
    assert isinstance(out, str) and len(out) > 0
