import os
import json
from ui.web import make_interface_with_controls

def test_web_controls_persist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    demo = make_interface_with_controls()
    path = tmp_path / ".oumnix_web_settings.json"
    assert demo is not None
    data = {"temperature": 1.2, "top_k": 33, "max_new_tokens": 9}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    demo = make_interface_with_controls()
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert set(loaded.keys()) == {"temperature", "top_k", "max_new_tokens"}
