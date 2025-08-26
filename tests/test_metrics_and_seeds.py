import os
from utils.metrics import Timer
from utils.seeds import set_seed


def test_timer_without_start():
    t = Timer()
    assert t.stop() == 0.0
    t.start()
    dt = t.stop()
    assert dt >= 0.0


def test_set_seed_env_and_flags(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    set_seed(123, deterministic=True)
    assert os.environ.get("PYTHONHASHSEED") == "123"
