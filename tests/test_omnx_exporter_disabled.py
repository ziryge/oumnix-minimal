import os
from utils.omnx_exporter import start_omnx_exporter


def test_omnx_exporter_disabled_returns_thread(monkeypatch):
    monkeypatch.setenv("OUMNIX_OMNX", "0")
    t = start_omnx_exporter(lambda: {"x": 1.0}, port=9101)
    # Should return a Thread object (daemon, not started) in disabled mode
    assert hasattr(t, "is_alive")
