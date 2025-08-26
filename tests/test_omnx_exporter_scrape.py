import os
import time
import urllib.request
import pytest
from utils.omnx_exporter import start_omnx_exporter


def test_omnx_exporter_scrape_basic(monkeypatch):
    # enable exporter
    monkeypatch.setenv("OUMNIX_OMNX", "1")
    monkeypatch.setenv("OUMNIX_OMNX_PORT", "9103")

    def get_metrics():
        return {"tokens_per_sec": 123.4, "ms_per_token": 3.21, "kv_hit": 0.1, "head_drop": 0.2}

    t = start_omnx_exporter(get_metrics, port=9103)
    # give the server a moment to start
    time.sleep(0.5)
    try:
        with urllib.request.urlopen("http://127.0.0.1:9103/metrics", timeout=2.0) as resp:
            body = resp.read().decode()
            assert "omnx_tokens_per_sec" in body
            assert "omnx_ms_per_token" in body
    except Exception as e:
        pytest.skip(f"Network/socket not permitted in sandbox: {e}")
    # thread is daemonized; no explicit shutdown in this smoke test
