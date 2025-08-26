import os
import time
from utils.logging_utils import get_logger
from utils.metrics import PerfTracker, Timer


def test_rotating_file_handler(tmp_path, monkeypatch):
    log_path = tmp_path / "oumnix.log"
    monkeypatch.setenv("OUMNIX_LOG_FILE", str(log_path))
    logger = get_logger("testlogger")
    logger.info("hello world")
    # Give a tiny moment for handler flush
    time.sleep(0.01)
    assert log_path.exists()
    content = log_path.read_text()
    assert "hello world" in content


def test_perf_tracker_updates_and_snapshot():
    pt = PerfTracker(window=10)
    pt.update(tokens=1000, seconds=1.0, kv_hit=0.1, head_drop=0.2)
    snap = pt.snapshot()
    assert snap["tokens_per_sec"] > 0
    assert snap["ms_per_token"] > 0
