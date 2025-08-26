import time
from collections import deque

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

class MovingAverage:
    def __init__(self, window: int = 100):
        self.window = window
        self.q = deque(maxlen=window)
        self.sum = 0.0

    def add(self, x: float):
        if len(self.q) == self.q.maxlen:
            self.sum -= self.q[0]
        self.q.append(x)
        self.sum += x

    def value(self) -> float:
        if not self.q:
            return 0.0
        return self.sum / len(self.q)

class Timer:
    def __init__(self):
        self.t = None

    def start(self):
        self.t = time.perf_counter()

    def stop(self) -> float:
        if self.t is None:
            return 0.0
        dt = time.perf_counter() - self.t
        self.t = None
        return dt

class PerfTracker:
    """Tracks rolling performance metrics and resource usage snapshots."""
    def __init__(self, window: int = 100):
        self.tokens_per_sec = MovingAverage(window)
        self.ms_per_token = MovingAverage(window)
        self.vram_gb = 0.0
        self.kv_hits = MovingAverage(window)
        self.head_drops = MovingAverage(window)

    def update(self, tokens: int, seconds: float, kv_hit: float = 0.0, head_drop: float = 0.0):
        if seconds <= 0:
            return
        tps = tokens / seconds
        self.tokens_per_sec.add(tps)
        self.ms_per_token.add((seconds / tokens) * 1000.0)
        self.kv_hits.add(kv_hit)
        self.head_drops.add(head_drop)

    def snapshot(self) -> dict:
        try:
            if torch is not None and hasattr(torch, 'cuda') and torch.cuda and torch.cuda.is_available():
                self.vram_gb = torch.cuda.max_memory_allocated() / 1e9
        except Exception:
            pass
        return {
            'tokens_per_sec': self.tokens_per_sec.value(),
            'ms_per_token': self.ms_per_token.value(),
            'vram_gb': self.vram_gb,
            'kv_hit': self.kv_hits.value(),
            'head_drop': self.head_drops.value(),
        }
