import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Dict, Optional

# Simple OMNX (Oumnix Metrics eXporter) text exposition format
# Usage:
#   start_omnx_exporter(lambda: {"tokens_per_sec": 123.4, "ms_per_token": 3.21}, port=9000)
# Then scrape http://localhost:9000/metrics

class _OMNXHandler(BaseHTTPRequestHandler):
    registry_getter: Callable[[], Dict[str, float]] = lambda: {}

    def do_GET(self):  # noqa: N802
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        try:
            metrics = type(self).registry_getter()
            body_lines = []
            for k, v in metrics.items():
                try:
                    num = float(v)
                except Exception:
                    continue
                # Replace illegal chars
                safe_key = k.replace(" ", "_").replace("/", "_").replace("-", "_")
                body_lines.append(f"omnx_{safe_key} {num}")
            body = ("\n".join(body_lines) + "\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception:
            self.send_response(500)
            self.end_headers()


def start_omnx_exporter(get_metrics: Callable[[], Dict[str, float]], port: Optional[int] = None) -> threading.Thread:
    """Start a background OMNX metrics exporter if OUMNIX_OMNX=1.

    Returns the thread if started, else a dummy non-started thread object.
    """
    enabled = os.environ.get("OUMNIX_OMNX", "0") == "1"
    port = int(os.environ.get("OUMNIX_OMNX_PORT", "0")) or (port or 9100)
    if not enabled:
        t = threading.Thread(target=lambda: None, daemon=True)
        return t

    def _serve():
        server = HTTPServer(("0.0.0.0", port), _OMNXHandler)
        _OMNXHandler.registry_getter = get_metrics
        # serve forever; swallow exceptions on shutdown
        try:
            server.serve_forever(poll_interval=0.5)
        except Exception:
            pass

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    return t
