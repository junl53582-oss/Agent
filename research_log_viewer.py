"""Loopback-only, read-only live log viewer. Never starts or changes research jobs."""
import argparse
import json
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from research_status import build_status, read_json


APP_ID = "stockpilot-readonly-live-logs-v1"
SHANGHAI = timezone(timedelta(hours=8))
PAGE = Path(__file__).with_name("research_log_viewer.html")


def tail_file(path, limit=131072):
    """Bound I/O and tolerate logs being appended; never open for writing."""
    try:
        with Path(path).open("rb") as handle:
            size = handle.seek(0, 2)
            start = max(0, size - limit)
            handle.seek(start)
            raw = handle.read(limit)
        if start:
            raw = raw.partition(b"\n")[2]
        return {"text": raw.decode("utf-8", errors="replace"), "size": size, "truncated": start > 0}
    except FileNotFoundError:
        return {"text": "", "size": 0, "truncated": False}


class SnapshotReader:
    def __init__(self, root, status_builder=build_status):
        self.root = Path(root).resolve()
        self.status_builder = status_builder
        self.lock = threading.Lock()
        self.cached = None
        self.cached_key = None
        self.checked_at = 0.0

    def read(self):
        registry = read_json(self.root / "artifacts/active_research.json")
        package = registry.get("package", "")
        if not isinstance(package, str) or not re.fullmatch(r"research_v[0-9]+(?:r[0-9]+)?", package):
            raise ValueError("研究候选配置无效，停止读取日志")
        folder = self.root / "artifacts" / package
        runtime = read_json(folder / "runtime_status.json")
        key = (package, json.dumps(runtime, sort_keys=True))
        with self.lock:
            if key != self.cached_key or time.monotonic() - self.checked_at >= 5:
                self.cached = self.status_builder(self.root)
                self.cached_key = key
                self.checked_at = time.monotonic()
            status = self.cached
        return {"app": APP_ID, "refreshed_at": datetime.now(SHANGHAI).isoformat(),
                "model": status["candidate_model"], "stage": status["candidate_stage"],
                "process": status["candidate_process"], "runtime": runtime,
                "stdout": tail_file(folder / "run_stdout.log"),
                "stderr": tail_file(folder / "run_stderr.log"),
                "report_ready": status["candidate_stage"] == "report_recorded",
                "active_model": status["active_model"], "execution_authorized": False}


def make_server(root, port=8765, reader=None):
    snapshot_reader = reader or SnapshotReader(root)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_body(self, code, body, kind):
            encoded = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", kind + "; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            port_number = self.server.server_port
            if self.headers.get("Host", "") not in (f"127.0.0.1:{port_number}", f"localhost:{port_number}"):
                self.send_body(403, "Loopback host required", "text/plain")
                return
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                self.send_body(403, "Same-origin access required", "text/plain")
                return
            path = urlsplit(self.path).path
            if path == "/":
                self.send_body(200, PAGE.read_bytes(), "text/html")
            elif path == "/health":
                self.send_body(200, json.dumps({"app": APP_ID}), "application/json")
            elif path == "/api/logs":
                try:
                    payload = snapshot_reader.read()
                    self.send_body(200, json.dumps(payload, ensure_ascii=False), "application/json")
                except (ValueError, OSError, KeyError) as error:
                    self.send_body(503, json.dumps({"error": str(error)}, ensure_ascii=False), "application/json")
            else:
                self.send_body(404, "Not found", "text/plain")

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = make_server(Path(__file__).parent, args.port)
    print(f"Read-only live logs: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
