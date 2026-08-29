import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from research_log_viewer import APP_ID, SnapshotReader, make_server, tail_file


class LiveLogsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "artifacts/research_v20r2"
        self.folder.mkdir(parents=True)
        (self.root / "artifacts/active_research.json").write_text(json.dumps({"package": "research_v20r2"}))
        self.builder_calls = 0
        def status(root):
            self.builder_calls += 1
            return {"candidate_model": "V20r2", "candidate_stage": "loading_dataset",
                    "candidate_process": {"state": "alive", "pid": 123, "identity_verified": True}, "active_model": "V6"}
        self.reader = SnapshotReader(self.root, status)

    def test_append_appears_without_restarting_reader(self):
        path = self.folder / "run_stdout.log"
        path.write_bytes(b"first\n")
        self.assertEqual(self.reader.read()["stdout"]["text"], "first\n")
        with path.open("a", encoding="utf-8", newline="") as handle:
            handle.write("second\n")
        self.assertEqual(self.reader.read()["stdout"]["text"], "first\nsecond\n")
        self.assertEqual(self.builder_calls, 1)

    def test_rotation_and_utf8(self):
        path = self.folder / "run_stdout.log"
        path.write_text("旧日志\n" * 20, encoding="utf-8")
        self.assertTrue(tail_file(path, 40)["truncated"])
        path.write_bytes("新日志\n".encode("utf-8"))
        self.assertEqual(tail_file(path)["text"], "新日志\n")

    def test_missing_log_is_safe_empty_state(self):
        result = self.reader.read()
        self.assertEqual(result["stderr"]["text"], "")
        self.assertFalse(result["execution_authorized"])

    def test_invalid_candidate_cannot_read_arbitrary_files(self):
        (self.root / "artifacts/active_research.json").write_text(json.dumps({"package": "../../outside"}))
        with self.assertRaises(ValueError):
            self.reader.read()

    def test_runtime_change_invalidates_process_cache(self):
        self.reader.read()
        (self.folder / "runtime_status.json").write_text(json.dumps({"stage": "fitting"}))
        self.reader.read()
        self.assertEqual(self.builder_calls, 2)

    def test_http_streaming_poll_and_read_only_routes(self):
        server = make_server(self.root, 0, self.reader)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            def request(path, method="GET", headers=None):
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                connection.request(method, path, headers=headers or {})
                response = connection.getresponse()
                data = response.read()
                status = response.status
                connection.close()
                return status, data
            self.assertEqual(server.server_address[0], "127.0.0.1")
            self.assertEqual(json.loads(request("/health")[1])["app"], APP_ID)
            first = json.loads(request("/api/logs")[1])
            (self.folder / "run_stdout.log").write_text("live update", encoding="utf-8")
            second = json.loads(request("/api/logs")[1])
            self.assertNotEqual(first["stdout"], second["stdout"])
            self.assertEqual(request("/../../.env")[0], 404)
            self.assertEqual(request("/api/logs", "POST")[0], 501)
            self.assertEqual(request("/api/logs", headers={"Host": "example.org"})[0], 403)
            self.assertEqual(request("/api/logs", headers={"Sec-Fetch-Site": "cross-site"})[0], 403)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
