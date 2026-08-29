import json
import tempfile
import unittest
from pathlib import Path

from research_status import build_status, read_json, runtime_status


class ResearchStatusTests(unittest.TestCase):
    def setUp(self):
        self.started = {"pid": 123, "started_at_utc": "2026-08-29T00:00:10+00:00", "lock_sha256": "abc"}
        self.runtime = {"pid": 123, "stage": "loading_dataset"}
        self.process = {"state": "alive", "pid": 123, "created_at": "2026-08-29T00:00:00+00:00", "command_line": "python.exe -B -u -m research_v20r1.cli run"}

    def status(self, process=None):
        return runtime_status(self.runtime, self.started, {}, "research_v20r1", lambda pid: process or self.process)

    def test_real_identity_not_just_pid(self):
        stage, process = self.status()
        self.assertEqual(stage, "loading_dataset")
        self.assertTrue(process["identity_verified"])
        self.assertNotIn("command_line", process)

    def test_exited_process_not_reported_running(self):
        self.assertEqual(self.status({"state": "exited"})[0], "interrupted")

    def test_pid_reuse_rejected_by_creation_time(self):
        process = {**self.process, "created_at": "2026-08-29T00:01:00+00:00"}
        self.assertEqual(self.status(process)[0], "process_identity_mismatch")

    def test_other_module_is_not_same_job(self):
        process = {**self.process, "command_line": "python.exe -m research_v20.cli run"}
        self.assertEqual(self.status(process)[0], "process_identity_mismatch")

    def test_permission_error_is_unknown_not_exited(self):
        self.assertEqual(self.status({"state": "unknown"})[0], "process_unverified")

    def test_finished_without_report_not_success(self):
        self.runtime["stage"] = "complete"
        self.assertEqual(self.status()[0], "incomplete_report")

    def test_report_must_match_run_lock(self):
        stage, _ = runtime_status(self.runtime, self.started, {"lock_sha256": "wrong"}, "research_v20r1")
        self.assertEqual(stage, "invalid_status")

    def test_corrupt_status_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            for content in ("{", "[]", "null"):
                path.write_text(content, encoding="utf-8")
                self.assertIn("read_error", read_json(path))

    def test_registry_selects_revision_without_promoting_it(self):
        with tempfile.TemporaryDirectory() as folder:
            artifacts = Path(folder) / "artifacts"
            revision = artifacts / "research_v20r1"
            revision.mkdir(parents=True)
            (artifacts / "active_research.json").write_text(json.dumps({"package": "research_v20r1", "active_model": "V20r1", "replacement_approved": True}))
            (revision / "plan.lock.json").write_text("{}")
            (revision / "runtime_status.json").write_text(json.dumps(self.runtime))
            (revision / "run.started.json").write_text(json.dumps(self.started))
            result = build_status(folder, process_reader=lambda pid: self.process)
            self.assertEqual(result["candidate_model"], "V20r1")
            self.assertEqual(result["active_model"], "V6")
            self.assertFalse(result["replacement_approved"])
            self.assertFalse(result["execution_authorized"])

    def test_registry_cannot_escape_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            artifacts = Path(folder) / "artifacts"
            artifacts.mkdir()
            (artifacts / "active_research.json").write_text(json.dumps({"package": "../../outside"}))
            self.assertEqual(build_status(folder)["candidate_stage"], "invalid_registry")


if __name__ == "__main__":
    unittest.main()
