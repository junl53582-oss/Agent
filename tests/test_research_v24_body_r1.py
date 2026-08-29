import json
from datetime import datetime, timezone

from research_v24_body_r1.runner import _attempt_id, _status


def test_attempt_ids_are_unique_and_sortable():
    first = _attempt_id(datetime(2026, 8, 29, 7, 0, 0, 1, tzinfo=timezone.utc))
    second = _attempt_id(datetime(2026, 8, 29, 7, 0, 0, 2, tzinfo=timezone.utc))
    assert first < second and first.endswith("Z") and second.endswith("Z")


def test_runtime_status_is_replaceable_not_a_frozen_result(tmp_path, monkeypatch):
    import research_v24_body_r1.runner as runner
    monkeypatch.setattr(runner, "DIRECTORY", tmp_path)
    _status("first", attempt_id="one")
    _status("second", attempt_id="two")
    status = json.loads((tmp_path / "runtime_status.json").read_text(encoding="utf-8"))
    assert status["stage"] == "second" and status["attempt_id"] == "two"

