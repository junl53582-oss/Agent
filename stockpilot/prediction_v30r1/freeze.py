from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from stockpilot.prediction.freeze import digest

from .config import V30R1Settings


def _inputs(settings: V30R1Settings) -> list[Path]:
    return [
        settings.artifact_dir / "protocol.json",
        settings.parent_dir / "validation.lock.json",
        *sorted(Path("stockpilot/prediction_v30r1").glob("*.py")),
        Path("tests/test_prediction_v30r1.py"),
    ]


def create_plan_lock(settings: V30R1Settings | None = None) -> dict:
    settings = settings or V30R1Settings()
    target = settings.artifact_dir / "plan.lock.json"
    if target.exists():
        raise RuntimeError("V30r1 plan lock already exists")
    files = _inputs(settings)
    payload = {
        "version": settings.version,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "performance_not_read_before_freeze": True,
        "parent_v30_result_already_known_for_diagnosis": True,
        "execution_authorized": False,
        "files": {str(path).replace("\\", "/"): digest(path) for path in files},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload | {"lock_sha256": digest(target)}


def verify_plan_lock(settings: V30R1Settings | None = None) -> dict:
    settings = settings or V30R1Settings()
    target = settings.artifact_dir / "plan.lock.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    mismatches = [name for name, expected in payload["files"].items() if not Path(name).exists() or digest(Path(name)) != expected]
    return {"intact": not mismatches, "mismatches": mismatches, "lock_sha256": digest(target)}
