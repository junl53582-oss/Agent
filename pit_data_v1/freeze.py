from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .core import ObservationSettings, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _paths(settings: ObservationSettings) -> list[Path]:
    return [
        *sorted((ROOT / "pit_data_v1").glob("*.py")),
        ROOT / "tests/test_pit_data_v1.py",
        ROOT / settings.artifact_root / "protocol.json",
        ROOT / settings.artifact_root / "test_receipt.json",
        ROOT / settings.membership_path,
        ROOT / settings.industry_path,
        ROOT / "artifacts/prediction_v30r1/validation.lock.json",
    ]


def create_lock(settings: ObservationSettings | None = None) -> dict:
    settings = settings or ObservationSettings()
    directory = ROOT / settings.artifact_root
    target = directory / "data.lock.json"
    if target.exists():
        raise RuntimeError("PIT data V1 is already frozen")
    receipt = json.loads((directory / "test_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("passed") is not True or receipt.get("live_source_run_before_freeze") is not False:
        raise RuntimeError("tests must pass before source observation")
    paths = _paths(settings)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError("cannot freeze missing inputs: " + ", ".join(missing))
    payload = {
        "version": settings.version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "prospective_data_capture_only_no_model_or_return_access",
        "allowed_dynamic_data_root": settings.data_root.as_posix(),
        "allowed_dynamic_artifact_roots": [
            (settings.artifact_root / "observations").as_posix(),
            (settings.artifact_root / "failures").as_posix(),
        ],
        "files": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths},
        "historical_pit_verified": False,
        "model_training_ready": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    directory.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "data.lock.sha256").write_text(sha256_file(target) + "\n", encoding="utf-8")
    return verify_lock(settings)


def verify_lock(settings: ObservationSettings | None = None) -> dict:
    settings = settings or ObservationSettings()
    directory = ROOT / settings.artifact_root
    target = directory / "data.lock.json"
    actual = sha256_file(target)
    expected = (directory / "data.lock.sha256").read_text(encoding="utf-8").strip()
    if actual != expected:
        raise RuntimeError("PIT data V1 lock file changed")
    payload = json.loads(target.read_text(encoding="utf-8"))
    mismatches = [
        name
        for name, digest in payload["files"].items()
        if not (ROOT / name).exists() or sha256_file(ROOT / name) != digest
    ]
    if mismatches:
        raise RuntimeError("PIT data V1 frozen input changed: " + ", ".join(mismatches))
    return {
        "lock_sha256": actual,
        "frozen_inputs_intact": True,
        "historical_pit_verified": False,
        "model_training_ready": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
