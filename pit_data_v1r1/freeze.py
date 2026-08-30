from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pit_data_v1.core import sha256_file
from pit_data_v1.freeze import verify_lock as verify_parent

from .core import ObservationSettings


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "artifacts/pit_data_v1"


def create_lock(settings: ObservationSettings | None = None) -> dict:
    settings = settings or ObservationSettings()
    directory = ROOT / settings.artifact_root
    target = directory / "data.lock.json"
    if target.exists():
        raise RuntimeError("PIT data V1r1 is already frozen")
    if (ROOT / settings.data_root).exists() and any((ROOT / settings.data_root).rglob("*")):
        raise RuntimeError("V1r1 dynamic data root must be empty before freeze")
    parent = verify_parent()
    failures = sorted((PARENT / "failures").glob("*.json"))
    if len(failures) != 1:
        raise RuntimeError("expected exactly one preserved V1 failure")
    failure = json.loads(failures[0].read_text(encoding="utf-8"))
    if "pagination mismatch" not in failure.get("error", ""):
        raise RuntimeError("V1 failure evidence changed")
    receipt = json.loads((directory / "test_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("passed") is not True or receipt.get("live_source_run_before_freeze") is not False:
        raise RuntimeError("repair tests must pass before live observation")
    files = [
        *sorted((ROOT / "pit_data_v1r1").glob("*.py")),
        ROOT / "tests/test_pit_data_v1r1.py",
        directory / "protocol.json",
        directory / "test_receipt.json",
        PARENT / "data.lock.json",
        PARENT / "data.lock.sha256",
        failures[0],
        ROOT / settings.membership_path,
        ROOT / settings.industry_path,
        ROOT / "artifacts/prediction_v30r1/validation.lock.json",
    ]
    payload = {
        "version": settings.version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "pagination_only_repair_prospective_capture_no_model_access",
        "parent_lock_sha256": parent["lock_sha256"],
        "parent_failure_preserved": True,
        "allowed_dynamic_data_root": settings.data_root.as_posix(),
        "allowed_dynamic_artifact_roots": [
            (settings.artifact_root / "observations").as_posix(),
            (settings.artifact_root / "failures").as_posix(),
        ],
        "files": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in files},
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
    parent = verify_parent()
    directory = ROOT / settings.artifact_root
    target = directory / "data.lock.json"
    actual = sha256_file(target)
    if actual != (directory / "data.lock.sha256").read_text(encoding="utf-8").strip():
        raise RuntimeError("PIT data V1r1 lock file changed")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload["parent_lock_sha256"] != parent["lock_sha256"]:
        raise RuntimeError("PIT data V1 parent lock changed")
    mismatches = [
        name
        for name, digest in payload["files"].items()
        if not (ROOT / name).exists() or sha256_file(ROOT / name) != digest
    ]
    if mismatches:
        raise RuntimeError("PIT data V1r1 frozen input changed: " + ", ".join(mismatches))
    return {
        "lock_sha256": actual,
        "frozen_inputs_intact": True,
        "historical_pit_verified": False,
        "model_training_ready": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
