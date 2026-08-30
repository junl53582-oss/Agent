from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pit_data_v1.core import sha256_file
from pit_data_v1r3.freeze import verify_lock as verify_parent

from .core import ObservationSettings


ROOT = Path(__file__).resolve().parents[1]


def create_lock(settings: ObservationSettings | None = None) -> dict:
    settings = settings or ObservationSettings()
    directory = ROOT / settings.artifact_root
    target = directory / "data.lock.json"
    if target.exists():
        raise RuntimeError("PIT data V2 is already frozen")
    dynamic_root = ROOT / settings.data_root
    if dynamic_root.exists() and any(dynamic_root.rglob("*")):
        raise RuntimeError("V2 dynamic data root must be empty before freeze")
    parent = verify_parent()
    receipt = json.loads((directory / "test_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("passed") is not True or receipt.get("live_source_run_before_freeze") is not False:
        raise RuntimeError("forward collector tests must pass before freeze")
    baseline = ROOT / settings.baseline_root / "20260830T051947935413Z"
    files = [
        *sorted((ROOT / "pit_data_v2").glob("*.py")),
        ROOT / "tests/test_pit_data_v2.py",
        directory / "protocol.json",
        directory / "test_receipt.json",
        directory / "source_status.json",
        ROOT / "artifacts/pit_data_v1r3/data.lock.json",
        ROOT / "artifacts/pit_data_v1r3/data.lock.sha256",
        ROOT / "artifacts/pit_data_v1r3/observations/20260830T051947935413Z.json",
        baseline / "manifest.json",
        baseline / "expectations.csv",
        baseline / "industry_prosperity.csv",
        ROOT / settings.membership_path,
        ROOT / settings.industry_path,
        ROOT / "artifacts/announcement_body_v5r2/data.lock.json",
        ROOT / "artifacts/announcement_body_v5r2/data.lock.sha256",
        ROOT / "artifacts/announcement_body_v5r2/observations/20260830T050907681987Z.json",
    ]
    payload = {
        "version": settings.version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "forward_only_source_isolated_pit_collection_no_model_access",
        "parent_lock_sha256": parent["lock_sha256"],
        "baseline_observation_id": "20260830T051947935413Z",
        "allowed_dynamic_data_root": settings.data_root.as_posix(),
        "allowed_dynamic_artifact_root": (settings.artifact_root / "observations").as_posix(),
        "files": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in files},
        "same_date_recapture_allowed": False,
        "historical_backfill_allowed": False,
        "return_or_label_access_allowed": False,
        "minimum_prospective_observations": settings.minimum_training_observations,
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
        raise RuntimeError("PIT data V2 lock file changed")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload["parent_lock_sha256"] != parent["lock_sha256"]:
        raise RuntimeError("PIT data V1r3 parent lock changed")
    mismatches = [name for name, digest in payload["files"].items() if not (ROOT / name).exists() or sha256_file(ROOT / name) != digest]
    if mismatches:
        raise RuntimeError("PIT data V2 frozen input changed: " + ", ".join(mismatches))
    return {
        "lock_sha256": actual,
        "frozen_inputs_intact": True,
        "historical_pit_verified": False,
        "model_training_ready": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
