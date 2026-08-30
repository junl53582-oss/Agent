from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pit_data_v1.core import sha256_file
from pit_data_v1r2.freeze import verify_lock as verify_parent

from .core import AdmissionSettings


ROOT = Path(__file__).resolve().parents[1]


def _raw_evidence(settings: AdmissionSettings) -> dict[str, str]:
    directory = ROOT / settings.parent_data_root / settings.parent_observation_id / "raw" / "expectations"
    paths = sorted(directory.glob("page_*.json"))
    if not paths:
        raise RuntimeError("parent raw expectation evidence is missing")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def create_lock(settings: AdmissionSettings | None = None) -> dict:
    settings = settings or AdmissionSettings()
    directory = ROOT / settings.artifact_root
    target = directory / "data.lock.json"
    if target.exists():
        raise RuntimeError("PIT data V1r3 is already frozen")
    dynamic_root = ROOT / settings.data_root
    if dynamic_root.exists() and any(dynamic_root.rglob("*")):
        raise RuntimeError("V1r3 dynamic data root must be empty before freeze")
    parent = verify_parent()
    parent_observation = ROOT / settings.parent_artifact_root / "observations" / f"{settings.parent_observation_id}.json"
    if not parent_observation.exists():
        raise RuntimeError("frozen parent observation report is missing")
    receipt = json.loads((directory / "test_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("passed") is not True or receipt.get("admission_run_before_freeze") is not False:
        raise RuntimeError("duplicate-admission tests must pass before repair")
    files = [
        *sorted((ROOT / "pit_data_v1r3").glob("*.py")),
        ROOT / "tests/test_pit_data_v1r3.py",
        directory / "protocol.json",
        directory / "test_receipt.json",
        ROOT / settings.parent_artifact_root / "data.lock.json",
        ROOT / settings.parent_artifact_root / "data.lock.sha256",
        parent_observation,
        ROOT / settings.membership_path,
        ROOT / settings.industry_path,
    ]
    payload = {
        "version": settings.version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "offline_admission_of_frozen_prospective_expectation_evidence",
        "parent_lock_sha256": parent["lock_sha256"],
        "parent_observation_id": settings.parent_observation_id,
        "parent_failure_preserved": True,
        "allowed_dynamic_data_root": settings.data_root.as_posix(),
        "allowed_dynamic_artifact_root": (settings.artifact_root / "observations").as_posix(),
        "files": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in files},
        "raw_evidence_files": _raw_evidence(settings),
        "network_access_allowed": False,
        "historical_pit_verified": False,
        "model_training_ready": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    directory.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "data.lock.sha256").write_text(sha256_file(target) + "\n", encoding="utf-8")
    return verify_lock(settings)


def verify_lock(settings: AdmissionSettings | None = None) -> dict:
    settings = settings or AdmissionSettings()
    parent = verify_parent()
    directory = ROOT / settings.artifact_root
    target = directory / "data.lock.json"
    actual = sha256_file(target)
    if actual != (directory / "data.lock.sha256").read_text(encoding="utf-8").strip():
        raise RuntimeError("PIT data V1r3 lock file changed")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload["parent_lock_sha256"] != parent["lock_sha256"]:
        raise RuntimeError("PIT data V1r2 parent lock changed")
    mismatches = [
        name for name, digest in {**payload["files"], **payload["raw_evidence_files"]}.items()
        if not (ROOT / name).exists() or sha256_file(ROOT / name) != digest
    ]
    if mismatches:
        raise RuntimeError("PIT data V1r3 frozen input changed: " + ", ".join(mismatches))
    return {
        "lock_sha256": actual,
        "frozen_inputs_intact": True,
        "raw_evidence_pages": len(payload["raw_evidence_files"]),
        "historical_pit_verified": False,
        "model_training_ready": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
