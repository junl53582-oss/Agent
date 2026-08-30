from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from stockpilot.prospective.freeze import verify_lock as verify_parent
from stockpilot.prospective.immutable import sha256_file, write_new_json


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "artifacts/prospective_alpha_v1r1"
LOCK = DIRECTORY / "plan.lock.json"


def frozen_paths() -> list[Path]:
    return [
        *sorted((ROOT / "stockpilot/prospective_r1").glob("*.py")),
        ROOT / "tests/test_prospective_alpha_v1r1.py",
        DIRECTORY / "protocol.json",
        DIRECTORY / "test_receipt.json",
        ROOT / "artifacts/prospective_alpha_v1/plan.lock.json",
    ]


def create_lock() -> dict:
    if LOCK.exists():
        raise RuntimeError("prospective alpha V1r1 is already frozen")
    parent = verify_parent()
    payload = {
        "version": "prospective-alpha-v1r1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_lock_sha256": parent["lock_sha256"],
        "scope": "atomic-date-reservation-and-nonempty-required-value-validation-only",
        "files": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in frozen_paths()},
        "model_training_entrypoint_present": False,
        "model_training_ready": False,
        "factor_validation_ready": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    write_new_json(LOCK, payload)
    return verify_lock()


def verify_lock() -> dict:
    parent = verify_parent()
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    if payload["parent_lock_sha256"] != parent["lock_sha256"]:
        raise RuntimeError("prospective alpha parent lock changed")
    mismatches = [
        name for name, digest in payload["files"].items()
        if not (ROOT / name).exists() or sha256_file(ROOT / name) != digest
    ]
    if mismatches:
        raise RuntimeError("prospective alpha V1r1 frozen input changed: " + ", ".join(mismatches))
    return {
        "lock_sha256": sha256_file(LOCK),
        "parent_lock_sha256": parent["lock_sha256"],
        "frozen_inputs_intact": True,
        "model_training_ready": False,
        "factor_validation_ready": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
