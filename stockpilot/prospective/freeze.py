from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .immutable import sha256_file, write_new_json


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "artifacts/prospective_alpha_v1"
LOCK = DIRECTORY / "plan.lock.json"


def frozen_paths() -> list[Path]:
    paths = sorted((ROOT / "stockpilot/prospective").glob("*.py"))
    paths += [
        ROOT / "tests/test_prospective_alpha_v1.py",
        ROOT / "tests/conftest.py",
        DIRECTORY / "protocol.json",
        DIRECTORY / "audit.json",
        DIRECTORY / "test_receipt.json",
        ROOT / "artifacts/pit_data_v2/data.lock.json",
        ROOT / "artifacts/pit_data_v2/data.lock.sha256",
        ROOT / "artifacts/pit_data_v1r3/data.lock.json",
        ROOT / "artifacts/pit_data_v1r3/data.lock.sha256",
        ROOT / "data/pit_observations_v1r3/20260830T051947935413Z/manifest.json",
        ROOT / "data/pit_observations_v1r3/20260830T051947935413Z/expectations.csv",
        ROOT / "artifacts/announcement_body_v5r2/observations/20260830T050907681987Z.json",
        ROOT / "artifacts/prediction_forward/v30r1_r2/plan.lock.json",
        ROOT / "artifacts/research_v6/plan.lock.json",
        ROOT / "artifacts/research_v18/plan.lock.json",
        ROOT / "artifacts/research_v18/plan.lock.sha256",
    ]
    return paths


def create_lock() -> dict:
    if LOCK.exists():
        raise RuntimeError("prospective alpha V1 is already frozen")
    missing = [path for path in frozen_paths() if not path.exists()]
    if missing:
        raise RuntimeError(f"cannot freeze missing paths: {missing}")
    payload = {
        "version": "prospective-alpha-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "observation_feature_label_validation_infrastructure_no_model_training",
        "files": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in frozen_paths()},
        "minimum_real_trading_observations": 20,
        "historical_backfill_allowed": False,
        "same_date_retry_allowed": False,
        "silent_fallback_allowed": False,
        "model_training_entrypoint_present": False,
        "model_training_ready": False,
        "factor_validation_ready": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    write_new_json(LOCK, payload)
    return verify_lock()


def verify_lock() -> dict:
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    mismatches = [
        name for name, digest in payload["files"].items()
        if not (ROOT / name).exists() or sha256_file(ROOT / name) != digest
    ]
    if mismatches:
        raise RuntimeError("prospective alpha frozen input changed: " + ", ".join(mismatches))
    return {
        "lock_sha256": sha256_file(LOCK),
        "frozen_inputs_intact": True,
        "model_training_ready": False,
        "factor_validation_ready": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
