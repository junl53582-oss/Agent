from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import ChallengerSettings
from .data import sha256


CODE_PATHS = (
    "stockpilot/research_challenger/__init__.py",
    "stockpilot/research_challenger/config.py",
    "stockpilot/research_challenger/data.py",
    "stockpilot/research_challenger/split.py",
    "stockpilot/research_challenger/factors.py",
    "stockpilot/research_challenger/models.py",
    "stockpilot/research_challenger/metrics.py",
    "stockpilot/research_challenger/pipeline.py",
    "stockpilot/research_challenger/freeze.py",
    "stockpilot/research_challenger/cli.py",
    "tests/test_research_challenger_v31.py",
    ".github/workflows/prospective-integrity.yml",
)

PARENT_PATHS = (
    "artifacts/prospective_alpha_v1r4/plan.lock.json",
    "artifacts/research_v6/plan.lock.json",
    "artifacts/prediction_forward/v30r1_r2/plan.lock.json",
    "artifacts/research_v20r2/plan.lock.json",
    "artifacts/prediction_v30/cache/manifest.json",
)

AMENDMENT_ROOT = Path("artifacts/research_v31/experiments/001_runtime_fix")


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def freeze_plan(settings: ChallengerSettings | None = None) -> dict:
    settings = settings or ChallengerSettings()
    settings.ensure_dirs()
    target = settings.artifact_dir / "plan.lock.json"
    if target.exists():
        raise RuntimeError("V31 plan lock already exists")
    protocol_sidecar = _sidecar(settings.protocol_path)
    if not protocol_sidecar.exists() or protocol_sidecar.read_text().strip().lower() != sha256(
        settings.protocol_path
    ):
        raise RuntimeError("V31 protocol sidecar mismatch")
    paths = [settings.protocol_path, *map(Path, CODE_PATHS), *map(Path, PARENT_PATHS)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"V31 freeze inputs missing: {missing}")
    payload = {
        "model_id": settings.model_id,
        "role": settings.role,
        "status": settings.status,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "final_oos_opened": False,
        "pre_registered_challenger": settings.pre_registered_challenger,
        "files": {path.as_posix(): sha256(path) for path in paths},
        "parent_v1r4_lock_sha256": sha256(Path(PARENT_PATHS[0])),
        "v6_lock_sha256": sha256(Path(PARENT_PATHS[1])),
        "v30r1_forward_r2_lock_sha256": sha256(Path(PARENT_PATHS[2])),
        "v20r2_lock_sha256": sha256(Path(PARENT_PATHS[3])),
        "prospective_data_allowed": False,
        "v6_modified": False,
        "v30_modified": False,
        "v31_may_replace_v6": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _sidecar(target).write_text(sha256(target) + "\n", encoding="ascii")
    return {"lock_sha256": sha256(target), "intact": True}


def freeze_amendment(settings: ChallengerSettings | None = None) -> dict:
    settings = settings or ChallengerSettings()
    original = settings.artifact_dir / "plan.lock.json"
    target = AMENDMENT_ROOT / "plan.lock.json"
    if target.exists():
        raise RuntimeError("V31 amendment lock already exists")
    if not original.exists() or not _sidecar(original).exists():
        raise RuntimeError("V31 original failed lock is missing")
    original_sha = sha256(original)
    if _sidecar(original).read_text().strip().lower() != original_sha:
        raise RuntimeError("V31 original failed lock sidecar mismatch")
    amendment = AMENDMENT_ROOT / "protocol_amendment.json"
    failure = AMENDMENT_ROOT / "failure_receipt.json"
    for path in (amendment, failure):
        if not path.exists() or not _sidecar(path).exists():
            raise RuntimeError(f"V31 amendment evidence missing: {path}")
        if _sidecar(path).read_text().strip().lower() != sha256(path):
            raise RuntimeError(f"V31 amendment evidence sidecar mismatch: {path}")
    paths = [
        settings.protocol_path,
        amendment,
        failure,
        *map(Path, CODE_PATHS),
        *map(Path, PARENT_PATHS),
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"V31 amendment inputs missing: {missing}")
    payload = {
        "model_id": settings.model_id,
        "amendment_id": "V31-IMPLEMENTATION-FIX-001",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_failed_plan_lock_sha256": original_sha,
        "final_oos_opened_before_fix": False,
        "target_or_gate_change": False,
        "files": {path.as_posix(): sha256(path) for path in paths},
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _sidecar(target).write_text(sha256(target) + "\n", encoding="ascii")
    return {"lock_sha256": sha256(target), "intact": True, "amendment": True}


def verify_plan_lock(settings: ChallengerSettings | None = None) -> dict:
    settings = settings or ChallengerSettings()
    amendment_target = AMENDMENT_ROOT / "plan.lock.json"
    target = (
        amendment_target
        if amendment_target.exists()
        else settings.artifact_dir / "plan.lock.json"
    )
    sidecar = _sidecar(target)
    if not target.exists() or not sidecar.exists():
        return {"intact": False, "mismatches": ["plan.lock.json"], "lock_sha256": None}
    actual_lock = sha256(target)
    mismatches = []
    if sidecar.read_text().strip().lower() != actual_lock:
        mismatches.append("plan.lock.json.sha256")
    payload = json.loads(target.read_text(encoding="utf-8"))
    for name, expected in payload.get("files", {}).items():
        path = Path(name)
        if not path.exists() or sha256(path) != expected:
            mismatches.append(name)
    return {
        "intact": not mismatches,
        "mismatches": mismatches,
        "lock_sha256": actual_lock,
        "amendment": target == amendment_target,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
