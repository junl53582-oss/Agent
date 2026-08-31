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

RUNTIME_AMENDMENT_ROOT = Path("artifacts/research_v31/experiments/001_runtime_fix")
CI_AMENDMENT_ROOT = Path("artifacts/research_v31/experiments/002_ci_verifier_fix")
AMENDMENT_ROOTS = (RUNTIME_AMENDMENT_ROOT, CI_AMENDMENT_ROOT)


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
    target = RUNTIME_AMENDMENT_ROOT / "plan.lock.json"
    if target.exists():
        raise RuntimeError("V31 amendment lock already exists")
    if not original.exists() or not _sidecar(original).exists():
        raise RuntimeError("V31 original failed lock is missing")
    original_sha = sha256(original)
    if _sidecar(original).read_text().strip().lower() != original_sha:
        raise RuntimeError("V31 original failed lock sidecar mismatch")
    amendment = RUNTIME_AMENDMENT_ROOT / "protocol_amendment.json"
    failure = RUNTIME_AMENDMENT_ROOT / "failure_receipt.json"
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


def freeze_ci_amendment(settings: ChallengerSettings | None = None) -> dict:
    """Freeze an operational-only correction to the clean-checkout verifier."""
    settings = settings or ChallengerSettings()
    parent = RUNTIME_AMENDMENT_ROOT / "plan.lock.json"
    target = CI_AMENDMENT_ROOT / "plan.lock.json"
    if target.exists():
        raise RuntimeError("V31 CI amendment lock already exists")
    if not parent.exists() or not _sidecar(parent).exists():
        raise RuntimeError("V31 runtime amendment lock is missing")
    parent_sha = sha256(parent)
    if _sidecar(parent).read_text().strip().lower() != parent_sha:
        raise RuntimeError("V31 runtime amendment lock sidecar mismatch")
    amendment = CI_AMENDMENT_ROOT / "protocol_amendment.json"
    failure = CI_AMENDMENT_ROOT / "failure_receipt.json"
    for path in (amendment, failure):
        if not path.exists() or not _sidecar(path).exists():
            raise RuntimeError(f"V31 CI amendment evidence missing: {path}")
        if _sidecar(path).read_text().strip().lower() != sha256(path):
            raise RuntimeError(f"V31 CI amendment evidence sidecar mismatch: {path}")
    paths = [
        settings.protocol_path,
        amendment,
        failure,
        settings.artifact_dir / "artifact_manifest.json",
        *map(Path, CODE_PATHS),
        *map(Path, PARENT_PATHS),
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"V31 CI amendment inputs missing: {missing}")
    payload = {
        "model_id": settings.model_id,
        "amendment_id": "V31-CI-VERIFIER-FIX-002",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_effective_plan_lock_sha256": parent_sha,
        "historical_oos_already_observed": True,
        "research_outputs_modified": False,
        "research_rerun_authorized": False,
        "target_or_gate_change": False,
        "files": {path.as_posix(): sha256(path) for path in paths},
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _sidecar(target).write_text(sha256(target) + "\n", encoding="ascii")
    return {
        "lock_sha256": sha256(target),
        "intact": True,
        "amendment": True,
        "amendment_id": payload["amendment_id"],
    }


def _effective_lock_target(settings: ChallengerSettings) -> Path:
    for root in reversed(AMENDMENT_ROOTS):
        candidate = root / "plan.lock.json"
        if candidate.exists():
            return candidate
    return settings.artifact_dir / "plan.lock.json"


def verify_plan_lock(settings: ChallengerSettings | None = None) -> dict:
    settings = settings or ChallengerSettings()
    target = _effective_lock_target(settings)
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
        "amendment": target.parent in AMENDMENT_ROOTS,
        "amendment_id": payload.get("amendment_id"),
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
