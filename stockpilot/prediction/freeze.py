from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import PredictionSettings


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def _locked_files(settings: PredictionSettings) -> list[Path]:
    code = sorted(Path("stockpilot/prediction").glob("*.py"))
    artifacts = [
        settings.artifact_dir / "protocol.json",
        settings.artifact_dir / "audit.json",
        settings.validation_dir / "report.json",
        settings.validation_dir / "data_audit.json",
        settings.validation_dir / "yearly_metrics.csv",
        settings.validation_dir / "regime_metrics.csv",
        settings.validation_dir / "sector_metrics.csv",
        settings.validation_dir / "calibration_table.csv",
        settings.validation_dir / "baseline_comparison.csv",
        settings.validation_dir / "fold_audit.csv",
        settings.validation_dir / "oos_predictions.csv",
        settings.models_dir / "manifest.json",
        settings.models_dir / "training_feature_profile.json",
        settings.models_dir / "latest_feature_panel.csv",
        settings.certification_dir / "status.json",
    ]
    artifacts.extend(sorted(settings.models_dir.glob("direction_h*.txt*")))
    artifacts.extend(sorted(settings.models_dir.glob("return_h*.txt*")))
    artifacts.extend(sorted(settings.models_dir.glob("calibrator_h*.json")))
    artifacts.extend(sorted(settings.models_dir.glob("*baseline_h*.json")))
    artifacts.extend(sorted(settings.prediction_dir.glob("????-??-??.csv*")))
    return sorted(set(code + artifacts), key=lambda path: str(path).replace("\\", "/"))


def create_validation_lock(settings: PredictionSettings | None = None) -> dict:
    settings = settings or PredictionSettings()
    target = settings.artifact_dir / "validation.lock.json"
    if target.exists():
        raise RuntimeError(f"V30 validation lock already exists: {target}")
    files = _locked_files(settings)
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise RuntimeError("cannot freeze missing V30 files: " + ", ".join(missing))
    payload = {
        "version": settings.version,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "immutable_result": True,
        "production_prediction_ready": False,
        "execution_authorized": False,
        "files": {str(path).replace("\\", "/"): digest(path) for path in files},
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["lock_sha256"] = digest(target)
    return payload


def verify_validation_lock(settings: PredictionSettings | None = None) -> dict:
    settings = settings or PredictionSettings()
    target = settings.artifact_dir / "validation.lock.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    mismatches = []
    for name, expected in payload["files"].items():
        path = Path(name)
        if not path.exists() or digest(path) != expected:
            mismatches.append(name)
    return {"intact": not mismatches, "mismatches": mismatches, "lock_sha256": digest(target)}

