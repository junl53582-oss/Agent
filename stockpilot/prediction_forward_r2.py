from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stockpilot.data import load_panel
from stockpilot.membership import load_membership_history
from stockpilot.prediction_forward import (
    ForwardPredictionSettings,
    _generate_from_panel,
    _immutable_json,
    _sha256,
    build_latest_pit_feature_panel,
    compare_feature_panel,
    stitch_hfq_market,
)
from stockpilot.prediction_forward_r1 import attach_optional_ranking
from stockpilot.prediction.storage import (
    write_immutable_prediction_snapshot,
    write_latest_metadata,
)


@dataclass(frozen=True)
class ForwardR2Settings(ForwardPredictionSettings):
    version: str = "V30r1-forward-r2"
    artifact_dir: Path = Path("artifacts/prediction_forward/v30r1_r2")
    failed_parent_dir: Path = Path("artifacts/prediction_forward/v30r1_r1")


V30_INFERENCE_ARTIFACTS = (
    "artifacts/prediction_v30/certification/status.json",
    "artifacts/prediction_v30/models/calibrator_h1.json",
    "artifacts/prediction_v30/models/calibrator_h5.json",
    "artifacts/prediction_v30/models/calibrator_h20.json",
    "artifacts/prediction_v30/models/direction_h1.txt",
    "artifacts/prediction_v30/models/direction_h1.txt.meta.json",
    "artifacts/prediction_v30/models/direction_h5.txt",
    "artifacts/prediction_v30/models/direction_h5.txt.meta.json",
    "artifacts/prediction_v30/models/direction_h20.txt",
    "artifacts/prediction_v30/models/direction_h20.txt.meta.json",
    "artifacts/prediction_v30/models/logistic_baseline_h1.json",
    "artifacts/prediction_v30/models/logistic_baseline_h5.json",
    "artifacts/prediction_v30/models/logistic_baseline_h20.json",
    "artifacts/prediction_v30/models/manifest.json",
    "artifacts/prediction_v30/models/return_h5.txt",
    "artifacts/prediction_v30/models/return_h5.txt.meta.json",
    "artifacts/prediction_v30/models/return_h20.txt",
    "artifacts/prediction_v30/models/return_h20.txt.meta.json",
    "artifacts/prediction_v30/models/ridge_baseline_h5.json",
    "artifacts/prediction_v30/models/ridge_baseline_h20.json",
    "artifacts/prediction_v30/models/training_feature_profile.json",
    "artifacts/prediction_v30/models/latest_feature_panel.csv",
)

V30R1_INFERENCE_ARTIFACTS = (
    "artifacts/prediction_v30r1/certification/status.json",
    "artifacts/prediction_v30r1/models/calibrator_h1.json",
    "artifacts/prediction_v30r1/models/calibrator_h5.json",
    "artifacts/prediction_v30r1/models/calibrator_h20.json",
    "artifacts/prediction_v30r1/models/manifest.json",
)

INFERENCE_CODE_PATHS = (
    "stockpilot/prediction_forward.py",
    "stockpilot/prediction_forward_r1.py",
    "stockpilot/data.py",
    "stockpilot/membership.py",
    "research_v9/data.py",
    "research_v10/fundamentals.py",
    "stockpilot/prediction/calibration.py",
    "stockpilot/prediction/certification.py",
    "stockpilot/prediction/confidence.py",
    "stockpilot/prediction/config.py",
    "stockpilot/prediction/drift.py",
    "stockpilot/prediction/models.py",
    "stockpilot/prediction/schema.py",
    "stockpilot/prediction/settlement.py",
    "stockpilot/prediction/storage.py",
)


def _lock_expected(lock_path: Path) -> dict[str, str]:
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    return {str(name).replace("\\", "/"): value for name, value in payload["files"].items()}


def verify_publishable_inference_bundle(
    settings: ForwardR2Settings | None = None,
) -> dict:
    """Verify only immutable artifacts required for inference.

    Large OOS evidence remains represented by the parent locks but is deliberately
    not needed to execute a prediction from an already frozen, non-certified model.
    This check never upgrades the parent's certification result.
    """
    settings = settings or ForwardR2Settings()
    lock_specs = (
        (settings.parent_v30_root / "validation.lock.json", V30_INFERENCE_ARTIFACTS),
        (settings.parent_root / "validation.lock.json", V30R1_INFERENCE_ARTIFACTS),
    )
    mismatches: list[str] = []
    checked: dict[str, str] = {}
    for lock_path, paths in lock_specs:
        expected = _lock_expected(lock_path)
        for name in paths:
            path = Path(name)
            actual = _sha256(path) if path.exists() else None
            checked[name] = actual or "MISSING"
            if expected.get(name) != actual:
                mismatches.append(name)
    v30r1_manifest = json.loads(
        (settings.parent_root / "models" / "manifest.json").read_text(encoding="utf-8")
    )
    v30_manifest = settings.parent_v30_root / "models" / "manifest.json"
    if v30r1_manifest.get("parent_model_manifest_sha256") != _sha256(v30_manifest):
        mismatches.append("parent_model_manifest_sha256")
    return {
        "intact": not mismatches,
        "mismatches": sorted(set(mismatches)),
        "checked_files": len(checked),
        "scope": "publishable_inference_bundle_only",
        "full_historical_validation_recertified": False,
    }


def _r2_frozen_paths(settings: ForwardR2Settings) -> list[Path]:
    paths = [
        settings.artifact_dir / "protocol.json",
        Path("stockpilot/prediction_forward_r2.py"),
        Path("tests/test_prediction_forward_r2.py"),
        settings.failed_parent_dir / "plan.lock.json",
        settings.parent_root / "validation.lock.json",
        settings.parent_v30_root / "validation.lock.json",
    ]
    paths.extend(Path(name) for name in INFERENCE_CODE_PATHS)
    paths.extend(Path(name) for name in V30_INFERENCE_ARTIFACTS)
    paths.extend(Path(name) for name in V30R1_INFERENCE_ARTIFACTS)
    return list(dict.fromkeys(paths))


def create_r2_plan_lock(settings: ForwardR2Settings | None = None) -> dict:
    settings = settings or ForwardR2Settings()
    target = settings.artifact_dir / "plan.lock.json"
    if target.exists():
        raise RuntimeError(f"forward-r2 plan lock already exists: {target}")
    paths = _r2_frozen_paths(settings)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError("cannot freeze forward-r2 plan: " + ", ".join(missing))
    bundle = verify_publishable_inference_bundle(settings)
    if not bundle["intact"]:
        raise RuntimeError(f"publishable inference bundle is not intact: {bundle}")
    payload = {
        "version": settings.version,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "inference_only_publishable_parent_bundle",
        "parent_r1_preserved": True,
        "full_historical_validation_recertified": False,
        "production_prediction_ready_may_not_be_promoted": True,
        "execution_authorized": False,
        "files": {path.as_posix(): _sha256(path) for path in paths},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload | {"lock_sha256": _sha256(target)}


def verify_r2_plan_lock(settings: ForwardR2Settings | None = None) -> dict:
    settings = settings or ForwardR2Settings()
    target = settings.artifact_dir / "plan.lock.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    mismatches = [
        name
        for name, expected in payload["files"].items()
        if not Path(name).exists() or _sha256(Path(name)) != expected
    ]
    return {
        "intact": not mismatches,
        "mismatches": mismatches,
        "lock_sha256": _sha256(target),
    }


def run_forward_r2(
    incremental_market_path: str | Path,
    as_of: str | pd.Timestamp,
    *,
    ranking_path: str | Path,
    settings: ForwardR2Settings | None = None,
) -> dict:
    settings = settings or ForwardR2Settings()
    as_of_date = pd.Timestamp(as_of).normalize()
    locks = {
        "publishable_inference_bundle": verify_publishable_inference_bundle(settings),
        "forward_r2": verify_r2_plan_lock(settings),
    }
    if not all(lock["intact"] for lock in locks.values()):
        raise RuntimeError(f"a forward-r2 input is not intact: {locks}")
    frozen = load_panel(settings.frozen_market_path)
    cutoff = pd.to_datetime(frozen["date"]).max()
    incremental_path = Path(incremental_market_path)
    ranking_path = Path(ranking_path)
    combined, market_audit = stitch_hfq_market(
        frozen,
        load_panel(incremental_path),
        load_membership_history(settings.membership_path),
        cutoff=cutoff,
        as_of=as_of_date,
        settings=settings,
    )
    parity_panel, parity_pit = build_latest_pit_feature_panel(
        combined[combined["date"] <= cutoff], cutoff, settings=settings
    )
    parity = compare_feature_panel(
        parity_panel,
        settings.parent_v30_root / "models" / "latest_feature_panel.csv",
    )
    if not parity["passed"]:
        raise RuntimeError(f"frozen feature parity failed: {parity}")
    current, pit_audit = build_latest_pit_feature_panel(combined, as_of_date, settings=settings)
    current, ranking_audit = attach_optional_ranking(current, ranking_path, as_of_date)
    feature_path = settings.feature_dir / f"{as_of_date.date()}.csv"
    write_immutable_prediction_snapshot(current, feature_path)
    audit_path = settings.audit_dir / f"{as_of_date.date()}.json"
    generated_at = (
        json.loads(audit_path.read_text(encoding="utf-8"))["generated_at_utc"]
        if audit_path.exists()
        else datetime.now(timezone.utc).isoformat()
    )
    audit = {
        "version": settings.version,
        "generated_at_utc": generated_at,
        "locks": locks,
        "input_hashes": {
            str(incremental_path): _sha256(incremental_path),
            str(ranking_path): _sha256(ranking_path),
            str(settings.frozen_market_path): _sha256(settings.frozen_market_path),
            str(settings.membership_path): _sha256(settings.membership_path),
            str(settings.fundamental_path): _sha256(settings.fundamental_path),
            str(settings.industry_path): _sha256(settings.industry_path),
        },
        "market_stitch": market_audit,
        "frozen_feature_parity": parity,
        "frozen_feature_pit": parity_pit,
        "latest_feature_pit": pit_audit,
        "auxiliary_ranking": ranking_audit,
        "full_historical_validation_recertified": False,
        "execution_authorized": False,
    }
    _immutable_json(audit_path, audit)
    result = _generate_from_panel(current, combined, as_of_date, settings)
    result.update(
        {
            "market_audit_passed": True,
            "feature_parity_passed": True,
            "pit_audit_passed": True,
            "publishable_inference_bundle_intact": True,
            "full_historical_validation_recertified": False,
            "auxiliary_ranking_coverage": ranking_audit["coverage"],
            "auxiliary_ranking_neutral_fallback_rows": ranking_audit["missing_rows"],
        }
    )
    write_latest_metadata(settings.artifact_dir / "latest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen V30r1 forward-r2 inference")
    parser.add_argument("--market", required=True)
    parser.add_argument("--as-of", required=True, dest="as_of")
    parser.add_argument("--ranking", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    result = run_forward_r2(args.market, args.as_of, ranking_path=args.ranking)
    frame = pd.read_csv(result["snapshot_path"], dtype={"symbol": str}).head(args.limit)
    columns = [
        "rank_5d",
        "symbol",
        "name",
        "p_up_1d",
        "p_up_5d",
        "p_up_20d",
        "expected_return_5d",
        "expected_return_20d",
        "confidence_level",
        "prediction_ready",
        "execution_authorized",
    ]
    print(frame[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
