from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_v10.features import V10_FEATURES
from stockpilot.data import load_panel
from stockpilot.prediction.certification import load_certification
from stockpilot.prediction.confidence import confidence_scores
from stockpilot.prediction.drift import drift_from_profile
from stockpilot.prediction.freeze import digest, verify_validation_lock
from stockpilot.prediction.metrics import expected_calibration_error
from stockpilot.prediction.models import LightGBMDirection, LightGBMReturn, LogisticRidge, RidgeReturn
from stockpilot.prediction.settlement import update_prediction_ledger
from stockpilot.prediction.storage import write_immutable_prediction_snapshot, write_latest_metadata

from .calibration import MonotonicPlattCalibrator
from .config import V30R1Settings
from .freeze import verify_plan_lock


def generate_latest_v30r1_predictions(settings: V30R1Settings | None = None) -> dict:
    settings = settings or V30R1Settings()
    manifest_path = settings.models_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("run V30r1 validation first")
    if not verify_validation_lock()["intact"] or not verify_plan_lock(settings)["intact"]:
        raise RuntimeError("V30 or V30r1 frozen inputs are not intact")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parent_manifest = json.loads(Path(manifest["parent_model_manifest"]).read_text(encoding="utf-8"))
    certification = load_certification(settings.certification_dir / "status.json")
    current = pd.read_csv(
        settings.parent_dir / "models" / "latest_feature_panel.csv", dtype={"symbol": str},
    )
    current["date"] = pd.to_datetime(current["date"])
    chosen_raw, calibrated, expected = {}, {}, {}
    for horizon in settings.horizons:
        selection = manifest["models"][str(horizon)]
        parent = parent_manifest["models"][str(horizon)]
        if selection["direction_source"] == "lightgbm":
            model = LightGBMDirection.load(Path(parent["direction"]))
        else:
            model = LogisticRidge.load(Path(parent["logistic_baseline"]))
        raw = model.predict_proba(current)
        chosen_raw[horizon] = raw
        calibrator = MonotonicPlattCalibrator.load(Path(selection["calibrator"]))
        calibrated[horizon] = calibrator.predict(raw)
        if horizon in settings.return_horizons:
            if selection["return_source"] == "lightgbm":
                return_model = LightGBMReturn.load(Path(parent["return"]))
            else:
                return_model = RidgeReturn.load(Path(parent["ridge_baseline"]))
            expected[horizon] = return_model.predict(current)
    profile = json.loads((settings.parent_dir / "models" / "training_feature_profile.json").read_text(encoding="utf-8"))
    drift, drift_status, drift_multiplier = drift_from_profile(
        profile, current, psi_warning=settings.psi_warning, psi_severe=settings.psi_severe,
        zscore_warning=settings.zscore_warning, zscore_severe=settings.zscore_severe,
    )
    drift.to_csv(settings.validation_dir / "latest_drift.csv", index=False, encoding="utf-8-sig")
    yearly = pd.read_csv(settings.validation_dir / "yearly_metrics.csv")
    regime = pd.read_csv(settings.validation_dir / "regime_metrics.csv")
    sector = pd.read_csv(settings.validation_dir / "sector_metrics.csv")
    calibration = pd.read_csv(settings.validation_dir / "calibration_table.csv")
    critical = yearly[yearly["horizon"].isin((5, 20))]
    oos_skill = float(np.clip((critical["roc_auc"].mean() - 0.5) / 0.10, 0, 1))
    calibration_quality = float(np.clip(
        1 - expected_calibration_error(calibration[calibration["horizon"].isin((5, 20))]) / 0.10, 0, 1
    ))
    regime_consistency = float((regime[regime["horizon"].isin((5, 20))]["roc_auc"] > 0.5).mean())
    sector_quality = sector[sector["horizon"].isin((5, 20))].groupby("broad_sector")["roc_auc"].mean().sub(0.5).div(0.1).clip(0, 1)
    combined_probability = pd.Series((calibrated[5] + calibrated[20]) / 2, index=current.index)
    confidence_score, confidence_level = confidence_scores(
        combined_probability, oos_skill=oos_skill, calibration_quality=calibration_quality,
        regime_consistency=regime_consistency,
        sector_stability=current["broad_sector"].map(sector_quality).fillna(0),
        feature_completeness=current[V10_FEATURES].notna().mean(axis=1),
        drift_multiplier=drift_multiplier,
        low_upper=settings.low_confidence_upper, medium_upper=settings.medium_confidence_upper,
    )
    output = current[["date", "symbol", "name", "close", "broad_sector", "regime", "ranking_component"]].copy()
    for horizon in settings.horizons:
        output[f"p_up_{horizon}d_raw"] = chosen_raw[horizon]
        output[f"p_up_{horizon}d"] = calibrated[horizon]
        output[f"p_up_{horizon}d_rank"] = output[f"p_up_{horizon}d"].rank(pct=True, method="average")
        output[f"rank_{horizon}d"] = output[f"p_up_{horizon}d"].rank(ascending=False, method="first").astype(int)
    for horizon in settings.return_horizons:
        output[f"expected_return_{horizon}d"] = expected[horizon]
    output["confidence_score"], output["confidence_level"] = confidence_score, confidence_level
    volatility_rank = current["volatility_20"].rank(pct=True).fillna(0.5)
    output["risk_penalty"] = volatility_rank
    output["risk_level"] = pd.cut(volatility_rank, [-np.inf, 0.33, 0.67, np.inf], labels=["LOW", "MEDIUM", "HIGH"]).astype(str)
    output["probability_component"] = combined_probability.rank(pct=True)
    output["expected_return_component"] = pd.Series(expected[5] + expected[20], index=current.index).rank(pct=True)
    output["candidate_score"] = (
        settings.candidate_ranking_weight * output["ranking_component"]
        + settings.candidate_probability_weight * output["probability_component"]
        + settings.candidate_return_weight * output["expected_return_component"]
        - settings.candidate_risk_penalty * output["risk_penalty"]
    )
    output["prediction_ready"] = certification.production_prediction_ready
    output["calibration_status"] = "PASSED" if certification.calibration_passed else "FAILED"
    output["drift_status"] = drift_status
    output["model_version"] = f"{settings.version}:{digest(manifest_path)[:12]}"
    output["training_cutoff"] = manifest["training_cutoff"]
    date_text = str(output["date"].max().date())
    snapshot = settings.prediction_dir / f"{date_text}.csv"
    if snapshot.exists():
        generated_at = pd.read_csv(snapshot, usecols=["generated_at_utc"])["generated_at_utc"].iloc[0]
    else:
        generated_at = datetime.now(timezone.utc).isoformat()
    output["generated_at_utc"] = generated_at
    output["execution_authorized"] = False
    output = output.sort_values(["rank_5d", "symbol"]).reset_index(drop=True)
    wrote, snapshot_hash = write_immutable_prediction_snapshot(output, snapshot)
    ledger = update_prediction_ledger(
        settings.prediction_dir, load_panel(settings.market_path),
        settings.artifact_dir / "live" / "prediction_ledger.csv",
        direction_thresholds=settings.direction_thresholds,
    )
    result = {
        "prediction_date": date_text, "prediction_count": len(output),
        "snapshot_path": str(snapshot), "sha256": snapshot_hash, "already_recorded": not wrote,
        "model_version": output["model_version"].iloc[0],
        "production_prediction_ready": certification.production_prediction_ready,
        "future_126d_confirmed": False, "future_confirmation_status": "COLLECTING",
        "execution_authorized": False, "drift_status": drift_status,
        "ledger_rows": len(ledger), "settled_rows": int(ledger["settled"].fillna(False).sum()),
    }
    write_latest_metadata(settings.artifact_dir / "live" / "latest.json", result)
    return result


def v30r1_status(settings: V30R1Settings | None = None) -> dict:
    settings = settings or V30R1Settings()
    path = settings.certification_dir / "status.json"
    if not path.exists():
        return {"version": settings.version, "status": "NOT_VALIDATED", "production_prediction_ready": False, "execution_authorized": False}
    result = json.loads(path.read_text(encoding="utf-8"))
    latest = settings.artifact_dir / "live" / "latest.json"
    if latest.exists():
        result["latest_prediction"] = json.loads(latest.read_text(encoding="utf-8"))
    return result
