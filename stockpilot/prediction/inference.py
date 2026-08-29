from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_v10.features import V10_FEATURES

from .calibration import PlattCalibrator
from .certification import load_certification
from .confidence import confidence_scores
from .config import PredictionSettings
from .drift import drift_from_profile
from .metrics import expected_calibration_error
from .models import LightGBMDirection, LightGBMReturn
from .storage import write_immutable_prediction_snapshot, write_latest_metadata
from .settlement import update_prediction_ledger


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_latest_panel(settings: PredictionSettings) -> pd.DataFrame:
    path = settings.models_dir / "latest_feature_panel.csv"
    if not path.exists():
        raise RuntimeError("V30 models are not validated; run stockpilot prediction-validate")
    data = pd.read_csv(path, dtype={"symbol": str})
    data["symbol"] = data["symbol"].astype(str).str.zfill(6)
    data["date"] = pd.to_datetime(data["date"])
    return data


def generate_latest_predictions(settings: PredictionSettings | None = None) -> dict:
    settings = settings or PredictionSettings()
    settings.ensure_dirs()
    manifest_path = settings.models_dir / "manifest.json"
    status_path = settings.certification_dir / "status.json"
    if not manifest_path.exists() or not status_path.exists():
        raise RuntimeError("V30 validation artifacts are missing; run stockpilot prediction-validate")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    certification = load_certification(status_path)
    current = _load_latest_panel(settings)
    probabilities: dict[int, np.ndarray] = {}
    raw_probabilities: dict[int, np.ndarray] = {}
    expected_returns: dict[int, np.ndarray] = {}
    for horizon in settings.horizons:
        entry = manifest["models"][str(horizon)]
        model = LightGBMDirection.load(Path(entry["direction"]))
        calibrator = PlattCalibrator.load(Path(entry["calibrator"]))
        raw = model.predict_proba(current)
        raw_probabilities[horizon] = raw
        probabilities[horizon] = calibrator.predict(raw)
        if horizon in settings.return_horizons:
            expected_returns[horizon] = LightGBMReturn.load(Path(entry["return"])).predict(current)
    profile = json.loads(Path(manifest["training_feature_profile"]).read_text(encoding="utf-8"))
    drift_report, drift_status, drift_multiplier = drift_from_profile(
        profile, current, psi_warning=settings.psi_warning, psi_severe=settings.psi_severe,
        zscore_warning=settings.zscore_warning, zscore_severe=settings.zscore_severe,
    )
    drift_report.to_csv(settings.validation_dir / "latest_drift.csv", index=False, encoding="utf-8-sig")
    yearly = pd.read_csv(settings.validation_dir / "yearly_metrics.csv")
    regime = pd.read_csv(settings.validation_dir / "regime_metrics.csv")
    sector = pd.read_csv(settings.validation_dir / "sector_metrics.csv")
    calibration = pd.read_csv(settings.validation_dir / "calibration_table.csv")
    critical_yearly = yearly[yearly["horizon"].isin((5, 20))]
    oos_skill = float(np.clip((critical_yearly["roc_auc"].mean() - 0.5) / 0.10, 0, 1))
    critical_calibration = calibration[calibration["horizon"].isin((5, 20))]
    calibration_quality = float(np.clip(1 - expected_calibration_error(critical_calibration) / 0.10, 0, 1))
    regime_consistency = float((regime[regime["horizon"].isin((5, 20))]["roc_auc"] > 0.5).mean())
    sector_quality = (
        sector[sector["horizon"].isin((5, 20))].groupby("broad_sector")["roc_auc"].mean()
        .sub(0.5).div(0.10).clip(0, 1)
    )
    sector_stability = current["broad_sector"].map(sector_quality).fillna(0)
    feature_completeness = current[V10_FEATURES].notna().mean(axis=1)
    combined_probability = pd.Series((probabilities[5] + probabilities[20]) / 2, index=current.index)
    confidence_score, confidence_level = confidence_scores(
        combined_probability, oos_skill=oos_skill, calibration_quality=calibration_quality,
        regime_consistency=regime_consistency, sector_stability=sector_stability,
        feature_completeness=feature_completeness, drift_multiplier=drift_multiplier,
        low_upper=settings.low_confidence_upper, medium_upper=settings.medium_confidence_upper,
    )
    output = current[["date", "symbol", "name", "close", "broad_sector", "regime", "ranking_component"]].copy()
    for horizon in settings.horizons:
        output[f"p_up_{horizon}d_raw"] = raw_probabilities[horizon]
        output[f"p_up_{horizon}d"] = probabilities[horizon]
        output[f"p_up_{horizon}d_rank"] = output[f"p_up_{horizon}d"].rank(pct=True, method="average")
        output[f"rank_{horizon}d"] = output[f"p_up_{horizon}d"].rank(ascending=False, method="first").astype(int)
    for horizon in settings.return_horizons:
        output[f"expected_return_{horizon}d"] = expected_returns[horizon]
    output["confidence_score"] = confidence_score
    output["confidence_level"] = confidence_level
    volatility_rank = pd.to_numeric(current["volatility_20"], errors="coerce").rank(pct=True).fillna(0.5)
    output["risk_penalty"] = volatility_rank
    output["risk_level"] = pd.cut(volatility_rank, [-np.inf, 0.33, 0.67, np.inf], labels=["LOW", "MEDIUM", "HIGH"]).astype(str)
    output["probability_component"] = combined_probability.rank(pct=True, method="average")
    combined_return = pd.Series(expected_returns[5] + expected_returns[20], index=current.index)
    output["expected_return_component"] = combined_return.rank(pct=True, method="average")
    output["candidate_score"] = (
        settings.candidate_ranking_weight * output["ranking_component"]
        + settings.candidate_probability_weight * output["probability_component"]
        + settings.candidate_return_weight * output["expected_return_component"]
        - settings.candidate_risk_penalty * output["risk_penalty"]
    )
    output["prediction_ready"] = certification.production_prediction_ready
    output["calibration_status"] = "PASSED" if certification.calibration_passed else "FAILED"
    output["drift_status"] = drift_status
    manifest_digest = _hash(manifest_path)
    output["model_version"] = f"{settings.version}:{manifest_digest[:12]}"
    output["training_cutoff"] = manifest["training_cutoff"]
    date_text = str(pd.to_datetime(output["date"]).max().date())
    snapshot_path = settings.prediction_dir / f"{date_text}.csv"
    if snapshot_path.exists():
        existing = pd.read_csv(snapshot_path, dtype={"symbol": str})
        generated_at = str(existing["generated_at_utc"].iloc[0])
    else:
        generated_at = datetime.now(timezone.utc).isoformat()
    output["generated_at_utc"] = generated_at
    output["execution_authorized"] = False
    output = output.sort_values(["rank_5d", "symbol"]).reset_index(drop=True)
    wrote, digest = write_immutable_prediction_snapshot(output, snapshot_path)
    from stockpilot.data import load_panel

    ledger = update_prediction_ledger(
        settings.prediction_dir,
        load_panel(settings.market_path),
        settings.artifact_dir / "live" / "prediction_ledger.csv",
        direction_thresholds=settings.direction_thresholds,
    )
    metadata = {
        "prediction_date": date_text, "prediction_count": len(output), "snapshot_path": str(snapshot_path),
        "sha256": digest, "already_recorded": not wrote, "model_version": output["model_version"].iloc[0],
        "production_prediction_ready": certification.production_prediction_ready,
        "future_126d_confirmed": certification.future_126d_confirmed,
        "future_confirmation_status": certification.future_confirmation_status,
        "execution_authorized": False, "drift_status": drift_status,
        "ledger_rows": int(len(ledger)), "settled_rows": int(ledger["settled"].fillna(False).sum()),
    }
    write_latest_metadata(settings.artifact_dir / "live" / "latest.json", metadata)
    return metadata


def prediction_status(settings: PredictionSettings | None = None) -> dict:
    settings = settings or PredictionSettings()
    path = settings.certification_dir / "status.json"
    if not path.exists():
        return {
            "model_version": settings.version, "production_prediction_ready": False,
            "future_126d_confirmed": False, "execution_authorized": False,
            "status": "NOT_VALIDATED",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    latest = settings.artifact_dir / "live" / "latest.json"
    if latest.exists():
        payload["latest_prediction"] = json.loads(latest.read_text(encoding="utf-8"))
    return payload


def prediction_history(symbol: str, settings: PredictionSettings | None = None) -> pd.DataFrame:
    settings = settings or PredictionSettings()
    normalized = str(symbol).zfill(6)
    pieces = []
    for path in sorted(settings.prediction_dir.glob("????-??-??.csv")):
        frame = pd.read_csv(path, dtype={"symbol": str})
        selected = frame[frame["symbol"].astype(str).str.zfill(6) == normalized]
        if not selected.empty:
            pieces.append(selected)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
