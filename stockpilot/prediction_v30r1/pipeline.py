from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stockpilot.prediction.certification import PredictionCertificationResult
from stockpilot.prediction.config import PredictionSettings
from stockpilot.prediction.data import load_prediction_dataset
from stockpilot.prediction.freeze import digest, verify_validation_lock
from stockpilot.prediction.metrics import binary_metrics, rank_ic_by_date
from stockpilot.prediction.pipeline import (
    _Progress,
    _aggregate_reports,
    _certify,
    _fit_return_fold,
    _fold_direction_predictions,
    _ids,
)

from .calibration import MonotonicPlattCalibrator
from .config import V30R1Settings
from .freeze import verify_plan_lock
from .selection import _return_metrics, select_direction_champion, select_return_champion


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _support_2018(settings: V30R1Settings, progress: _Progress) -> pd.DataFrame:
    path = settings.validation_dir / "support_2018.csv"
    if path.exists():
        data = pd.read_csv(path, dtype={"symbol": str}, parse_dates=["date"])
        return data
    progress.write("support", "building the missing 2018 OOS support fold from the real PIT cache")
    data = load_prediction_dataset(PredictionSettings())
    data = data[data["eligible"].fillna(False)].reset_index(drop=True)
    pieces = []
    for horizon in settings.horizons:
        direction, _ = _fold_direction_predictions(data, horizon, 2018, settings, progress)
        if horizon in settings.return_horizons:
            returns = _fit_return_fold(data, horizon, 2018, settings, progress)
            direction = direction.merge(
                returns, on=["date", "symbol"], how="left", validate="one_to_one"
            )
        else:
            direction["expected_return"] = np.nan
            direction["ridge_expected_return"] = np.nan
        pieces.append(direction)
    support = pd.concat(pieces, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    support.to_csv(path, index=False, encoding="utf-8-sig")
    return support


def _rolling_history(all_oos: pd.DataFrame, year: int, horizon: int, years: int) -> pd.DataFrame:
    earliest = year - years
    return all_oos[
        (all_oos["horizon"] == horizon)
        & (all_oos["test_year"] < year)
        & (all_oos["test_year"] >= earliest)
    ].copy()


def _return_comparison(oos: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in (5, 20):
        part = oos[oos["horizon"] == horizon]
        for name, column in (
            ("selected_return_head", "selected_expected_return"),
            ("lightgbm_return", "expected_return"),
            ("ridge_return", "ridge_expected_return"),
        ):
            metrics = _return_metrics(part, column)
            rows.append({"horizon": horizon, "model": name, **metrics})
    return pd.DataFrame(rows)


def run_v30r1_validation(settings: V30R1Settings | None = None) -> dict:
    settings = settings or V30R1Settings()
    settings.ensure_dirs()
    report_path = settings.validation_dir / "report.json"
    progress = _Progress(settings)
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    parent_lock = verify_validation_lock(PredictionSettings())
    plan_lock = verify_plan_lock(settings)
    if not parent_lock["intact"] or not plan_lock["intact"]:
        raise RuntimeError("V30 parent or V30r1 plan lock is not intact")
    parent_oos_path = settings.parent_dir / "validation" / "oos_predictions.csv"
    parent_oos = pd.read_csv(parent_oos_path, dtype={"symbol": str}, parse_dates=["date"])
    support = _support_2018(settings, progress)
    all_oos = pd.concat([support, parent_oos], ignore_index=True, sort=False)
    pieces, selections = [], []
    for horizon in settings.horizons:
        for year in settings.oos_years:
            history = _rolling_history(all_oos, year, horizon, settings.calibration_years)
            current = all_oos[(all_oos["horizon"] == horizon) & (all_oos["test_year"] == year)].copy()
            direction_source, direction_evidence = select_direction_champion(history)
            direction_column = "raw_probability" if direction_source == "lightgbm" else "logistic_probability"
            calibrator = MonotonicPlattCalibrator().fit(
                history[direction_column].to_numpy(), history["actual"].to_numpy(),
                calibration_ids=_ids(history), model_training_ids=set(),
            )
            current["selected_raw_probability"] = current[direction_column]
            current["probability"] = calibrator.predict(current[direction_column].to_numpy())
            current["direction_source"] = direction_source
            selection = {
                "test_year": year, "horizon": horizon,
                "history_years": sorted(history["test_year"].astype(int).unique().tolist()),
                "direction_source": direction_source,
                "calibration_slope": calibrator.slope,
                "calibration_intercept": calibrator.intercept,
                "calibration_fallback_to_prevalence": calibrator.fallback_to_prevalence,
                "direction_evidence": direction_evidence,
            }
            if horizon in settings.return_horizons:
                return_source, return_evidence = select_return_champion(history)
                return_column = "expected_return" if return_source == "lightgbm" else "ridge_expected_return"
                current["selected_expected_return"] = current[return_column]
                selection.update({"return_source": return_source, "return_evidence": return_evidence})
            else:
                current["selected_expected_return"] = np.nan
            selections.append(selection)
            pieces.append(current)
            progress.write("replay", f"H{horizon} {year}: {direction_source}, slope={calibrator.slope:.4f}")
    oos = pd.concat(pieces, ignore_index=True)
    # Reuse the report builder's expected-return field while keeping both original heads.
    oos["candidate_expected_return"] = oos["selected_expected_return"]
    oos["expected_return_for_report"] = oos["selected_expected_return"]
    original_lgb_return = oos["expected_return"].copy()
    oos["expected_return"] = oos["selected_expected_return"]
    yearly, regime, sector, calibration, baselines = _aggregate_reports(oos, settings)
    oos["expected_return"] = original_lgb_return
    return_comparison = _return_comparison(oos)
    for frame, name in (
        (oos, "oos_predictions.csv"), (yearly, "yearly_metrics.csv"),
        (regime, "regime_metrics.csv"), (sector, "sector_metrics.csv"),
        (calibration, "calibration_table.csv"), (baselines, "baseline_comparison.csv"),
        (return_comparison, "return_baseline_comparison.csv"),
        (pd.json_normalize(selections), "selection_audit.csv"),
    ):
        frame.to_csv(settings.validation_dir / name, index=False, encoding="utf-8-sig")
    parent_report = json.loads((settings.parent_dir / "validation" / "report.json").read_text(encoding="utf-8"))
    certification, evidence = _certify(
        parent_report["data"], oos.assign(expected_return=oos["selected_expected_return"]),
        yearly, regime, sector, calibration, baselines, parent_report["folds"], settings,
    )
    monotonic = all(float(item["calibration_slope"]) >= 0 for item in selections)
    retained_return_heads_valid = all(
        item.get("return_source") != "lightgbm"
        or bool(item["return_evidence"]["lightgbm_retained"])
        for item in selections
    )
    checks = certification.to_dict()
    checks["leakage_test_passed"] = bool(checks["leakage_test_passed"] and parent_lock["intact"] and plan_lock["intact"])
    checks["calibration_passed"] = bool(checks["calibration_passed"] and monotonic)
    checks["baseline_beaten"] = bool(checks["baseline_beaten"] and retained_return_heads_valid)
    revised = PredictionCertificationResult.evaluate(
        future_126d_confirmed=False,
        **{name: checks[name] for name in (
            "data_verified", "pit_verified", "label_maturity_verified", "leakage_test_passed",
            "purged_walk_forward_passed", "calibration_passed", "baseline_beaten",
            "stability_passed", "regime_passed", "probability_quality_passed",
            "cost_aware_stress_passed",
        )},
    )
    revised.save(settings.certification_dir / "status.json")
    latest_history = all_oos[all_oos["test_year"].between(2023, 2025)]
    latest_models = {}
    for horizon in settings.horizons:
        history = latest_history[latest_history["horizon"] == horizon]
        direction_source, direction_evidence = select_direction_champion(history)
        column = "raw_probability" if direction_source == "lightgbm" else "logistic_probability"
        calibrator = MonotonicPlattCalibrator().fit(
            history[column].to_numpy(), history["actual"].to_numpy(),
            calibration_ids=_ids(history), model_training_ids=set(),
        )
        calibrator_path = settings.models_dir / f"calibrator_h{horizon}.json"
        calibrator.save(calibrator_path)
        entry = {
            "direction_source": direction_source,
            "direction_evidence": direction_evidence,
            "calibrator": str(calibrator_path),
        }
        if horizon in settings.return_horizons:
            return_source, return_evidence = select_return_champion(history)
            entry.update({"return_source": return_source, "return_evidence": return_evidence})
        latest_models[str(horizon)] = entry
    manifest = {
        "version": settings.version, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_validation_lock": parent_lock["lock_sha256"], "plan_lock": plan_lock["lock_sha256"],
        "parent_model_manifest": str(settings.parent_dir / "models" / "manifest.json"),
        "parent_model_manifest_sha256": digest(settings.parent_dir / "models" / "manifest.json"),
        "training_cutoff": parent_report["data"]["date_max"], "models": latest_models,
        "execution_authorized": False,
    }
    _write_json(settings.models_dir / "manifest.json", manifest)
    evidence.update({
        "parent_lock_intact": parent_lock["intact"], "plan_lock_intact": plan_lock["intact"],
        "all_calibrators_monotonic": monotonic,
        "retained_return_heads_valid": retained_return_heads_valid,
    })
    report = {
        "version": settings.version, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent": "V30", "repair_scope": "monotonic calibration and past-OOS champion selection",
        "data": parent_report["data"], "oos_years": list(settings.oos_years),
        "purge_gaps": settings.purge_gaps, "folds": parent_report["folds"],
        "selection": selections, "certification": revised.to_dict(),
        "certification_evidence": evidence,
        "production_prediction_ready": revised.production_prediction_ready,
        "future_126d_confirmed": False, "execution_authorized": False,
    }
    _write_json(report_path, report)
    progress.write("complete", f"V30r1 replay complete; prediction_ready={revised.production_prediction_ready}")
    return report
