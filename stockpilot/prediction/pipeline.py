from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_v10.features import V10_FEATURES

from .calibration import PlattCalibrator
from .certification import PredictionCertificationResult
from .config import PredictionSettings
from .data import load_prediction_dataset, pit_data_audit
from .drift import build_reference_profile
from .metrics import binary_metrics, calibration_table, expected_calibration_error, rank_ic_by_date
from .models import LightGBMDirection, LightGBMReturn, LogisticRidge, RidgeReturn, deterministic_sample
from .split import PurgedWalkForwardSplit


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


class _Progress:
    def __init__(self, settings: PredictionSettings) -> None:
        self.log_path = settings.artifact_dir / "validation.log"
        self.status_path = settings.artifact_dir / "runtime_status.json"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, stage: str, message: str, **extra: Any) -> None:
        payload = {"updated_at_utc": _utc(), "stage": stage, "message": message, **extra}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        _json(self.status_path, payload)
        print(f"[{stage}] {message}", flush=True)


def _fold(frame: pd.DataFrame, year: int, horizon: int, settings: PredictionSettings):
    splitter = PurgedWalkForwardSplit(
        (year,), settings.purge_gaps[horizon], settings.training_window_years
    )
    folds = splitter.split(frame, horizon)
    if len(folds) != 1:
        raise RuntimeError(f"missing purged fold for horizon={horizon}, year={year}")
    return folds[0]


def _ids(frame: pd.DataFrame) -> set[str]:
    return set(pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d") + ":" + frame["symbol"].astype(str))


def _market_regime(values: pd.Series) -> pd.Series:
    return values.map({"risk_on": "bull", "risk_off": "bear", "neutral": "sideways"}).fillna("sideways")


def _fold_direction_predictions(
    data: pd.DataFrame,
    horizon: int,
    year: int,
    settings: PredictionSettings,
    progress: _Progress,
) -> tuple[pd.DataFrame, dict]:
    fold = _fold(data, year, horizon, settings)
    train = deterministic_sample(data.loc[fold.train_index].sort_values(["date", "symbol"]), settings.training_row_cap)
    validation = data.loc[fold.validation_index].sort_values(["date", "symbol"]).copy()
    target = f"tradable_up_{horizon}d"
    progress.write("direction", f"H{horizon} {year}: fitting {len(train):,} rows; validating {len(validation):,}")
    model = LightGBMDirection().fit(train, V10_FEATURES, target)
    logistic = LogisticRidge(settings.ridge_alpha, settings.logistic_max_iter).fit(train, V10_FEATURES, target)
    validation["raw_probability"] = model.predict_proba(validation)
    validation["logistic_probability"] = logistic.predict_proba(validation)
    prevalence = float(train[target].mean())
    validation["unconditional_probability"] = prevalence
    recent_dates = pd.DatetimeIndex(pd.to_datetime(train["date"]).drop_duplicates().sort_values())[-252:]
    rolling = train[pd.to_datetime(train["date"]).isin(recent_dates)][target].mean()
    validation["rolling_market_probability"] = float(rolling)
    momentum = pd.to_numeric(validation["momentum"], errors="coerce").fillna(0).clip(-0.5, 0.5)
    validation["momentum_probability"] = (0.5 + 0.30 * momentum).clip(0.01, 0.99)
    validation["actual"] = pd.to_numeric(validation[target], errors="coerce")
    validation["actual_return"] = pd.to_numeric(validation[f"future_return_{horizon}d"], errors="coerce")
    validation["horizon"] = horizon
    validation["test_year"] = year
    validation["market_regime"] = _market_regime(validation["regime"])
    train_daily_vol = train.groupby("date")["volatility_20"].mean()
    vol_cutoff = float(train_daily_vol.median())
    validation_daily_vol = validation.groupby("date")["volatility_20"].transform("mean")
    validation["volatility_regime"] = np.where(validation_daily_vol >= vol_cutoff, "high_volatility", "low_volatility")
    proof = {
        "year": year, "horizon": horizon, "train_rows": len(train), "validation_rows": len(validation),
        "train_start": str(pd.to_datetime(train["date"]).min().date()),
        "train_decision_end": str(pd.to_datetime(train["date"]).max().date()),
        "train_label_end": str(pd.to_datetime(train[f"label_end_date_{horizon}d"]).max().date()),
        "validation_start": str(fold.validation_start.date()), "validation_end": str(fold.validation_end.date()),
        "purge_cutoff": str(fold.purge_cutoff.date()), "purge_gap": fold.purge_gap_trading_days,
        "label_maturity_passed": bool(pd.to_datetime(train[f"label_end_date_{horizon}d"]).max() < fold.validation_start),
        "training_validation_overlap": len(_ids(train).intersection(_ids(validation))),
    }
    keep = [
        "date", "symbol", "close", "broad_sector", "regime", "market_regime", "volatility_regime",
        "horizon", "test_year", "actual", "actual_return", "raw_probability",
        "logistic_probability", "unconditional_probability", "rolling_market_probability",
        "momentum_probability",
    ]
    return validation[keep], proof


def _fit_return_fold(
    data: pd.DataFrame,
    horizon: int,
    year: int,
    settings: PredictionSettings,
    progress: _Progress,
) -> pd.DataFrame:
    fold = _fold(data, year, horizon, settings)
    train = deterministic_sample(data.loc[fold.train_index].sort_values(["date", "symbol"]), settings.training_row_cap)
    validation = data.loc[fold.validation_index].sort_values(["date", "symbol"])
    target = f"future_return_{horizon}d"
    progress.write("return", f"H{horizon} {year}: fitting return heads")
    model = LightGBMReturn().fit(train, V10_FEATURES, target)
    ridge = RidgeReturn(settings.ridge_alpha).fit(train, V10_FEATURES, target)
    return pd.DataFrame({
        "date": pd.to_datetime(validation["date"]), "symbol": validation["symbol"].astype(str),
        "expected_return": model.predict(validation), "ridge_expected_return": ridge.predict(validation),
    })


def _aggregate_reports(oos: pd.DataFrame, settings: PredictionSettings) -> tuple[pd.DataFrame, ...]:
    yearly_rows, baseline_rows, calibration_rows, regime_rows, sector_rows = [], [], [], [], []
    baseline_columns = {
        "lightgbm_calibrated": "probability", "unconditional": "unconditional_probability",
        "rolling_market": "rolling_market_probability", "momentum": "momentum_probability",
        "logistic_ridge": "logistic_probability",
    }
    for horizon, group in oos.groupby("horizon"):
        for model_name, column in baseline_columns.items():
            metrics = binary_metrics(group["actual"], group[column])
            metrics.update({"horizon": int(horizon), "model": model_name})
            baseline_rows.append(metrics)
        table = calibration_table(group["actual"].to_numpy(), group["probability"].to_numpy())
        table.insert(0, "horizon", int(horizon))
        calibration_rows.extend(table.to_dict(orient="records"))
        for year, year_group in group.groupby("test_year"):
            metrics = binary_metrics(year_group["actual"], year_group["probability"])
            metrics.update({
                "horizon": int(horizon), "test_year": int(year),
                "rank_ic": rank_ic_by_date(year_group, "probability", "actual_return"),
                "mean_expected_return": float(year_group.get("expected_return", pd.Series(dtype=float)).mean()),
                "actual_return": float(year_group["actual_return"].mean()),
                "win_rate": float(year_group["actual"].mean()),
            })
            yearly_rows.append(metrics)
        for dimension, column in (("market", "market_regime"), ("volatility", "volatility_regime")):
            for regime, part in group.groupby(column):
                metrics = binary_metrics(part["actual"], part["probability"])
                metrics.update({
                    "horizon": int(horizon), "regime_dimension": dimension, "regime": str(regime),
                    "rank_ic": rank_ic_by_date(part, "probability", "actual_return"),
                    "mean_expected_return": float(part.get("expected_return", pd.Series(dtype=float)).mean()),
                    "actual_return": float(part["actual_return"].mean()), "win_rate": float(part["actual"].mean()),
                })
                regime_rows.append(metrics)
        for sector, part in group.groupby("broad_sector"):
            metrics = binary_metrics(part["actual"], part["probability"])
            metrics.update({
                "horizon": int(horizon), "broad_sector": str(sector),
                "mean_p_up": float(part["probability"].mean()), "actual_up_rate": float(part["actual"].mean()),
                "expected_return": float(part.get("expected_return", pd.Series(dtype=float)).mean()),
                "actual_return": float(part["actual_return"].mean()),
            })
            sector_rows.append(metrics)
    return tuple(pd.DataFrame(rows) for rows in (yearly_rows, regime_rows, sector_rows, calibration_rows, baseline_rows))


def _certify(
    audit: dict,
    oos: pd.DataFrame,
    yearly: pd.DataFrame,
    regime: pd.DataFrame,
    sector: pd.DataFrame,
    calibration: pd.DataFrame,
    baselines: pd.DataFrame,
    proofs: list[dict],
    settings: PredictionSettings,
) -> tuple[PredictionCertificationResult, dict]:
    critical = (5, 20)
    aggregate = baselines.set_index(["horizon", "model"])
    calibration_errors = {h: expected_calibration_error(calibration[calibration["horizon"] == h]) for h in critical}
    baseline_checks = {}
    probability_checks = {}
    stability_checks = {}
    stress_checks = {}
    for horizon in critical:
        model = aggregate.loc[(horizon, "lightgbm_calibrated")]
        naive = aggregate.loc[(horizon, "unconditional")]
        logistic = aggregate.loc[(horizon, "logistic_ridge")]
        baseline_checks[horizon] = bool(
            model["brier"] < naive["brier"] and model["log_loss"] < naive["log_loss"]
            and model["brier"] < logistic["brier"] and model["roc_auc"] >= logistic["roc_auc"]
        )
        part = oos[oos["horizon"] == horizon]
        threshold = part["probability"].quantile(0.90)
        top = part[part["probability"] >= threshold]
        probability_checks[horizon] = bool(
            model["roc_auc"] > settings.auc_minimum
            and rank_ic_by_date(part, "probability", "actual_return") > 0
            and top["actual_return"].mean() > part["actual_return"].mean()
            and top["actual"].mean() > part["actual"].mean()
        )
        annual = yearly[yearly["horizon"] == horizon]
        stability_checks[horizon] = int(((annual["roc_auc"] > 0.5) & (annual["rank_ic"] > 0)).sum()) >= settings.minimum_positive_skill_years
        stress_checks[horizon] = bool(top["actual_return"].mean() > settings.direction_thresholds[horizon])
    regime_passed_count = int(((regime["horizon"].isin(critical)) & (regime["roc_auc"] > 0.5)).groupby(regime["regime"]).any().sum())
    sector_passed_count = int(((sector["horizon"].isin(critical)) & (sector["roc_auc"] > 0.5)).groupby(sector["broad_sector"]).any().sum())
    checks = {
        "data_verified": bool(audit["passed"]),
        "pit_verified": all(audit["checks"][key] for key in ("membership_not_future", "fundamentals_not_future", "industry_not_future")),
        "label_maturity_verified": all(proof["label_maturity_passed"] for proof in proofs),
        "leakage_test_passed": all(proof["training_validation_overlap"] == 0 for proof in proofs),
        "purged_walk_forward_passed": all(proof["purge_gap"] >= settings.purge_gaps[proof["horizon"]] for proof in proofs),
        "calibration_passed": all(calibration_errors[h] < settings.calibration_error_maximum for h in critical),
        "baseline_beaten": all(baseline_checks.values()),
        "stability_passed": all(stability_checks.values()),
        "regime_passed": regime_passed_count >= settings.minimum_regimes_passed,
        "probability_quality_passed": all(probability_checks.values()),
        "cost_aware_stress_passed": all(stress_checks.values()),
    }
    result = PredictionCertificationResult.evaluate(future_126d_confirmed=False, **checks)
    evidence = {
        "calibration_ece": calibration_errors, "baseline_checks": baseline_checks,
        "probability_checks": probability_checks, "stability_checks": stability_checks,
        "stress_checks": stress_checks, "regimes_passed": regime_passed_count,
        "sectors_passed": sector_passed_count,
        "future_126d_used_as_prediction_gate": False,
    }
    return result, evidence


def _fit_latest_models(data: pd.DataFrame, oos: pd.DataFrame, settings: PredictionSettings, progress: _Progress) -> dict:
    latest_date = pd.to_datetime(data["date"]).max()
    manifest: dict[str, Any] = {
        "version": settings.version, "generated_at_utc": _utc(), "training_cutoff": str(latest_date.date()),
        "features": V10_FEATURES, "models": {},
        "input_hashes": {str(path): _sha256(path) for path in (
            settings.market_path, settings.membership_path, settings.fundamental_path, settings.industry_path
        )},
    }
    eligible = data[data["eligible"].fillna(False)].sort_values(["date", "symbol"])
    current = eligible[pd.to_datetime(eligible["date"]).eq(latest_date)].copy()
    from research_v4.config import V4Settings
    from research_v4.stability import learn_factor_specs
    from research_v5.models import fit_v5_models
    from research_v6.model import score_v6

    v5_models = fit_v5_models(data, int(latest_date.year))
    v4_specs, _ = learn_factor_specs(data, int(latest_date.year), V4Settings())
    v6_scored = score_v6(current, v5_models, v4_specs)
    current["ranking_component"] = v6_scored["score"].rank(pct=True, method="average").reindex(current.index).fillna(0.5)
    names = pd.read_csv(settings.names_path, dtype={"symbol": str}) if settings.names_path.exists() else pd.DataFrame(columns=["symbol", "name"])
    names["symbol"] = names["symbol"].astype(str).str.zfill(6)
    current = current.merge(names[["symbol", "name"]].drop_duplicates("symbol", keep="last"), on="symbol", how="left")
    current["name"] = current["name"].fillna("")
    latest_columns = list(dict.fromkeys([
        "date", "symbol", "name", "open", "close", "broad_sector", "regime",
        "volatility_20", "ranking_component", *V10_FEATURES,
    ]))
    current[latest_columns].to_csv(settings.models_dir / "latest_feature_panel.csv", index=False, encoding="utf-8-sig")
    latest_train_union = []
    for horizon in settings.horizons:
        label_end = pd.to_datetime(eligible[f"label_end_date_{horizon}d"])
        target = f"tradable_up_{horizon}d"
        train = eligible[eligible[target].notna() & label_end.le(latest_date)]
        train = train[pd.to_datetime(train["date"]).ge(latest_date - pd.DateOffset(years=settings.training_window_years))]
        train = deterministic_sample(train, settings.training_row_cap)
        latest_train_union.append(train[V10_FEATURES])
        progress.write("final-model", f"H{horizon}: fitting final direction model on {len(train):,} mature rows")
        direction = LightGBMDirection().fit(train, V10_FEATURES, target)
        direction_path = settings.models_dir / f"direction_h{horizon}.txt"
        direction.save(direction_path)
        calibration_year = max(settings.oos_years)
        calibration_rows = oos[(oos["horizon"] == horizon) & (oos["test_year"] == calibration_year)]
        calibrator = PlattCalibrator().fit(
            calibration_rows["raw_probability"].to_numpy(), calibration_rows["actual"].to_numpy(),
            calibration_ids=_ids(calibration_rows), model_training_ids=set(),
        )
        calibrator_path = settings.models_dir / f"calibrator_h{horizon}.json"
        calibrator.save(calibrator_path)
        entry = {
            "direction": str(direction_path), "calibrator": str(calibrator_path),
            "mature_training_rows": len(train),
            "mature_label_end": str(pd.to_datetime(train[f"label_end_date_{horizon}d"]).max().date()),
            "calibration_year": calibration_year,
        }
        if horizon in settings.return_horizons:
            progress.write("final-model", f"H{horizon}: fitting final return model")
            return_model = LightGBMReturn().fit(train, V10_FEATURES, f"future_return_{horizon}d")
            return_path = settings.models_dir / f"return_h{horizon}.txt"
            return_model.save(return_path)
            entry["return"] = str(return_path)
        logistic = LogisticRidge(settings.ridge_alpha, settings.logistic_max_iter).fit(train, V10_FEATURES, target)
        logistic_path = settings.models_dir / f"logistic_baseline_h{horizon}.json"
        logistic.save(logistic_path)
        entry["logistic_baseline"] = str(logistic_path)
        if horizon in settings.return_horizons:
            ridge = RidgeReturn(settings.ridge_alpha).fit(train, V10_FEATURES, f"future_return_{horizon}d")
            ridge_path = settings.models_dir / f"ridge_baseline_h{horizon}.json"
            ridge.save(ridge_path)
            entry["ridge_baseline"] = str(ridge_path)
        manifest["models"][str(horizon)] = entry
    profile = build_reference_profile(pd.concat(latest_train_union, ignore_index=True), V10_FEATURES)
    profile_path = settings.models_dir / "training_feature_profile.json"
    _json(profile_path, profile)
    manifest["training_feature_profile"] = str(profile_path)
    _json(settings.models_dir / "manifest.json", manifest)
    return manifest


def run_prediction_validation(settings: PredictionSettings | None = None) -> dict:
    settings = settings or PredictionSettings()
    settings.ensure_dirs()
    progress = _Progress(settings)
    report_path = settings.validation_dir / "report.json"
    if report_path.exists():
        progress.write("complete", "existing immutable V30 validation report returned")
        return json.loads(report_path.read_text(encoding="utf-8"))
    progress.write("data", "loading real PIT dataset")
    data = load_prediction_dataset(settings)
    audit = pit_data_audit(data)
    _json(settings.validation_dir / "data_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError("V30 real PIT data audit failed")
    data = data[data["eligible"].fillna(False)].reset_index(drop=True)
    progress.write("data", f"real PIT dataset ready: {len(data):,} eligible rows")
    support_years = (min(settings.oos_years) - 1, *settings.oos_years)
    raw_by_horizon: dict[int, pd.DataFrame] = {}
    proofs: list[dict] = []
    for horizon in settings.horizons:
        pieces = []
        for year in support_years:
            piece, proof = _fold_direction_predictions(data, horizon, year, settings, progress)
            pieces.append(piece)
            proofs.append(proof)
        raw_by_horizon[horizon] = pd.concat(pieces, ignore_index=True)
    calibrated_pieces = []
    for horizon, raw in raw_by_horizon.items():
        for year in settings.oos_years:
            previous = raw[raw["test_year"] == year - 1]
            current = raw[raw["test_year"] == year].copy()
            calibrator = PlattCalibrator().fit(
                previous["raw_probability"].to_numpy(), previous["actual"].to_numpy(),
                calibration_ids=_ids(previous),
                model_training_ids={f"model-train-before-{year - 1}"},
            )
            current["probability"] = calibrator.predict(current["raw_probability"].to_numpy())
            calibrated_pieces.append(current)
    oos = pd.concat(calibrated_pieces, ignore_index=True)
    return_pieces = []
    for horizon in settings.return_horizons:
        for year in settings.oos_years:
            result = _fit_return_fold(data, horizon, year, settings, progress)
            result["horizon"] = horizon
            return_pieces.append(result)
    returns = pd.concat(return_pieces, ignore_index=True)
    oos = oos.merge(returns, on=["date", "symbol", "horizon"], how="left", validate="one_to_one")
    oos.to_csv(settings.validation_dir / "oos_predictions.csv", index=False, encoding="utf-8-sig")
    yearly, regime, sector, calibration, baselines = _aggregate_reports(oos, settings)
    for frame, name in (
        (yearly, "yearly_metrics.csv"), (regime, "regime_metrics.csv"),
        (sector, "sector_metrics.csv"), (calibration, "calibration_table.csv"),
        (baselines, "baseline_comparison.csv"), (pd.DataFrame(proofs), "fold_audit.csv"),
    ):
        frame.to_csv(settings.validation_dir / name, index=False, encoding="utf-8-sig")
    certification, evidence = _certify(audit, oos, yearly, regime, sector, calibration, baselines, proofs, settings)
    certification.save(settings.certification_dir / "status.json")
    manifest = _fit_latest_models(data, oos, settings, progress)
    report = {
        "version": settings.version, "generated_at_utc": _utc(), "purpose": "probabilistic_prediction_layer",
        "data": audit, "oos_years": list(settings.oos_years), "purge_gaps": settings.purge_gaps,
        "features": len(V10_FEATURES), "folds": proofs, "certification": certification.to_dict(),
        "certification_evidence": evidence, "model_manifest": str(settings.models_dir / "manifest.json"),
        "production_prediction_ready": certification.production_prediction_ready,
        "future_126d_confirmed": certification.future_126d_confirmed,
        "execution_authorized": False,
    }
    _json(report_path, report)
    progress.write("complete", f"validation complete; prediction_ready={certification.production_prediction_ready}")
    return report
