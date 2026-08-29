from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_v10.features import V10_FEATURES, build_v10_dataset
from research_v10.fundamentals import attach_extended_fundamentals_asof, load_extended_fundamentals
from research_v9.data import attach_industry_asof, attach_membership_weight, load_industry_history
from stockpilot.data import load_panel, validate_panel
from stockpilot.membership import attach_point_in_time_membership, load_membership_history
from stockpilot.prediction.certification import load_certification
from stockpilot.prediction.confidence import confidence_scores
from stockpilot.prediction.drift import drift_from_profile
from stockpilot.prediction.freeze import digest, verify_validation_lock
from stockpilot.prediction.metrics import expected_calibration_error
from stockpilot.prediction.models import LightGBMDirection, LightGBMReturn, LogisticRidge, RidgeReturn
from stockpilot.prediction.settlement import update_prediction_ledger
from stockpilot.prediction.storage import (
    write_immutable_prediction_snapshot,
    write_latest_metadata,
)
from stockpilot.prediction_audit import verify_result_lock
from stockpilot.prediction_v30r1.calibration import MonotonicPlattCalibrator
from stockpilot.prediction_v30r1.config import V30R1Settings


@dataclass(frozen=True)
class ForwardPredictionSettings:
    version: str = "V30r1-forward"
    artifact_dir: Path = Path("artifacts/prediction_forward/v30r1")
    frozen_market_path: Path = Path("data/market_history_v10_hfq.csv")
    membership_path: Path = Path("data/universes/000300/history_v10.csv")
    fundamental_path: Path = Path("data/fundamentals_pit_v10_extended.csv")
    industry_path: Path = Path("data/industry_history_v10.csv")
    names_path: Path = Path("data/stock_names.csv")
    parent_root: Path = Path("artifacts/prediction_v30r1")
    parent_v30_root: Path = Path("artifacts/prediction_v30")
    feature_lookback_calendar_days: int = 1200
    minimum_current_coverage: float = 0.95
    minimum_overlap_days: int = 10
    maximum_ratio_relative_deviation: float = 0.0025
    maximum_return_absolute_difference: float = 0.0025

    @property
    def prediction_dir(self) -> Path:
        return self.artifact_dir / "predictions"

    @property
    def feature_dir(self) -> Path:
        return self.artifact_dir / "features"

    @property
    def audit_dir(self) -> Path:
        return self.artifact_dir / "audit"


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def _normalize_market(frame: pd.DataFrame) -> pd.DataFrame:
    data = validate_panel(frame)
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data["symbol"] = data["symbol"].astype(str).str.zfill(6)
    return data


def _current_members(membership: pd.DataFrame, as_of: pd.Timestamp) -> tuple[pd.Timestamp, set[str]]:
    history = membership[pd.to_datetime(membership["snapshot_date"]) <= as_of]
    if history.empty:
        raise RuntimeError("no point-in-time membership snapshot is available")
    snapshot = pd.to_datetime(history["snapshot_date"]).max()
    symbols = set(
        history.loc[pd.to_datetime(history["snapshot_date"]).eq(snapshot), "symbol"]
        .astype(str)
        .str.zfill(6)
    )
    return snapshot, symbols


def stitch_hfq_market(
    frozen: pd.DataFrame,
    incremental: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    cutoff: str | pd.Timestamp,
    as_of: str | pd.Timestamp,
    settings: ForwardPredictionSettings | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Append HFQ bars without rewriting the frozen price scale.

    HFQ providers restate old prices after a corporate action.  A per-symbol
    overlap factor anchors the new download to the already frozen scale.  The
    overlap must preserve returns and have one stable factor across OHLC.
    """
    settings = settings or ForwardPredictionSettings()
    frozen = _normalize_market(frozen)
    incremental = _normalize_market(incremental)
    cutoff_date = pd.Timestamp(cutoff).normalize()
    as_of_date = pd.Timestamp(as_of).normalize()
    if as_of_date <= cutoff_date:
        raise ValueError("as_of must be later than the frozen cutoff")
    incremental = incremental[incremental["date"] <= as_of_date].copy()
    overlap = frozen[frozen["date"] <= cutoff_date].merge(
        incremental[incremental["date"] <= cutoff_date],
        on=["date", "symbol"], suffixes=("_frozen", "_new"), validate="one_to_one",
    )
    factors: dict[str, float] = {}
    diagnostics: list[dict] = []
    rejected_symbols: set[str] = set()
    for symbol, group in overlap.groupby("symbol", sort=True):
        if group["date"].nunique() < settings.minimum_overlap_days:
            continue
        ratios: list[float] = []
        for column in ("open", "high", "low", "close"):
            old = pd.to_numeric(group[f"{column}_frozen"], errors="coerce")
            new = pd.to_numeric(group[f"{column}_new"], errors="coerce")
            ratios.extend((old / new).replace([np.inf, -np.inf], np.nan).dropna().tolist())
        factor = float(np.median(ratios))
        relative_deviation = float(np.max(np.abs(np.asarray(ratios) / factor - 1)))
        ordered = group.sort_values("date")
        old_return = pd.to_numeric(ordered["close_frozen"], errors="coerce").pct_change()
        new_return = pd.to_numeric(ordered["close_new"], errors="coerce").pct_change()
        return_difference = float((old_return - new_return).abs().max())
        accepted = bool(
            relative_deviation <= settings.maximum_ratio_relative_deviation
            and return_difference <= settings.maximum_return_absolute_difference
        )
        diagnostics.append({
            "symbol": symbol,
            "overlap_days": int(group["date"].nunique()),
            "stitch_factor": factor,
            "max_ratio_relative_deviation": relative_deviation,
            "max_return_absolute_difference": return_difference,
            "accepted": accepted,
        })
        if accepted:
            factors[symbol] = factor
        else:
            rejected_symbols.add(symbol)
    snapshot, current_members = _current_members(membership, as_of_date)
    latest = incremental[incremental["date"].eq(as_of_date)]
    covered_current = current_members.intersection(set(latest["symbol"]))
    coverage = len(covered_current) / len(current_members) if current_members else 0.0
    unanchored = sorted(covered_current.difference(factors))
    if coverage < settings.minimum_current_coverage:
        raise RuntimeError(f"current PIT membership market coverage is too low: {coverage:.3%}")
    if unanchored:
        raise RuntimeError(
            "current symbols lack a valid HFQ overlap anchor or failed overlap consistency: "
            + ",".join(unanchored[:20])
        )
    extension = incremental[incremental["date"] > cutoff_date].copy()
    extension["stitch_factor"] = extension["symbol"].map(factors)
    extension = extension[extension["stitch_factor"].notna()].copy()
    for column in ("open", "high", "low", "close"):
        extension[column] = pd.to_numeric(extension[column], errors="coerce") * extension["stitch_factor"]
    extension = extension.drop(columns="stitch_factor")
    combined = pd.concat(
        [frozen[frozen["date"] <= cutoff_date], extension], ignore_index=True, sort=False,
    )
    combined = _normalize_market(combined)
    if combined.duplicated(["date", "symbol"]).any():
        raise RuntimeError("stitched market contains duplicate date/symbol rows")
    diagnostics_frame = pd.DataFrame(diagnostics)
    audit = {
        "cutoff": str(cutoff_date.date()),
        "as_of": str(as_of_date.date()),
        "membership_snapshot": str(snapshot.date()),
        "membership_size": len(current_members),
        "latest_current_members_covered": len(covered_current),
        "latest_current_coverage": coverage,
        "latest_all_symbols": int(latest["symbol"].nunique()),
        "anchored_symbols": len(factors),
        "isolated_noncurrent_symbols": sorted(rejected_symbols),
        "isolated_noncurrent_symbol_count": len(rejected_symbols),
        "overlap_rows": int(len(overlap)),
        "extension_rows": int(len(extension)),
        "maximum_observed_ratio_relative_deviation": float(
            diagnostics_frame["max_ratio_relative_deviation"].max()
        ),
        "maximum_observed_return_absolute_difference": float(
            diagnostics_frame["max_return_absolute_difference"].max()
        ),
        "passed": True,
        "diagnostics": diagnostics,
    }
    return combined, audit


def build_latest_pit_feature_panel(
    market: pd.DataFrame,
    as_of: str | pd.Timestamp,
    *,
    ranking_path: Path | None = None,
    settings: ForwardPredictionSettings | None = None,
) -> tuple[pd.DataFrame, dict]:
    settings = settings or ForwardPredictionSettings()
    as_of_date = pd.Timestamp(as_of).normalize()
    market = _normalize_market(market)
    full_history = market.sort_values(["symbol", "date"])
    full_history_returns = full_history.groupby("symbol")["close"].pct_change()
    full_history_bad = set(
        full_history_returns.abs().groupby(full_history["symbol"]).max()
        .loc[lambda values: values > 0.35].index.astype(str)
    )
    window_start = as_of_date - pd.Timedelta(days=settings.feature_lookback_calendar_days)
    panel = market[market["date"].between(window_start, as_of_date)].copy()
    membership = load_membership_history(settings.membership_path)
    panel = attach_point_in_time_membership(panel, membership)
    panel = attach_membership_weight(panel, membership)
    panel = attach_extended_fundamentals_asof(
        panel, load_extended_fundamentals(settings.fundamental_path),
    )
    panel = attach_industry_asof(panel, load_industry_history(settings.industry_path))
    dataset = build_v10_dataset(panel)
    # build_dataset computes its quality flag over the supplied frame.  A bounded
    # feature window must still inherit the full frozen-history anomaly decision.
    dataset.loc[dataset["symbol"].isin(full_history_bad), "eligible"] = False
    eligible = dataset["eligible"].fillna(False)
    dataset["market_momentum_60"] = (
        dataset["momentum_60"].where(eligible).groupby(dataset["date"]).transform("mean")
    )
    dataset["positive_20d_breadth"] = (
        dataset["ret_20"].gt(0).where(eligible).groupby(dataset["date"]).transform("mean")
    )
    risk_on = (dataset["market_momentum_60"] > 0.02) & (dataset["positive_20d_breadth"] > 0.55)
    risk_off = (dataset["market_momentum_60"] < -0.02) & (dataset["positive_20d_breadth"] < 0.45)
    dataset["regime"] = np.select([risk_on, risk_off], ["risk_on", "risk_off"], "neutral")
    current = dataset[
        pd.to_datetime(dataset["date"]).eq(as_of_date) & dataset["eligible"].fillna(False)
    ].copy()
    if current.empty:
        raise RuntimeError(f"no eligible PIT predictions for {as_of_date.date()}")
    names = (
        pd.read_csv(settings.names_path, dtype={"symbol": str})
        if settings.names_path.exists()
        else pd.DataFrame(columns=["symbol", "name"])
    )
    names["symbol"] = names["symbol"].astype(str).str.zfill(6)
    current = current.merge(
        names[["symbol", "name"]].drop_duplicates("symbol", keep="last"),
        on="symbol", how="left",
    )
    current["name"] = current["name"].fillna("")
    if ranking_path is not None:
        ranking = pd.read_csv(ranking_path, dtype={"symbol": str})
        ranking["symbol"] = ranking["symbol"].astype(str).str.zfill(6)
        ranking["date"] = pd.to_datetime(ranking["date"]).dt.normalize()
        ranking = ranking[ranking["date"].eq(as_of_date)]
        if ranking.empty or "score" not in ranking:
            raise RuntimeError("same-date V6 ranking evidence is missing")
        ranking["ranking_component"] = pd.to_numeric(ranking["score"], errors="coerce").rank(
            pct=True, method="average"
        )
        current = current.merge(
            ranking[["symbol", "ranking_component"]], on="symbol", how="left", validate="one_to_one",
        )
        ranking_coverage = float(current["ranking_component"].notna().mean())
        current["ranking_component"] = current["ranking_component"].fillna(0.5)
    else:
        current["ranking_component"] = 0.5
        ranking_coverage = 0.0
    columns = list(dict.fromkeys([
        "date", "symbol", "name", "open", "close", "broad_sector", "regime",
        "volatility_20", "ranking_component", *V10_FEATURES,
    ]))
    current = current[columns].sort_values("symbol").reset_index(drop=True)
    audit = {
        "as_of": str(as_of_date.date()),
        "window_start": str(window_start.date()),
        "market_rows": int(len(panel)),
        "market_symbols": int(panel["symbol"].nunique()),
        "eligible_predictions": int(len(current)),
        "full_history_bad_price_symbols": len(full_history_bad),
        "membership_not_future": bool(
            (
                dataset["membership_snapshot_date"].isna()
                | (pd.to_datetime(dataset["membership_snapshot_date"]) <= dataset["date"])
            ).all()
        ),
        "fundamentals_not_future": bool(
            (
                dataset["available_date"].isna()
                | (pd.to_datetime(dataset["available_date"]) <= dataset["date"])
            ).all()
        ),
        "industry_not_future": bool(
            (
                dataset["industry_effective_date"].isna()
                | (pd.to_datetime(dataset["industry_effective_date"]) <= dataset["date"])
            ).all()
        ),
        "ranking_coverage": ranking_coverage,
    }
    audit["passed"] = bool(
        audit["membership_not_future"]
        and audit["fundamentals_not_future"]
        and audit["industry_not_future"]
        and len(current) >= 200
        and (ranking_path is None or ranking_coverage >= 0.95)
    )
    if not audit["passed"]:
        raise RuntimeError(f"latest PIT feature audit failed: {audit}")
    return current, audit


def compare_feature_panel(
    actual: pd.DataFrame,
    expected_path: Path,
    *,
    numeric_tolerance: float = 1e-10,
) -> dict:
    expected = pd.read_csv(expected_path, dtype={"symbol": str})
    expected["symbol"] = expected["symbol"].astype(str).str.zfill(6)
    actual = actual.copy()
    actual["symbol"] = actual["symbol"].astype(str).str.zfill(6)
    if set(actual["symbol"]) != set(expected["symbol"]):
        return {
            "passed": False,
            "reason": "symbol_set_mismatch",
            "actual_symbols": len(set(actual["symbol"])),
            "expected_symbols": len(set(expected["symbol"])),
        }
    merged = expected.merge(actual, on="symbol", suffixes=("_expected", "_actual"), validate="one_to_one")
    numeric_columns = ["open", "close", "volatility_20", *V10_FEATURES]
    differences = {}
    for column in dict.fromkeys(numeric_columns):
        left = pd.to_numeric(merged[f"{column}_expected"], errors="coerce")
        right = pd.to_numeric(merged[f"{column}_actual"], errors="coerce")
        differences[column] = float((left - right).abs().max())
    categorical = {
        column: bool((merged[f"{column}_expected"].fillna("") == merged[f"{column}_actual"].fillna("")).all())
        for column in ("broad_sector", "regime")
    }
    passed = max(differences.values()) <= numeric_tolerance and all(categorical.values())
    return {
        "passed": bool(passed),
        "symbols": len(merged),
        "numeric_tolerance": numeric_tolerance,
        "maximum_numeric_difference": max(differences.values()),
        "numeric_differences": differences,
        "categorical_equal": categorical,
    }


def _immutable_json(path: Path, payload: dict) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != encoded:
        raise RuntimeError(f"immutable audit hash mismatch: {path}")
    if not path.exists():
        path.write_bytes(encoded)


def create_forward_plan_lock(settings: ForwardPredictionSettings | None = None) -> dict:
    settings = settings or ForwardPredictionSettings()
    target = settings.artifact_dir / "plan.lock.json"
    if target.exists():
        raise RuntimeError(f"forward plan lock already exists: {target}")
    files = [
        settings.artifact_dir / "protocol.json",
        Path("stockpilot/prediction_forward.py"),
        Path("stockpilot/cli.py"),
        Path("stockpilot/api.py"),
        Path("dashboard.py"),
        Path("tests/test_prediction_forward.py"),
        settings.parent_root / "validation.lock.json",
        settings.parent_v30_root / "validation.lock.json",
    ]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise RuntimeError("cannot freeze forward plan: " + ", ".join(missing))
    payload = {
        "version": settings.version,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "inference_only_no_retraining",
        "production_prediction_ready_may_not_be_promoted": True,
        "execution_authorized": False,
        "files": {path.as_posix(): _sha256(path) for path in files},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload | {"lock_sha256": _sha256(target)}


def verify_forward_plan_lock(settings: ForwardPredictionSettings | None = None) -> dict:
    settings = settings or ForwardPredictionSettings()
    target = settings.artifact_dir / "plan.lock.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    mismatches = [
        name for name, expected in payload["files"].items()
        if not Path(name).exists() or _sha256(Path(name)) != expected
    ]
    return {"intact": not mismatches, "mismatches": mismatches, "lock_sha256": _sha256(target)}


def _generate_from_panel(
    current: pd.DataFrame,
    combined_market: pd.DataFrame,
    as_of: pd.Timestamp,
    settings: ForwardPredictionSettings,
) -> dict:
    parent_settings = V30R1Settings()
    manifest_path = settings.parent_root / "models" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parent_manifest = json.loads(Path(manifest["parent_model_manifest"]).read_text(encoding="utf-8"))
    certification = load_certification(settings.parent_root / "certification" / "status.json")
    chosen_raw: dict[int, np.ndarray] = {}
    calibrated: dict[int, np.ndarray] = {}
    expected: dict[int, np.ndarray] = {}
    for horizon in parent_settings.horizons:
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
        if horizon in parent_settings.return_horizons:
            if selection["return_source"] == "lightgbm":
                return_model = LightGBMReturn.load(Path(parent["return"]))
            else:
                return_model = RidgeReturn.load(Path(parent["ridge_baseline"]))
            expected[horizon] = return_model.predict(current)
    profile = json.loads(
        (settings.parent_v30_root / "models" / "training_feature_profile.json").read_text(encoding="utf-8")
    )
    drift, drift_status, drift_multiplier = drift_from_profile(
        profile, current,
        psi_warning=parent_settings.psi_warning, psi_severe=parent_settings.psi_severe,
        zscore_warning=parent_settings.zscore_warning, zscore_severe=parent_settings.zscore_severe,
    )
    drift_path = settings.artifact_dir / "drift" / f"{as_of.date()}.csv"
    drift_payload = drift.sort_values(list(drift.columns)).to_csv(
        index=False, lineterminator="\n", float_format="%.10g"
    ).encode("utf-8-sig")
    drift_path.parent.mkdir(parents=True, exist_ok=True)
    if drift_path.exists() and drift_path.read_bytes() != drift_payload:
        raise RuntimeError(f"immutable drift report hash mismatch: {drift_path}")
    if not drift_path.exists():
        drift_path.write_bytes(drift_payload)
    yearly = pd.read_csv(settings.parent_root / "validation" / "yearly_metrics.csv")
    regime = pd.read_csv(settings.parent_root / "validation" / "regime_metrics.csv")
    sector = pd.read_csv(settings.parent_root / "validation" / "sector_metrics.csv")
    calibration = pd.read_csv(settings.parent_root / "validation" / "calibration_table.csv")
    critical = yearly[yearly["horizon"].isin((5, 20))]
    oos_skill = float(np.clip((critical["roc_auc"].mean() - 0.5) / 0.10, 0, 1))
    calibration_quality = float(np.clip(
        1 - expected_calibration_error(calibration[calibration["horizon"].isin((5, 20))]) / 0.10,
        0, 1,
    ))
    regime_consistency = float((regime[regime["horizon"].isin((5, 20))]["roc_auc"] > 0.5).mean())
    sector_quality = (
        sector[sector["horizon"].isin((5, 20))].groupby("broad_sector")["roc_auc"].mean()
        .sub(0.5).div(0.1).clip(0, 1)
    )
    combined_probability = pd.Series((calibrated[5] + calibrated[20]) / 2, index=current.index)
    confidence_score, confidence_level = confidence_scores(
        combined_probability, oos_skill=oos_skill, calibration_quality=calibration_quality,
        regime_consistency=regime_consistency,
        sector_stability=current["broad_sector"].map(sector_quality).fillna(0),
        feature_completeness=current[V10_FEATURES].notna().mean(axis=1),
        drift_multiplier=drift_multiplier,
        low_upper=parent_settings.low_confidence_upper,
        medium_upper=parent_settings.medium_confidence_upper,
    )
    output = current[[
        "date", "symbol", "name", "close", "broad_sector", "regime", "ranking_component",
    ]].copy()
    for horizon in parent_settings.horizons:
        output[f"p_up_{horizon}d_raw"] = chosen_raw[horizon]
        output[f"p_up_{horizon}d"] = calibrated[horizon]
        output[f"p_up_{horizon}d_rank"] = output[f"p_up_{horizon}d"].rank(pct=True, method="average")
        output[f"rank_{horizon}d"] = output[f"p_up_{horizon}d"].rank(
            ascending=False, method="first"
        ).astype(int)
    for horizon in parent_settings.return_horizons:
        output[f"expected_return_{horizon}d"] = expected[horizon]
    output["confidence_score"], output["confidence_level"] = confidence_score, confidence_level
    volatility_rank = current["volatility_20"].rank(pct=True).fillna(0.5)
    output["risk_penalty"] = volatility_rank
    output["risk_level"] = pd.cut(
        volatility_rank, [-np.inf, 0.33, 0.67, np.inf], labels=["LOW", "MEDIUM", "HIGH"],
    ).astype(str)
    output["probability_component"] = combined_probability.rank(pct=True)
    output["expected_return_component"] = pd.Series(
        expected[5] + expected[20], index=current.index,
    ).rank(pct=True)
    output["candidate_score"] = (
        parent_settings.candidate_ranking_weight * output["ranking_component"]
        + parent_settings.candidate_probability_weight * output["probability_component"]
        + parent_settings.candidate_return_weight * output["expected_return_component"]
        - parent_settings.candidate_risk_penalty * output["risk_penalty"]
    )
    output["prediction_ready"] = bool(certification.production_prediction_ready)
    output["calibration_status"] = "PASSED" if certification.calibration_passed else "FAILED"
    output["drift_status"] = drift_status
    output["model_version"] = f"V30r1:{digest(manifest_path)[:12]}"
    output["training_cutoff"] = manifest["training_cutoff"]
    snapshot = settings.prediction_dir / f"{as_of.date()}.csv"
    if snapshot.exists():
        generated_at = pd.read_csv(snapshot, usecols=["generated_at_utc"])["generated_at_utc"].iloc[0]
    else:
        generated_at = datetime.now(timezone.utc).isoformat()
    output["generated_at_utc"] = generated_at
    output["execution_authorized"] = False
    output = output.sort_values(["rank_5d", "symbol"]).reset_index(drop=True)
    wrote, snapshot_hash = write_immutable_prediction_snapshot(output, snapshot)
    ledger = update_prediction_ledger(
        settings.prediction_dir, combined_market,
        settings.artifact_dir / "prediction_ledger.csv",
        direction_thresholds=parent_settings.direction_thresholds,
    )
    result = {
        "prediction_date": str(as_of.date()),
        "prediction_count": len(output),
        "snapshot_path": str(snapshot),
        "sha256": snapshot_hash,
        "already_recorded": not wrote,
        "model_version": output["model_version"].iloc[0],
        "training_cutoff": manifest["training_cutoff"],
        "production_prediction_ready": bool(certification.production_prediction_ready),
        "future_126d_confirmed": False,
        "future_confirmation_status": "COLLECTING",
        "execution_authorized": False,
        "drift_status": drift_status,
        "ledger_rows": len(ledger),
        "settled_rows": int(ledger["settled"].fillna(False).sum()),
    }
    write_latest_metadata(settings.artifact_dir / "latest.json", result)
    return result


def run_forward_prediction(
    incremental_market_path: str | Path,
    as_of: str | pd.Timestamp,
    *,
    ranking_path: str | Path,
    settings: ForwardPredictionSettings | None = None,
) -> dict:
    settings = settings or ForwardPredictionSettings()
    as_of_date = pd.Timestamp(as_of).normalize()
    v30_lock = verify_validation_lock()
    parent_lock = verify_result_lock(settings.parent_root)
    forward_lock = verify_forward_plan_lock(settings)
    if not v30_lock["intact"] or not parent_lock["intact"] or not forward_lock["intact"]:
        raise RuntimeError("a frozen V30/V30r1/forward input is not intact")
    frozen = load_panel(settings.frozen_market_path)
    cutoff = pd.to_datetime(frozen["date"]).max()
    incremental_path = Path(incremental_market_path)
    ranking_path = Path(ranking_path)
    membership = load_membership_history(settings.membership_path)
    combined, market_audit = stitch_hfq_market(
        frozen, load_panel(incremental_path), membership,
        cutoff=cutoff, as_of=as_of_date, settings=settings,
    )
    parity_panel, parity_pit = build_latest_pit_feature_panel(
        combined[combined["date"] <= cutoff], cutoff, settings=settings,
    )
    parity = compare_feature_panel(
        parity_panel, settings.parent_v30_root / "models" / "latest_feature_panel.csv",
    )
    if not parity["passed"]:
        raise RuntimeError(f"frozen feature parity failed: {parity}")
    current, pit_audit = build_latest_pit_feature_panel(
        combined, as_of_date, ranking_path=ranking_path, settings=settings,
    )
    feature_path = settings.feature_dir / f"{as_of_date.date()}.csv"
    write_immutable_prediction_snapshot(current, feature_path)
    audit_path = settings.audit_dir / f"{as_of_date.date()}.json"
    if audit_path.exists():
        audit_generated_at = json.loads(audit_path.read_text(encoding="utf-8"))["generated_at_utc"]
    else:
        audit_generated_at = datetime.now(timezone.utc).isoformat()
    audit = {
        "version": settings.version,
        "generated_at_utc": audit_generated_at,
        "parent_v30_lock": v30_lock,
        "parent_v30r1_lock": parent_lock,
        "forward_plan_lock": forward_lock,
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
        "execution_authorized": False,
    }
    _immutable_json(audit_path, audit)
    result = _generate_from_panel(current, combined, as_of_date, settings)
    result["market_audit_passed"] = True
    result["feature_parity_passed"] = True
    result["pit_audit_passed"] = True
    write_latest_metadata(settings.artifact_dir / "latest.json", result)
    return result


def forward_status(settings: ForwardPredictionSettings | None = None) -> dict:
    settings = settings or ForwardPredictionSettings()
    latest = settings.artifact_dir / "latest.json"
    result = (
        json.loads(latest.read_text(encoding="utf-8"))
        if latest.exists()
        else {"version": settings.version, "status": "NOT_RUN"}
    )
    result["parent_v30_lock"] = verify_validation_lock()
    result["parent_v30r1_lock"] = verify_result_lock(settings.parent_root)
    if (settings.artifact_dir / "plan.lock.json").exists():
        result["forward_plan_lock"] = verify_forward_plan_lock(settings)
    result["execution_authorized"] = False
    return result
