"""Immutable forward-evidence registry around the frozen DAILY PIT Gen2 runtime.

This module is deliberately an observer/orchestrator.  It does not implement a
model, feature, label, ranking, portfolio, or execution policy.  Predictions and
settlements remain owned by the already-frozen 009/010/011 chain; this layer
copies their verified identities into a separate append-only evidence stream and
computes metrics only from mature settlement artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from stockpilot.daily_pit import pipeline as daily_pipeline
from stockpilot.daily_pit import runtime as daily_runtime
from stockpilot.prospective_r2.calendar import load_verified_calendar
from stockpilot.prospective_r2.integrity import (
    canonical_frame_bytes,
    canonical_json_bytes,
    read_verified_json,
    sha256_bytes,
    verify_immutable,
    write_immutable_json,
)
from stockpilot.research_challenger import prospective_gen2_runtime_locked as runtime010

SHANGHAI = ZoneInfo("Asia/Shanghai")
BASELINE_SHA = "0119ca98e4db9156ec1008b8155fa4342131943d"
MODEL_ID = "GEN2-LGBM-20D-SECTOR-BALANCED-TOP20"
CONTRACT_VERSION = "DAILY_PIT_FORWARD_EVIDENCE_V1"
PROTECTED_PREFIXES = (
    "research_v6/",
    "artifacts/research_v6/",
    "stockpilot/research_challenger/prospective_gen2.py",
    "stockpilot/research_challenger/prospective_gen2_runtime.py",
    "stockpilot/research_challenger/prospective_gen2_runtime_locked.py",
    "stockpilot/daily_pit/",
    "stockpilot/prospective_r4/",
    "stockpilot/prediction_forward",
    "artifacts/research_challenger/gen02/experiments/007_human_readjudication/",
)
CONTROL_COLUMNS = (
    "benchmark_weight_rank",
    "volatility_60_rank",
    "momentum",
    "liquidity",
)


class ForwardEvidenceError(RuntimeError):
    """Stable fail-closed forward-evidence error."""


@dataclass(frozen=True)
class ForwardEvidenceSettings:
    root: Path = Path("artifacts/forward_evidence/gen2")
    baseline_sha: str = BASELINE_SHA
    model_id: str = MODEL_ID
    runtime_settings: daily_runtime.DailyRuntimeSettings = field(
        default_factory=daily_runtime.DailyRuntimeSettings
    )
    top_ks: tuple[int, ...] = (10, 20, 30, 50)
    cost_bps: tuple[int, ...] = (0, 10, 20, 30, 50)
    checkpoints: tuple[int, ...] = (5, 10, 20, 40, 60)
    quantiles: int = 5
    bootstrap_block_length: int = 20
    bootstrap_replications: int = 1_000
    random_seed: int = 42
    earliest_prediction_time: time = time(18, 30)
    verify_git_boundary: bool = True

    @property
    def protocol_path(self) -> Path:
        return self.root / "protocol.json"

    @property
    def state_path(self) -> Path:
        return self.root / "forward_evidence_state.json"

    @property
    def prediction_root(self) -> Path:
        return self.root / "predictions"

    @property
    def settlement_root(self) -> Path:
        return self.root / "settlements"

    @property
    def attempt_root(self) -> Path:
        return self.root / "attempts"

    @property
    def checkpoint_root(self) -> Path:
        return self.root / "checkpoints"


def _utc(now: datetime) -> str:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(_jsonable(value)))
    os.replace(temporary, path)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, encoding="utf-8", errors="strict"
    ).strip()


def _changed_paths(baseline_sha: str) -> set[str]:
    commands = (
        ("diff", "--name-only", f"{baseline_sha}..HEAD"),
        ("diff", "--name-only"),
        ("diff", "--cached", "--name-only"),
        ("ls-files", "--others", "--exclude-standard"),
    )
    result: set[str] = set()
    for command in commands:
        result.update(line.strip().replace("\\", "/") for line in _git(*command).splitlines())
    return {name for name in result if name}


def _protected_changes(paths: set[str]) -> list[str]:
    return sorted(
        name
        for name in paths
        if any(name == prefix or name.startswith(prefix) for prefix in PROTECTED_PREFIXES)
    )


def _baseline_guard(settings: ForwardEvidenceSettings) -> dict[str, Any]:
    lock = daily_runtime.verify_effective_daily_runtime_freeze(settings.runtime_settings)
    if lock.get("effective_daily_input_lock_intact") is not True:
        raise ForwardEvidenceError(f"FORWARD_BASELINE_LOCK_INVALID:{lock.get('failures', [])}")
    changed: list[str] = []
    head = settings.baseline_sha
    if settings.verify_git_boundary:
        try:
            _git("cat-file", "-e", f"{settings.baseline_sha}^{{commit}}")
            head = _git("rev-parse", "HEAD")
            changed = _protected_changes(_changed_paths(settings.baseline_sha))
        except Exception as error:
            raise ForwardEvidenceError(f"FORWARD_BASELINE_GIT_CHECK_FAILED:{error}") from error
        if changed:
            raise ForwardEvidenceError(f"FORWARD_PROTECTED_SURFACE_CHANGED:{changed}")
    spec = read_verified_json(settings.runtime_settings.human_dir / "challenger_spec.json")
    if spec.get("model_id") != settings.model_id:
        raise ForwardEvidenceError("FORWARD_OBSERVATION_MODEL_CHANGED")
    return {
        "baseline_sha": settings.baseline_sha,
        "git_sha": head,
        "model_id": settings.model_id,
        "model_spec_hash": spec["model_spec_hash"],
        "feature_policy_hash": spec["feature_policy_hash"],
        "training_policy_hash": spec["training_policy_hash"],
        "portfolio_policy_hash": spec["portfolio_policy_hash"],
        "cost_policy_hash": spec["cost_policy_hash"],
        "effective_lock_identity": {
            "human_007": lock["human_007_lock_sha256"],
            "operational_008": lock["parent_008_lock_sha256"],
            "runtime_009": lock["runtime_009_lock_sha256"],
            "runtime_010": lock["self_verification_010_lock_sha256"],
            "daily_pit_011": lock["daily_011_lock_sha256"],
        },
        "protected_changes": changed,
        "lock_status": "VALID",
    }


def _protocol(settings: ForwardEvidenceSettings, baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "phase": "FORWARD_OBSERVATION",
        "baseline_sha": baseline["baseline_sha"],
        "observation_model": baseline["model_id"],
        "model_spec_hash": baseline["model_spec_hash"],
        "feature_policy_hash": baseline["feature_policy_hash"],
        "training_policy_hash": baseline["training_policy_hash"],
        "portfolio_policy_hash": baseline["portfolio_policy_hash"],
        "cost_policy_hash": baseline["cost_policy_hash"],
        "effective_lock_identity": baseline["effective_lock_identity"],
        "prospective_start_date": "2026-09-01",
        "horizon_trading_sessions": 20,
        "maturity_semantics": "T+1 open entry to T+21 open exit",
        "top_k_policy": {"production_research": 20, "diagnostic": list(settings.top_ks)},
        "quantile_policy": {
            "buckets": settings.quantiles,
            "method": "same-session score rank; Q1 lowest, Q5 highest",
        },
        "residual_policy": {
            "method": "same-session OLS residuals for score and realized return",
            "controls": ["sector", "size", "volatility", "momentum", "liquidity"],
            "beta": "NOT_EVALUABLE_FROM_FROZEN_DAILY_FEATURE_SCHEMA",
        },
        "regime_policy": {
            "market": "frozen risk_on/risk_off/neutral thresholds from prediction_forward",
            "volatility": "current cross-sectional 20D volatility versus prior 252-session median",
        },
        "cost_bps": list(settings.cost_bps),
        "checkpoints": list(settings.checkpoints),
        "historical_baseline_reference_only": {
            "rank_ic": 0.049877,
            "icir": 0.26535,
            "residual_ic": 0.02049,
        },
        "historical_optimization_allowed": False,
        "automatic_promotion_allowed": False,
        "research_only": True,
        "execution_authorized": False,
        "broker_requests_allowed": 0,
    }


def initialize(
    settings: ForwardEvidenceSettings | None = None,
    *,
    baseline_verifier: Callable[[ForwardEvidenceSettings], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    settings = settings or ForwardEvidenceSettings()
    baseline = (baseline_verifier or _baseline_guard)(settings)
    expected = _protocol(settings, baseline)
    if settings.protocol_path.is_file():
        actual = read_verified_json(settings.protocol_path)
        if actual != expected:
            raise ForwardEvidenceError("FORWARD_EVIDENCE_PROTOCOL_CHANGED")
        digest = verify_immutable(settings.protocol_path)
    else:
        digest = write_immutable_json(settings.protocol_path, expected)
    if not settings.state_path.is_file():
        state = _empty_state(settings, baseline)
        _atomic_json(settings.state_path, state)
    return {
        "status": "FORWARD_EVIDENCE_INITIALIZED",
        "protocol_sha256": digest,
        **baseline,
        "research_only": True,
        "execution_authorized": False,
        "broker_requests": 0,
    }


def _verify_chain(root: Path, artifact_name: str) -> list[Path]:
    paths = sorted(root.glob(f"*/{artifact_name}"))
    previous: str | None = None
    for path in paths:
        manifest_path = path.parent / "manifest.json"
        manifest = read_verified_json(manifest_path)
        digest = verify_immutable(path)
        if manifest.get(artifact_name) != digest:
            raise ForwardEvidenceError(f"FORWARD_REGISTRY_MANIFEST_MISMATCH:{path}")
        if manifest.get("previous_manifest_sha256") != previous:
            raise ForwardEvidenceError(f"FORWARD_REGISTRY_CHAIN_MISMATCH:{path}")
        previous = verify_immutable(manifest_path)
    return paths


def verify_forward_evidence(
    settings: ForwardEvidenceSettings | None = None,
    *,
    baseline_verifier: Callable[[ForwardEvidenceSettings], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    settings = settings or ForwardEvidenceSettings()
    initialized = initialize(settings, baseline_verifier=baseline_verifier)
    predictions = _verify_chain(settings.prediction_root, "prediction.json")
    settlements = _verify_chain(settings.settlement_root, "settlement.json")
    prediction_dates = {path.parent.name for path in predictions}
    orphaned = sorted(
        path.parent.name for path in settlements if path.parent.name not in prediction_dates
    )
    if orphaned:
        raise ForwardEvidenceError(f"FORWARD_SETTLEMENT_WITHOUT_PREDICTION:{orphaned}")
    return {
        **initialized,
        "integrity_status": "VALID",
        "prediction_records": len(predictions),
        "settlement_records": len(settlements),
    }


def _previous_manifest(root: Path, current_date: str) -> str | None:
    paths = sorted(path for path in root.glob("*/manifest.json") if path.parent.name < current_date)
    return verify_immutable(paths[-1]) if paths else None


def _core_prediction_paths(
    settings: ForwardEvidenceSettings, target_date: str
) -> tuple[Path, Path, Path]:
    directory = settings.runtime_settings.prediction_root / target_date
    return directory / "prediction.json", directory / "prediction.csv", directory / "manifest.json"


def _prediction_controls(settings: ForwardEvidenceSettings, target_date: str) -> pd.DataFrame:
    path = settings.runtime_settings.daily_input_root / target_date / "panel.parquet"
    if not path.is_file():
        raise ForwardEvidenceError(f"FORWARD_DAILY_PANEL_MISSING:{target_date}")
    columns = ["symbol", *CONTROL_COLUMNS]
    panel = pd.read_parquet(path, columns=columns)
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
    if panel["symbol"].duplicated().any():
        raise ForwardEvidenceError("FORWARD_CONTROL_PANEL_DUPLICATE")
    return panel


def _prediction_regime(settings: ForwardEvidenceSettings, target_date: str) -> dict[str, Any]:
    daily_dir = settings.runtime_settings.daily_input_root / target_date
    market = pd.read_csv(daily_dir / "market.csv", dtype={"symbol": str})
    frozen = pd.read_csv(
        settings.runtime_settings.pit_settings().frozen_market_path, dtype={"symbol": str}
    )
    combined = pd.concat([frozen, market], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
    combined["symbol"] = combined["symbol"].astype(str).str.zfill(6)
    combined = combined[combined["date"].le(pd.Timestamp(target_date))]
    combined = combined.sort_values(["symbol", "date"]).drop_duplicates(
        ["date", "symbol"], keep="last"
    )
    grouped = combined.groupby("symbol", sort=False)["close"]
    combined["ret_20"] = grouped.pct_change(20, fill_method=None)
    combined["momentum_60"] = grouped.pct_change(60, fill_method=None)
    daily_return = grouped.pct_change(fill_method=None)
    combined["volatility_20"] = daily_return.groupby(combined["symbol"]).transform(
        lambda values: values.rolling(20, min_periods=20).std() * math.sqrt(252)
    )
    current = combined[combined["date"].eq(pd.Timestamp(target_date))].copy()
    eligible = set(_prediction_controls(settings, target_date)["symbol"])
    current = current[current["symbol"].isin(eligible)]
    momentum = float(current["momentum_60"].mean())
    breadth = float(current["ret_20"].gt(0).mean())
    if momentum > 0.02 and breadth > 0.55:
        regime = "risk_on"
    elif momentum < -0.02 and breadth < 0.45:
        regime = "risk_off"
    else:
        regime = "neutral"
    date_vol = (
        combined[combined["date"].lt(pd.Timestamp(target_date))]
        .groupby("date")["volatility_20"]
        .mean()
        .dropna()
        .tail(252)
    )
    current_vol = float(current["volatility_20"].mean())
    prior_median = float(date_vol.median()) if len(date_vol) >= 60 else float("nan")
    vol_state = (
        "insufficient_history"
        if not np.isfinite(prior_median)
        else ("high_vol" if current_vol > prior_median else "low_vol")
    )
    return _jsonable(
        {
            "market_regime": regime,
            "market_momentum_60": momentum,
            "positive_20d_breadth": breadth,
            "volatility_regime": vol_state,
            "current_mean_volatility_20": current_vol,
            "prior_252_session_volatility_median": prior_median,
            "classified_from_prediction_time_data_only": True,
        }
    )


def register_prediction(
    target_date: str,
    settings: ForwardEvidenceSettings,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    destination = settings.prediction_root / target_date
    existing = destination / "prediction.json"
    core_json, core_csv, core_manifest_path = _core_prediction_paths(settings, target_date)
    core_manifest_hash = verify_immutable(core_manifest_path)
    if existing.is_file():
        value = read_verified_json(existing)
        if value.get("core_prediction_manifest_sha256") != core_manifest_hash:
            raise ForwardEvidenceError("DUPLICATE_PREDICTION_CONFLICT")
        return value | {"idempotent": True}
    core = read_verified_json(core_json)
    core_manifest = read_verified_json(core_manifest_path)
    if verify_immutable(core_csv) != core_manifest.get("prediction.csv"):
        raise ForwardEvidenceError("CORE_PREDICTION_MANIFEST_INVALID")
    frame = pd.read_csv(core_csv, dtype={"symbol": str})
    forbidden = [
        name
        for name in frame
        if any(token in name.lower() for token in ("actual_", "future_", "realized_"))
    ]
    if forbidden or core.get("future_label_fields_present") is not False:
        raise ForwardEvidenceError(f"PRE_MATURITY_OUTCOME_PRESENT:{forbidden}")
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    controls = _prediction_controls(settings, target_date)
    frame = frame.merge(controls, on="symbol", how="left", validate="one_to_one")
    if frame[list(CONTROL_COLUMNS)].isna().any().any():
        raise ForwardEvidenceError("FORWARD_MONITOR_CONTROL_MISSING")
    frame = frame.sort_values(["rank", "symbol"], kind="mergesort").reset_index(drop=True)
    score_hash = sha256_bytes(canonical_frame_bytes(frame[["symbol", "score"]], ["symbol"]))
    ranking_hash = sha256_bytes(canonical_frame_bytes(frame[["symbol", "rank"]], ["symbol"]))
    universe_hash = sha256_bytes(canonical_json_bytes(sorted(frame["symbol"].tolist())))
    model_signature = core.get("training_evidence", {}).get("model_signature")
    model_hash = sha256_bytes(canonical_json_bytes(model_signature))
    daily_manifest = settings.runtime_settings.daily_input_root / target_date / "manifest.json"
    daily_manifest_hash = verify_immutable(daily_manifest)
    acquisition_path = (
        settings.runtime_settings.daily_input_root / target_date / "source_receipt.json"
    )
    acquisition = read_verified_json(acquisition_path)
    if acquisition.get("target_date") != target_date:
        raise ForwardEvidenceError("FORWARD_PROVIDER_RECEIPT_TARGET_MISMATCH")
    rows = frame[
        [
            "symbol",
            "score",
            "rank",
            "industry",
            "broad_sector",
            "benchmark_weight",
            "selected_for_new_portfolio",
            "portfolio_weight",
            *CONTROL_COLUMNS,
        ]
    ].to_dict("records")
    top = {f"top{k}": frame.nsmallest(k, "rank")["symbol"].tolist() for k in settings.top_ks}
    prediction_id = f"GEN2-{target_date}-{core_manifest_hash[:16]}"
    value = _jsonable(
        {
            "contract_version": CONTRACT_VERSION,
            "prediction_id": prediction_id,
            "prediction_date": target_date,
            "target_date": target_date,
            "decision_timestamp": core["created_at_utc"],
            "pit_cutoff": f"{target_date}T18:30:00+08:00",
            "git_sha": baseline["git_sha"],
            "baseline_sha": baseline["baseline_sha"],
            "model_id": core["model_id"],
            "model_hash": model_hash,
            "model_spec_hash": core["model_spec_hash"],
            "effective_lock_identity": baseline["effective_lock_identity"],
            "feature_manifest_hash": daily_manifest_hash,
            "seal_hash": core["input_seal_sha256"],
            "universe_hash": universe_hash,
            "score_hash": score_hash,
            "ranking_hash": ranking_hash,
            "candidate_definition": "raw_score_rank_le_20_observational",
            "candidate_list": top["top20"],
            "top_lists": top,
            "scores_and_ranks": rows,
            "decision": core["portfolio_action"],
            "is_rebalance_day": core["is_rebalance_day"],
            "maturity_date": core["label_maturity_date"],
            "maturity_session": core["label_maturity_date"],
            "expected_settlement_eligibility_timestamp": (
                f"{core['label_maturity_date']}T18:30:00+08:00"
            ),
            "maturity_status": "PENDING_MATURITY",
            "prediction_time_regime": _prediction_regime(settings, target_date),
            "provider_evidence": {
                "provider": acquisition.get("provider_sources"),
                "acquisition_timestamp": acquisition.get("acquired_at_utc"),
                "target_date": acquisition.get("target_date"),
                "effective_timestamp": f"{target_date}T15:00:00+08:00",
                "manifest_hash": verify_immutable(
                    settings.runtime_settings.daily_input_root
                    / target_date
                    / "market_manifest.json"
                ),
                "row_count": acquisition.get("target_rows"),
                "cutoff": acquisition.get("request_end_date"),
                "pit_validation_status": "VALID",
                "provider_request_count": acquisition.get("provider_request_count"),
            },
            "core_prediction_manifest_sha256": core_manifest_hash,
            "research_only": True,
            "execution_authorized": False,
            "portfolio_mutation": False,
            "real_orders": False,
            "real_trades": False,
            "broker_requests": 0,
            "promotion": False,
        }
    )
    digest = write_immutable_json(existing, value)
    previous = _previous_manifest(settings.prediction_root, target_date)
    write_immutable_json(
        destination / "manifest.json",
        {
            "prediction.json": digest,
            "core_prediction_manifest_sha256": core_manifest_hash,
            "previous_manifest_sha256": previous,
        },
    )
    return value | {"idempotent": False}


def _residual_ic(frame: pd.DataFrame) -> float | None:
    columns = ["score", "actual_return_20d", "broad_sector", *CONTROL_COLUMNS]
    clean = frame.dropna(subset=columns).copy()
    if len(clean) < 40:
        return None
    sector = pd.get_dummies(clean["broad_sector"].astype(str), drop_first=True, dtype=float)
    controls = np.column_stack(
        [
            np.ones(len(clean)),
            clean[list(CONTROL_COLUMNS)].to_numpy(dtype=float),
            sector.to_numpy(dtype=float),
        ]
    )
    score = clean["score"].to_numpy(dtype=float)
    realized = clean["actual_return_20d"].to_numpy(dtype=float)
    score_residual = score - controls @ np.linalg.lstsq(controls, score, rcond=None)[0]
    return_residual = realized - controls @ np.linalg.lstsq(controls, realized, rcond=None)[0]
    value = pd.Series(score_residual).corr(pd.Series(return_residual), method="spearman")
    return None if pd.isna(value) else float(value)


def _slice_ic(frame: pd.DataFrame, mask: pd.Series) -> float | None:
    part = frame[mask].dropna(subset=["score", "actual_return_20d"])
    if len(part) < 20 or part["score"].nunique() < 2:
        return None
    value = part["score"].corr(part["actual_return_20d"], method="spearman")
    return None if pd.isna(value) else float(value)


def _turnover(frame: pd.DataFrame, previous: dict[str, Any] | None, k: int) -> dict[str, Any]:
    current = set(frame.nsmallest(k, "rank")["symbol"])
    if previous is None:
        return {"name_retention": None, "total_turnover": 1.0}
    previous_names = set(previous["top_lists"][f"top{k}"])
    retention = len(current & previous_names) / k
    return {"name_retention": retention, "total_turnover": 2.0 * (1.0 - retention)}


def register_settlement(
    prediction_date: str,
    settings: ForwardEvidenceSettings,
) -> dict[str, Any]:
    destination = settings.settlement_root / prediction_date
    existing = destination / "settlement.json"
    core_dir = settings.runtime_settings.settlement_root / prediction_date
    core_json = core_dir / "settlement.json"
    core_csv = core_dir / "settlement.csv"
    core_hash = verify_immutable(core_json)
    if existing.is_file():
        value = read_verified_json(existing)
        if value.get("core_settlement_sha256") != core_hash:
            raise ForwardEvidenceError("DUPLICATE_SETTLEMENT_CONFLICT")
        return value | {"idempotent": True}
    prediction = read_verified_json(settings.prediction_root / prediction_date / "prediction.json")
    core = read_verified_json(core_json)
    if verify_immutable(core_csv) != core.get("settlement_csv_sha256"):
        raise ForwardEvidenceError("CORE_SETTLEMENT_MANIFEST_INVALID")
    outcomes = pd.read_csv(core_csv, dtype={"symbol": str})
    outcomes["symbol"] = outcomes["symbol"].astype(str).str.zfill(6)
    frozen = pd.DataFrame(prediction["scores_and_ranks"])
    frozen["symbol"] = frozen["symbol"].astype(str).str.zfill(6)
    realized = outcomes[["symbol", "actual_return_20d", "settled"]].merge(
        frozen, on="symbol", how="inner", validate="one_to_one"
    )
    realized = realized[realized["settled"].astype(bool)].copy()
    if len(realized) < 20:
        raise ForwardEvidenceError("FORWARD_SETTLEMENT_COVERAGE_INSUFFICIENT")
    realized["realized_rank"] = (
        realized["actual_return_20d"].rank(ascending=False, method="min").astype(int)
    )
    pearson = realized["score"].corr(realized["actual_return_20d"], method="pearson")
    spearman = realized["score"].corr(realized["actual_return_20d"], method="spearman")
    realized["quantile"] = (
        pd.qcut(
            realized["score"].rank(method="first"),
            settings.quantiles,
            labels=False,
        )
        + 1
    )
    quantile = realized.groupby("quantile")["actual_return_20d"].mean().to_dict()
    proxy = core.get("research_proxy_return")
    prior_paths = sorted(
        path
        for path in settings.prediction_root.glob("*/prediction.json")
        if path.parent.name < prediction_date
    )
    previous = read_verified_json(prior_paths[-1]) if prior_paths else None
    top_k: dict[str, Any] = {}
    for k in settings.top_ks:
        part = realized.nsmallest(k, "rank")
        gross = float(part["actual_return_20d"].mean())
        activity = _turnover(realized, previous, k)
        excess = None if proxy is None else gross - float(proxy)
        top_k[f"top{k}"] = {
            "gross_return": gross,
            "research_proxy_return": proxy,
            "gross_proxy_alpha": excess,
            **activity,
            "cost_sensitivity": {
                str(bps): None
                if excess is None
                else excess - float(activity["total_turnover"]) * bps / 10_000
                for bps in settings.cost_bps
            },
        }
    cap_rank = realized["benchmark_weight"].rank(pct=True)
    vol_rank = realized["volatility_60_rank"].rank(pct=True)
    regime = prediction["prediction_time_regime"]
    weak = {
        "2025_like_deterioration": bool(float(spearman) <= 0.001249),
        "risk_off": float(spearman) if regime["market_regime"] == "risk_off" else None,
        "technology": _slice_ic(realized, realized["broad_sector"].astype(str).eq("technology")),
        "large_cap": _slice_ic(realized, cap_rank.ge(2 / 3)),
        "low_volatility": _slice_ic(realized, vol_rank.le(1 / 3)),
    }
    rows = (
        realized[["symbol", "actual_return_20d", "realized_rank", "rank", "score", "quantile"]]
        .sort_values("symbol")
        .to_dict("records")
    )
    value = _jsonable(
        {
            "contract_version": CONTRACT_VERSION,
            "settlement_id": f"SETTLEMENT-{prediction['prediction_id']}",
            "prediction_id": prediction["prediction_id"],
            "prediction_date": prediction_date,
            "maturity_date": prediction["maturity_date"],
            "settlement_timestamp": datetime.now(timezone.utc).isoformat(),
            "realized_20d_returns": rows,
            "pearson_ic": pearson,
            "spearman_rank_ic": spearman,
            "residual_rank_ic": _residual_ic(realized),
            "quantile_returns": {f"Q{int(key)}": value for key, value in quantile.items()},
            "q5_minus_q1": quantile.get(5, np.nan) - quantile.get(1, np.nan),
            "adjacent_bucket_consistency": sum(
                quantile.get(index + 1, -np.inf) >= quantile.get(index, np.inf)
                for index in range(1, settings.quantiles)
            )
            / (settings.quantiles - 1),
            "top_k": top_k,
            "candidate_outcome": top_k["top20"],
            "prediction_time_regime": regime,
            "weak_area_tracking": weak,
            "settlement_witness_hash": core.get("market_witness_sha256"),
            "market_source_hash": core.get("market_source_sha256"),
            "core_settlement_sha256": core_hash,
            "prediction_artifact_mutated": False,
            "research_only": True,
            "execution_authorized": False,
            "broker_requests": 0,
            "promotion": False,
        }
    )
    digest = write_immutable_json(existing, value)
    previous_manifest = _previous_manifest(settings.settlement_root, prediction_date)
    write_immutable_json(
        destination / "manifest.json",
        {
            "settlement.json": digest,
            "core_settlement_sha256": core_hash,
            "previous_manifest_sha256": previous_manifest,
        },
    )
    return value | {"idempotent": False}


def _bootstrap(values: list[float], settings: ForwardEvidenceSettings) -> dict[str, Any]:
    clean = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    block = settings.bootstrap_block_length
    if len(clean) < block * 2:
        return {"samples": len(clean), "ci_lower": None, "ci_upper": None}
    rng = np.random.default_rng(settings.random_seed)
    starts = np.arange(len(clean) - block + 1)
    blocks = math.ceil(len(clean) / block)
    estimates = []
    for _ in range(settings.bootstrap_replications):
        selected = rng.choice(starts, size=blocks, replace=True)
        sample = np.concatenate([clean[start : start + block] for start in selected])[: len(clean)]
        estimates.append(float(sample.mean()))
    return {
        "samples": len(clean),
        "block_length": block,
        "replications": settings.bootstrap_replications,
        "ci_lower": float(np.quantile(estimates, 0.025)),
        "ci_upper": float(np.quantile(estimates, 0.975)),
    }


def _metrics(
    settlements: list[dict[str, Any]], settings: ForwardEvidenceSettings
) -> dict[str, Any]:
    ranks = pd.Series([item["spearman_rank_ic"] for item in settlements], dtype=float).dropna()
    pearsons = pd.Series([item["pearson_ic"] for item in settlements], dtype=float).dropna()
    residuals = pd.Series(
        [item.get("residual_rank_ic") for item in settlements], dtype=float
    ).dropna()
    std = float(ranks.std(ddof=1)) if len(ranks) > 1 else float("nan")
    quantile_means = {
        f"Q{bucket}": float(
            pd.Series(
                [item["quantile_returns"].get(f"Q{bucket}") for item in settlements],
                dtype=float,
            ).mean()
        )
        for bucket in range(1, settings.quantiles + 1)
    }
    monotonicity = (
        sum(
            quantile_means[f"Q{bucket + 1}"] >= quantile_means[f"Q{bucket}"]
            for bucket in range(1, settings.quantiles)
        )
        / (settings.quantiles - 1)
        if settlements
        else None
    )
    top_k: dict[str, Any] = {}
    for k in settings.top_ks:
        key = f"top{k}"
        entries = [item["top_k"][key] for item in settlements]
        top_k[key] = {
            "gross_return": _mean([item["gross_return"] for item in entries]),
            "gross_proxy_alpha": _mean([item["gross_proxy_alpha"] for item in entries]),
            "name_retention": _mean([item["name_retention"] for item in entries]),
            "rank_persistence": _rank_persistence(settings, k),
            "cost_sensitivity": {
                str(bps): _mean([item["cost_sensitivity"][str(bps)] for item in entries])
                for bps in settings.cost_bps
            },
        }
    regimes: dict[str, list[float]] = {}
    for item in settlements:
        labels = item["prediction_time_regime"]
        for label in (labels["market_regime"], labels["volatility_regime"]):
            regimes.setdefault(label, []).append(item["spearman_rank_ic"])
    regime_metrics = {name: _mean(values) for name, values in regimes.items()}
    weakest = min(regime_metrics, key=regime_metrics.get) if regime_metrics else None
    return _jsonable(
        {
            "settled_sessions": len(settlements),
            "pearson_ic": float(pearsons.mean()) if len(pearsons) else None,
            "rank_ic": float(ranks.mean()) if len(ranks) else None,
            "median_rank_ic": float(ranks.median()) if len(ranks) else None,
            "rank_ic_std": std,
            "icir": float(ranks.mean() / std) if np.isfinite(std) and std > 0 else None,
            "positive_ic_ratio": float((ranks > 0).mean()) if len(ranks) else None,
            "residual_ic": float(residuals.mean()) if len(residuals) else None,
            "rank_ic_block_bootstrap": _bootstrap(ranks.tolist(), settings),
            "quantile_mean_returns": quantile_means if settlements else {},
            "quantile_monotonicity": monotonicity,
            "q5_minus_q1": (quantile_means["Q5"] - quantile_means["Q1"] if settlements else None),
            "top_k": top_k,
            "regime_rank_ic": regime_metrics,
            "weakest_observed_regime": weakest,
        }
    )


def _mean(values: list[Any]) -> float | None:
    clean = pd.to_numeric(pd.Series(values, dtype=object), errors="coerce").dropna()
    return float(clean.mean()) if len(clean) else None


def _rank_persistence(settings: ForwardEvidenceSettings, k: int) -> float | None:
    predictions = [
        read_verified_json(path)
        for path in sorted(settings.prediction_root.glob("*/prediction.json"))
    ]
    values: list[float] = []
    for left, right in pairwise(predictions):
        first = pd.DataFrame(left["scores_and_ranks"])[["symbol", "rank"]].rename(
            columns={"rank": "left"}
        )
        second = pd.DataFrame(right["scores_and_ranks"])[["symbol", "rank"]].rename(
            columns={"rank": "right"}
        )
        joined = first.merge(second, on="symbol")
        joined = joined[(joined["left"] <= k) | (joined["right"] <= k)]
        if len(joined) >= 3:
            values.append(float(joined["left"].corr(joined["right"], method="spearman")))
    return _mean(values)


def _empty_state(settings: ForwardEvidenceSettings, baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "baseline_sha": baseline["baseline_sha"],
        "observation_model": baseline["model_id"],
        "observation_start_date": "2026-09-01",
        "total_prediction_sessions": 0,
        "pending_predictions": 0,
        "matured_predictions": 0,
        "failed_or_blocked_sessions": 0,
        "latest_prediction": None,
        "latest_settlement": None,
        "cumulative_rank_ic": None,
        "cumulative_icir": None,
        "positive_ic_ratio": None,
        "residual_ic": None,
        "top20_gross_proxy": None,
        "top20_20bps_proxy": None,
        "quantile_monotonicity": None,
        "weakest_observed_regime": None,
        "integrity_status": "VALID",
        "status": "EVIDENCE_ACCUMULATING",
        "next_scheduled_action": "WAIT_FOR_NEXT_ELIGIBLE_DAILY_PIT_SESSION",
        "research_only": True,
        "execution_authorized": False,
        "broker_requests": 0,
    }


def build_state(
    settings: ForwardEvidenceSettings | None = None,
    *,
    status: str | None = None,
    next_action: str | None = None,
    integrity_status: str = "VALID",
) -> dict[str, Any]:
    settings = settings or ForwardEvidenceSettings()
    predictions = [
        read_verified_json(path)
        for path in sorted(settings.prediction_root.glob("*/prediction.json"))
    ]
    settlements = [
        read_verified_json(path)
        for path in sorted(settings.settlement_root.glob("*/settlement.json"))
    ]
    blocked_sessions = {path.parent.name for path in settings.attempt_root.glob("*/*.json")}
    settled_dates = {item["prediction_date"] for item in settlements}
    pending = [item for item in predictions if item["prediction_date"] not in settled_dates]
    metrics = _metrics(settlements, settings)
    top20 = metrics.get("top_k", {}).get("top20", {})
    state = {
        "contract_version": CONTRACT_VERSION,
        "baseline_sha": settings.baseline_sha,
        "observation_model": settings.model_id,
        "observation_start_date": "2026-09-01",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_prediction_sessions": len(predictions),
        "pending_predictions": len(pending),
        "matured_predictions": len(settlements),
        "failed_or_blocked_sessions": len(blocked_sessions),
        "latest_prediction": predictions[-1]["prediction_date"] if predictions else None,
        "latest_settlement": settlements[-1]["prediction_date"] if settlements else None,
        "cumulative_rank_ic": metrics["rank_ic"],
        "cumulative_icir": metrics["icir"],
        "positive_ic_ratio": metrics["positive_ic_ratio"],
        "residual_ic": metrics["residual_ic"],
        "top20_gross_proxy": top20.get("gross_proxy_alpha"),
        "top20_20bps_proxy": top20.get("cost_sensitivity", {}).get("20"),
        "quantile_monotonicity": metrics["quantile_monotonicity"],
        "weakest_observed_regime": metrics["weakest_observed_regime"],
        "forward_metrics": metrics,
        "integrity_status": integrity_status,
        "status": status
        or ("WAITING_FOR_MATURITY" if pending and not settlements else "EVIDENCE_ACCUMULATING"),
        "next_scheduled_action": next_action or "WAIT_FOR_NEXT_ELIGIBLE_DAILY_PIT_SESSION",
        "research_only": True,
        "execution_authorized": False,
        "production_execution": False,
        "broker_requests": 0,
        "promotion": False,
    }
    _atomic_json(settings.state_path, state)
    return _jsonable(state)


def _attempt(
    target_date: str,
    reason: str,
    now: datetime,
    settings: ForwardEvidenceSettings,
    *,
    provider_requests: int = 0,
) -> dict[str, Any]:
    attempt_id = _utc(now).replace(":", "").replace("+", "_")
    value = {
        "attempt_id": attempt_id,
        "target_date": target_date,
        "decision": "NO_FORWARD_PREDICTION",
        "reason": reason,
        "checked_at_utc": _utc(now),
        "provider_requests": provider_requests,
        "research_only": True,
        "execution_authorized": False,
    }
    write_immutable_json(settings.attempt_root / target_date / f"{attempt_id}.json", value)
    return value


def _session_readiness(
    target_date: str, now: datetime, settings: ForwardEvidenceSettings
) -> str | None:
    sessions = load_verified_calendar(settings.runtime_settings.calendar_path).sessions()
    if pd.Timestamp(target_date) not in sessions:
        return "NOT_VERIFIED_TRADING_SESSION"
    local = now.astimezone(SHANGHAI)
    if target_date != local.date().isoformat():
        return "TARGET_DATE_MUST_BE_TODAY"
    if local.timetz().replace(tzinfo=None) < settings.earliest_prediction_time:
        return "DATA_WINDOW_NOT_OPEN"
    return None


def _sync_core(settings: ForwardEvidenceSettings, baseline: dict[str, Any]) -> tuple[int, int]:
    new_predictions = 0
    new_settlements = 0
    for path in sorted(settings.runtime_settings.prediction_root.glob("*/prediction.json")):
        result = register_prediction(path.parent.name, settings, baseline)
        new_predictions += int(not result.get("idempotent", False))
    for path in sorted(settings.runtime_settings.settlement_root.glob("*/settlement.json")):
        result = register_settlement(path.parent.name, settings)
        new_settlements += int(not result.get("idempotent", False))
    return new_predictions, new_settlements


def _settle_matured(
    now: datetime,
    market_path: Path | None,
    settings: ForwardEvidenceSettings,
) -> int:
    if market_path is None:
        return 0
    verify_immutable(market_path)
    verify_immutable(market_path.with_suffix(market_path.suffix + ".witness.json"))
    count = 0
    actual = now.astimezone(SHANGHAI).date().isoformat()
    for path in sorted(settings.runtime_settings.prediction_root.glob("*/prediction.json")):
        prediction = read_verified_json(path)
        date = prediction["prediction_date"]
        if (settings.runtime_settings.settlement_root / date / "settlement.json").is_file():
            continue
        if prediction["label_maturity_date"] > actual:
            continue
        runtime010.settle_prediction(
            date,
            market_path,
            now=now,
            settings=settings.runtime_settings,
        )
        count += 1
    return count


def _checkpoint(state: dict[str, Any], settings: ForwardEvidenceSettings) -> bool:
    count = state["matured_predictions"]
    if count not in settings.checkpoints:
        return False
    path = settings.checkpoint_root / f"matured_{count:03d}.json"
    if path.is_file():
        verify_immutable(path)
        return False
    write_immutable_json(
        path,
        {
            "checkpoint": count,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "forward_metrics": state["forward_metrics"],
            "historical_data_merged": False,
            "tuning_performed": False,
            "automatic_promotion_allowed": False,
        },
    )
    return True


def run_daily(
    target_date: str,
    *,
    confirm_real_provider_acquisition: bool = False,
    settlement_market: Path | None = None,
    now: datetime | None = None,
    settings: ForwardEvidenceSettings | None = None,
    baseline_verifier: Callable[[ForwardEvidenceSettings], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    settings = settings or ForwardEvidenceSettings()
    now = now or datetime.now(timezone.utc)
    initialized = initialize(settings, baseline_verifier=baseline_verifier)
    baseline = {
        key: initialized[key]
        for key in (
            "baseline_sha",
            "git_sha",
            "model_id",
            "model_spec_hash",
            "feature_policy_hash",
            "training_policy_hash",
            "portfolio_policy_hash",
            "cost_policy_hash",
            "effective_lock_identity",
            "protected_changes",
            "lock_status",
        )
    }
    verify_forward_evidence(settings, baseline_verifier=lambda _: baseline)
    before_predictions = len(list(settings.prediction_root.glob("*/prediction.json")))
    before_settlements = len(list(settings.settlement_root.glob("*/settlement.json")))
    provider_requests = 0
    new_prediction_id: str | None = None
    reason = _session_readiness(target_date, now, settings)
    core_json, _, _ = _core_prediction_paths(settings, target_date)
    try:
        if reason is None and not core_json.is_file():
            daily_dir = settings.runtime_settings.daily_input_root / target_date
            if not (daily_dir / "market_manifest.json").is_file():
                if not confirm_real_provider_acquisition:
                    reason = "REAL_PROVIDER_CONFIRMATION_REQUIRED"
                else:
                    acquired = daily_pipeline.acquire_market(
                        target_date,
                        [],
                        now=now,
                        settings=settings.runtime_settings.pit_settings(),
                    )
                    provider_requests += int(acquired.get("provider_requests_made", 0))
            if reason is None and not (daily_dir / "manifest.json").is_file():
                daily_pipeline.materialize_features(
                    target_date, settings=settings.runtime_settings.pit_settings()
                )
            if (
                reason is None
                and not (
                    settings.runtime_settings.input_seal_root / f"{target_date}.json"
                ).is_file()
            ):
                daily_runtime.seal_inputs(target_date, now=now, settings=settings.runtime_settings)
            if reason is None:
                gate = daily_runtime.preflight(
                    target_date, now=now, settings=settings.runtime_settings
                )
                if gate.get("daily_prediction_allowed") is not True:
                    reason = f"PREFLIGHT:{gate.get('failures', [])}"
                else:
                    daily_runtime.generate_prediction(
                        target_date, now=now, settings=settings.runtime_settings
                    )
        if reason is not None and not core_json.is_file():
            _attempt(
                target_date,
                reason,
                now,
                settings,
                provider_requests=provider_requests,
            )
        _settle_matured(now, settlement_market, settings)
        _sync_core(settings, baseline)
        if (settings.prediction_root / target_date / "prediction.json").is_file():
            new_prediction_id = read_verified_json(
                settings.prediction_root / target_date / "prediction.json"
            )["prediction_id"]
    except Exception as error:  # noqa: BLE001 - outer operational boundary must fail closed
        text = f"{type(error).__name__}:{error}"
        try:
            _attempt(target_date, text, now, settings, provider_requests=provider_requests)
        except FileExistsError:
            pass
        invalid_tokens = (
            "PIT_",
            "FUTURE",
            "MUTAT",
            "LOCK_INVALID",
            "PROTECTED_SURFACE_CHANGED",
            "DUPLICATE",
            "WITNESS",
            "BROKER",
            "EXECUTION",
        )
        invalid = any(token in text.upper() for token in invalid_tokens)
        state = build_state(
            settings,
            status="FORWARD_EVIDENCE_INVALID" if invalid else "FORWARD_EVIDENCE_BLOCKED",
            next_action=text,
            integrity_status="INVALID" if invalid else "BLOCKED",
        )
        return _daily_report(
            state,
            target_date,
            new_prediction_id=None,
            newly_matured=0,
            pit_status="INVALID" if invalid else "BLOCKED",
            lock_status=baseline["lock_status"],
            provider_requests=provider_requests,
            reason=text,
            settings=settings,
        )
    new_predictions = (
        len(list(settings.prediction_root.glob("*/prediction.json"))) - before_predictions
    )
    new_settlements = (
        len(list(settings.settlement_root.glob("*/settlement.json"))) - before_settlements
    )
    next_action = (
        f"RETRY_AFTER_{target_date}T18:30:00+08:00"
        if reason == "DATA_WINDOW_NOT_OPEN"
        else (
            "RUN_WITH_EXPLICIT_REAL_PROVIDER_CONFIRMATION"
            if reason == "REAL_PROVIDER_CONFIRMATION_REQUIRED"
            else "WAIT_FOR_NEXT_MATURITY_OR_ELIGIBLE_SESSION"
        )
    )
    state = build_state(
        settings,
        status=(
            "EVIDENCE_ACCUMULATING"
            if new_predictions or not state_has_pending(settings)
            else "WAITING_FOR_MATURITY"
        ),
        next_action=next_action,
    )
    if _checkpoint(state, settings):
        state = build_state(
            settings,
            status="FORWARD_EVIDENCE_CHECKPOINT_REACHED",
            next_action="CONTINUE_OBSERVATION_WITHOUT_TUNING",
        )
    return _daily_report(
        state,
        target_date,
        new_prediction_id=new_prediction_id if new_predictions else None,
        newly_matured=new_settlements,
        pit_status="VALID" if new_predictions else f"NO_FORWARD_PREDICTION:{reason or 'DUPLICATE'}",
        lock_status=baseline["lock_status"],
        provider_requests=provider_requests,
        reason=reason,
        settings=settings,
    )


def state_has_pending(settings: ForwardEvidenceSettings) -> bool:
    predictions = len(list(settings.prediction_root.glob("*/prediction.json")))
    settlements = len(list(settings.settlement_root.glob("*/settlement.json")))
    return predictions > settlements


def _daily_report(
    state: dict[str, Any],
    target_date: str,
    *,
    new_prediction_id: str | None,
    newly_matured: int,
    pit_status: str,
    lock_status: str,
    provider_requests: int,
    reason: str | None,
    settings: ForwardEvidenceSettings,
) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return {
        "report": "STOCKPILOT_FORWARD_EVIDENCE_STATUS",
        "git_sha": _git("rev-parse", "HEAD")
        if settings.verify_git_boundary
        else settings.baseline_sha,
        "model": settings.model_id,
        "model_hash": read_verified_json(settings.protocol_path)["model_spec_hash"],
        "date_checked": target_date,
        "new_prediction_generated": new_prediction_id is not None,
        "prediction_id": new_prediction_id,
        "pending_maturity_count": state["pending_predictions"],
        "newly_matured_count": newly_matured,
        "total_matured_sessions": state["matured_predictions"],
        "current_forward_rank_ic": state["cumulative_rank_ic"],
        "current_icir": state["cumulative_icir"],
        "positive_ic_ratio": state["positive_ic_ratio"],
        "residual_ic": state["residual_ic"],
        "top20_gross_proxy": state["top20_gross_proxy"],
        "top20_20bps_proxy": state["top20_20bps_proxy"],
        "q1_to_q5_monotonicity": state["quantile_monotonicity"],
        "weakest_regime": state["weakest_observed_regime"],
        "pit_status": pit_status,
        "lock_status": lock_status,
        "production_execution": False,
        "provider_requests": provider_requests,
        "broker_requests": 0,
        "working_tree": "CLEAN" if not status else "INFRASTRUCTURE_CHANGES_PRESENT",
        "next_eligible_action": state["next_scheduled_action"],
        "reason": reason,
        "recommendations": {
            "P0": state["next_scheduled_action"],
            "P1": "SETTLE_ONLY_AFTER_20_SESSION_MATURITY_WITH_VALIDATED_WITNESS",
            "P2": "NEW_DATA_SOURCE_RESEARCH_RECOMMENDATION_ONLY_NO_AUTO_INTEGRATION",
        },
        "final_state": state["status"],
    }


def _format_report(value: dict[str, Any]) -> str:
    fields = (
        ("Git SHA", "git_sha"),
        ("Model", "model"),
        ("Model hash", "model_hash"),
        ("Date checked", "date_checked"),
        ("New prediction generated", "new_prediction_generated"),
        ("Prediction ID", "prediction_id"),
        ("Pending maturity count", "pending_maturity_count"),
        ("Newly matured count", "newly_matured_count"),
        ("Total matured sessions", "total_matured_sessions"),
        ("Current forward Rank IC", "current_forward_rank_ic"),
        ("Current ICIR", "current_icir"),
        ("Positive IC ratio", "positive_ic_ratio"),
        ("Residual IC", "residual_ic"),
        ("Top20 gross proxy", "top20_gross_proxy"),
        ("Top20 20bps proxy", "top20_20bps_proxy"),
        ("Q1→Q5 monotonicity", "q1_to_q5_monotonicity"),
        ("Weakest regime", "weakest_regime"),
        ("PIT status", "pit_status"),
        ("Lock status", "lock_status"),
        ("Production execution", "production_execution"),
        ("Provider requests", "provider_requests"),
        ("Broker requests", "broker_requests"),
        ("Working tree", "working_tree"),
        ("Next eligible action", "next_eligible_action"),
    )
    lines = ["# STOCKPILOT_FORWARD_EVIDENCE_STATUS", ""]
    lines.extend(f"* {label}: {value.get(key)}" for label, key in fields)
    lines.extend(
        [
            "",
            "## P0",
            str(value["recommendations"]["P0"]),
            "",
            "## P1",
            str(value["recommendations"]["P1"]),
            "",
            "## P2",
            str(value["recommendations"]["P2"]),
            "",
            "Final state:",
            str(value["final_state"]),
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frozen Gen2 forward-evidence monitor")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("initialize")
    commands.add_parser("verify")
    commands.add_parser("status")
    run = commands.add_parser("run")
    run.add_argument("--date")
    run.add_argument("--confirm-real-provider-acquisition", action="store_true")
    run.add_argument("--settlement-market", type=Path)
    run.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    settings = ForwardEvidenceSettings()
    if args.command == "initialize":
        result = initialize(settings)
    elif args.command == "verify":
        result = verify_forward_evidence(settings)
    elif args.command == "status":
        verify_forward_evidence(settings)
        result = build_state(settings)
    else:
        now = datetime.now(timezone.utc)
        date = args.date or now.astimezone(SHANGHAI).date().isoformat()
        result = run_daily(
            date,
            confirm_real_provider_acquisition=args.confirm_real_provider_acquisition,
            settlement_market=args.settlement_market,
            now=now,
            settings=settings,
        )
        if not args.json:
            print(_format_report(result))
            return 0
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
