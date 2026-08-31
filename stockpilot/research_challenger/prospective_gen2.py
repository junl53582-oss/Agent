from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research_v6.model import _sector_quotas
from stockpilot.prospective_r2.calendar import load_verified_calendar
from stockpilot.prospective_r2.integrity import (
    canonical_frame_bytes,
    canonical_json_bytes,
    read_verified_json,
    sha256_bytes,
    sha256_file,
    verify_immutable,
    write_immutable_bytes,
    write_immutable_frame,
    write_immutable_json,
)

from .config import ChallengerSettings
from .data import add_research_targets, assert_feature_columns_safe, verify_dataset_manifest
from .factors import select_factors_train_only
from .models import LightGBMModel, TrainOnlyPreprocessor, deterministic_full_date_sample


SHANGHAI = ZoneInfo("Asia/Shanghai")
HUMAN_DIR = Path(
    "artifacts/research_challenger/gen02/experiments/007_human_readjudication"
)
OPERATIONAL_AMENDMENT_DIR = HUMAN_DIR / "experiments/008_operational_portability_fix"
CORRECTNESS_LOCK = Path(
    "artifacts/research_challenger/gen02/experiments/005_correctness_hardening/plan.lock.json"
)
CORRECTNESS_INTERPRETATION_LOCK = Path(
    "artifacts/research_challenger/gen02/experiments/005_correctness_hardening/"
    "experiments/006_postrun_eligibility_interpretation/plan.lock.json"
)
V1R4_LOCK = Path("artifacts/prospective_alpha_v1r4/plan.lock.json")
V6_LOCK = Path("artifacts/research_v6/plan.lock.json")
CALENDAR_PATH = Path("artifacts/prospective_alpha_v1r2/trading_calendar_2026.json")
EXPECTED_CORRECTNESS_LOCK = (
    "2f6a670279aeccf69dc7ed596179562ad129ed28f53115e151d1d3db7d2a05bc"
)
EXPECTED_INTERPRETATION_LOCK = (
    "fa2789e6c4315f13ec91a5647d61bfdf31c49bf52915362ade3c9deab92b1bc2"
)
EXPECTED_V1R4_LOCK = (
    "39a6dfd2f63b68d1dfad77c039240ad5526786d1d2a0cfa67fd96426f27e1c1f"
)
EXPECTED_V6_LOCK = (
    "94edfc9e05bd30a58a14e7e11a988a1b7fb0d5358e462df1b20cb23dca4c0f4d"
)


@dataclass(frozen=True)
class ProspectiveGen2Settings:
    human_dir: Path = HUMAN_DIR
    human_lock_path: Path = HUMAN_DIR / "plan.lock.json"
    data_root: Path = Path("data/prospective_gen2")
    prediction_root: Path = Path("data/prospective_gen2/predictions")
    settlement_root: Path = Path("data/prospective_gen2/settlements")
    calendar_path: Path = CALENDAR_PATH
    dataset_path: Path = Path("artifacts/prediction_v30/cache/eligible_panel.parquet")
    dataset_manifest_path: Path = Path("artifacts/prediction_v30/cache/manifest.json")
    correctness_lock_path: Path = CORRECTNESS_LOCK
    correctness_interpretation_lock_path: Path = CORRECTNESS_INTERPRETATION_LOCK
    v1r4_lock_path: Path = V1R4_LOCK
    v6_lock_path: Path = V6_LOCK
    expected_correctness_lock: str = EXPECTED_CORRECTNESS_LOCK
    expected_interpretation_lock: str = EXPECTED_INTERPRETATION_LOCK
    expected_v1r4_lock: str = EXPECTED_V1R4_LOCK
    expected_v6_lock: str = EXPECTED_V6_LOCK
    top_k: int = 20
    horizon: int = 20
    rebalance_trading_days: int = 20
    training_window_years: int = 8
    validation_years: int = 1
    purge_gap_trading_days: int = 21
    selection_horizon: int = 5
    selection_purge_gap: int = 6
    random_seed: int = 42
    lightgbm_rounds: int = 80
    commission: float = 0.0003
    slippage: float = 0.0005
    sell_stamp_duty: float = 0.0005
    minimum_pipeline_days: int = 20
    provisional_review_days: int = 60
    evidence_review_days: int = 120


def _utc(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _policy_hash(value: dict) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _require_sidecar(path: Path, expected: str | None = None) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    # Some legacy frozen parents predate canonical sidecars.  They are accepted
    # only when a protocol-pinned exact digest is supplied; newer parents must
    # pass the full payload+sidecar verifier.
    digest = verify_immutable(path) if sidecar.is_file() else sha256_file(path)
    if not sidecar.is_file() and expected is None:
        raise RuntimeError(f"LEGACY_PARENT_REQUIRES_PINNED_HASH:{path}")
    if expected is not None and digest.lower() != expected.lower():
        raise RuntimeError(f"FROZEN_LOCK_MISMATCH:{path}")
    return digest


def model_specification(settings: ProspectiveGen2Settings) -> dict:
    return {
        "model_family": "lightgbm_regression",
        "objective": "regression_l1",
        "learning_rate": 0.04,
        "num_leaves": 15,
        "max_depth": 5,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "lambda_l1": 1.0,
        "lambda_l2": 5.0,
        "num_boost_round": settings.lightgbm_rounds,
        "random_seed": settings.random_seed,
        "deterministic": True,
    }


def feature_policy(settings: ProspectiveGen2Settings) -> dict:
    base = ChallengerSettings()
    return {
        "candidate_features": list(base.factor_columns),
        "selection": "V31 train-only BH-FDR/correlation/stability algorithm",
        "selection_horizon": settings.selection_horizon,
        "selection_purge_gap_trading_days": settings.selection_purge_gap,
        "selection_frequency": "once per prediction calendar year",
        "preprocessing": "train-only 1%/99% winsorization, median imputation, z-score",
        "future_columns_forbidden": True,
        "prospective_inputs_used": False,
    }


def training_policy(settings: ProspectiveGen2Settings) -> dict:
    return {
        "semantics": "DETERMINISTIC_PROTOCOL_RETRAIN",
        "horizon_trading_days": settings.horizon,
        "target": "cross-sectional rank of 20D T+1-open to T+21-open return",
        "training_window_years": settings.training_window_years,
        "validation_years": settings.validation_years,
        "purge_gap_trading_days": settings.purge_gap_trading_days,
        "annual_model_boundary": "calendar year, matching frozen Gen2/V31 OOS extension",
        "label_maturity": "label_end_date_20d must precede prediction-year boundary",
        "retuning_allowed": False,
        "hyperparameter_search_allowed": False,
    }


def portfolio_policy(settings: ProspectiveGen2Settings) -> dict:
    return {
        "name": "sector_balanced_top20",
        "top_k": settings.top_k,
        "rebalance_trading_days": settings.rebalance_trading_days,
        "weighting": "equal",
        "sector_quota": "research_v6.model._sector_quotas",
        "maximum_sector_weight_gate": 0.45,
        "cash_when_unfilled": True,
        "trading_allowed": False,
    }


def cost_policy(settings: ProspectiveGen2Settings) -> dict:
    return {
        "commission": settings.commission,
        "slippage": settings.slippage,
        "sell_stamp_duty": settings.sell_stamp_duty,
        "alpha_semantics": "research_proxy_alpha_only",
        "official_benchmark_status": "UNAPPROVED",
    }


def first_session_after(value: str, calendar_path: Path = CALENDAR_PATH) -> str:
    calendar = load_verified_calendar(calendar_path)
    later = calendar.sessions()[calendar.sessions() > pd.Timestamp(value)]
    if len(later) == 0:
        raise RuntimeError("NO_VERIFIED_FUTURE_TRADING_SESSION")
    return str(later[0].date())


def label_end_session(value: str, settings: ProspectiveGen2Settings) -> str:
    calendar = load_verified_calendar(settings.calendar_path)
    later = calendar.sessions()[calendar.sessions() > pd.Timestamp(value)]
    # T+1 open is entry; T+21 open is the 20D exit.
    if len(later) < settings.horizon + 1:
        raise RuntimeError("CALENDAR_DOES_NOT_COVER_LABEL_MATURITY")
    return str(later[settings.horizon].date())


def human_protocol(settings: ProspectiveGen2Settings, freeze_date: str) -> dict:
    return {
        "protocol_id": "GEN02-HUMAN-READJUDICATION-007",
        "classification": "HUMAN_GOVERNANCE_AND_PROSPECTIVE_RESEARCH_ONLY",
        "freeze_date_shanghai": freeze_date,
        "prospective_start_date": first_session_after(freeze_date, settings.calendar_path),
        "backfill_allowed": False,
        "historical_research_reopen_allowed": False,
        "automatic_promotion_allowed": False,
        "shadow_trading_allowed": False,
        "real_money_execution_allowed": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
        "review_checkpoints_trading_days": {
            "pipeline_only": settings.minimum_pipeline_days,
            "provisional_research_review_only": settings.provisional_review_days,
            "prospective_evidence_review_ready": settings.evidence_review_days,
        },
        "model": model_specification(settings),
        "features": feature_policy(settings),
        "training": training_policy(settings),
        "portfolio": portfolio_policy(settings),
        "cost": cost_policy(settings),
    }


def human_decision(settings: ProspectiveGen2Settings, freeze_date: str) -> dict:
    return {
        "decision_id": "GEN02-HUMAN-READJUDICATION-007",
        "operative_champion": "V6",
        "v6_champion": True,
        "gen2_historical_status": "HISTORICAL_RESEARCH_CLOSED",
        "gen2_promotion_status": "NOT_PROMOTED",
        "gen2_prospective_status": "PROSPECTIVE_RESEARCH_ONLY_APPROVED",
        "prospective_research_observation_approved": True,
        "production_shadow_eligible": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
        "automatic_promotion_allowed": False,
        "shadow_trading_allowed": False,
        "real_money_execution_allowed": False,
        "human_freeze_date_shanghai": freeze_date,
        "prospective_start_date": first_session_after(freeze_date, settings.calendar_path),
        "reasons": [
            "correctness recalculation produced one mechanically 8/8 configuration",
            "2026 is not an untouched holdout",
            "Top-K net research-proxy alpha bootstrap confidence interval crosses zero",
            "2025 RankIC is only approximately +0.00125",
            "2025 top-decile IC remains negative",
            "evidence is insufficient to replace V6 but sufficient for frozen research-only observation",
        ],
    }


def human_audit(settings: ProspectiveGen2Settings) -> dict:
    locks = {
        "correctness": _require_sidecar(
            settings.correctness_lock_path, settings.expected_correctness_lock
        ),
        "interpretation": _require_sidecar(
            settings.correctness_interpretation_lock_path,
            settings.expected_interpretation_lock,
        ),
        "v1r4": _require_sidecar(settings.v1r4_lock_path, settings.expected_v1r4_lock),
        "v6": _require_sidecar(settings.v6_lock_path, settings.expected_v6_lock),
    }
    return {
        "audit_id": "GEN02-HUMAN-READJUDICATION-AUDIT-007",
        "parent_locks": locks,
        "original_gen2_cross_boundary_label_bug_detected": True,
        "original_gen2_2025_decision_rows_with_label_end_in_2026_existed": True,
        "corrected_run_consumed_2026_realized_labels": False,
        "corrected_run_holdout_opened": False,
        "untouched_2026_holdout": False,
        "historical_tuning_runs": 0,
        "hyperparameter_changes": 0,
        "feature_changes": 0,
        "portfolio_changes": 0,
        "provider_requests": {"market": 0, "financial": 0, "benchmark": 0},
        "v1r4_modified": False,
        "v30_modified": False,
        "v30r1_modified": False,
        "v6_modified": False,
    }


def freeze_human_readjudication(
    settings: ProspectiveGen2Settings | None = None,
    *,
    now: datetime | None = None,
    source_commit: str | None = None,
) -> dict:
    settings = settings or ProspectiveGen2Settings()
    now = now or datetime.now(timezone.utc)
    freeze_date = now.astimezone(SHANGHAI).date().isoformat()
    if settings.human_dir.exists() and any(settings.human_dir.iterdir()):
        raise RuntimeError("HUMAN_READJUDICATION_ALREADY_EXISTS")
    settings.human_dir.mkdir(parents=True, exist_ok=True)
    protocol = human_protocol(settings, freeze_date)
    audit = human_audit(settings)
    decision = human_decision(settings, freeze_date)
    protocol_hash = write_immutable_json(settings.human_dir / "protocol.json", protocol)
    audit_hash = write_immutable_json(settings.human_dir / "audit.json", audit)
    decision_hash = write_immutable_json(settings.human_dir / "decision.json", decision)
    spec = {
        "model_id": "GEN2-LGBM-20D-SECTOR-BALANCED-TOP20",
        "source_commit": source_commit or _git("rev-parse", "HEAD"),
        "parent_correctness_lock_sha256": settings.expected_correctness_lock,
        "human_decision_sha256": decision_hash,
        "model_specification": model_specification(settings),
        "model_spec_hash": _policy_hash(model_specification(settings)),
        "feature_policy": feature_policy(settings),
        "feature_policy_hash": _policy_hash(feature_policy(settings)),
        "training_policy": training_policy(settings),
        "training_policy_hash": _policy_hash(training_policy(settings)),
        "portfolio_policy": portfolio_policy(settings),
        "portfolio_policy_hash": _policy_hash(portfolio_policy(settings)),
        "cost_policy": cost_policy(settings),
        "cost_policy_hash": _policy_hash(cost_policy(settings)),
        "prospective_training_semantics": "DETERMINISTIC_PROTOCOL_RETRAIN",
        "research_only": True,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    spec_hash = write_immutable_json(settings.human_dir / "challenger_spec.json", spec)
    manifest = {
        "protocol.json": protocol_hash,
        "audit.json": audit_hash,
        "decision.json": decision_hash,
        "challenger_spec.json": spec_hash,
    }
    manifest_hash = write_immutable_json(
        settings.human_dir / "artifact_manifest.json", manifest
    )
    files = [
        Path("stockpilot/research_challenger/prospective_gen2.py"),
        Path("tests/test_research_challenger_gen2_prospective.py"),
        settings.human_dir / "protocol.json",
        settings.human_dir / "audit.json",
        settings.human_dir / "decision.json",
        settings.human_dir / "challenger_spec.json",
        settings.human_dir / "artifact_manifest.json",
        settings.correctness_lock_path,
        settings.correctness_interpretation_lock_path,
        settings.v1r4_lock_path,
        settings.v6_lock_path,
        settings.calendar_path,
    ]
    missing = [path.as_posix() for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"HUMAN_FREEZE_INPUT_MISSING:{missing}")
    lock = {
        "lock_id": "GEN02-HUMAN-READJUDICATION-007",
        "created_at_utc": _utc(now),
        "source_commit": spec["source_commit"],
        "files": {path.as_posix(): sha256_file(path) for path in files},
        "artifact_manifest_sha256": manifest_hash,
        "prospective_start_date": decision["prospective_start_date"],
        "automatic_promotion_allowed": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    lock_hash = write_immutable_json(settings.human_lock_path, lock)
    return {
        "status": "PROSPECTIVE_RESEARCH_ONLY_APPROVED",
        "lock_sha256": lock_hash,
        "artifact_manifest_sha256": manifest_hash,
        "prospective_start_date": decision["prospective_start_date"],
        "provider_requests": 0,
    }


def verify_human_freeze(settings: ProspectiveGen2Settings | None = None) -> dict:
    settings = settings or ProspectiveGen2Settings()
    amendment = settings.human_dir / "experiments/008_operational_portability_fix/plan.lock.json"
    lock_path = amendment if amendment.is_file() else settings.human_lock_path
    lock = read_verified_json(lock_path)
    mismatches = []
    for name, expected in lock["files"].items():
        path = Path(name)
        if not path.is_file() or sha256_file(path) != expected:
            mismatches.append(name)
    decision = read_verified_json(settings.human_dir / "decision.json")
    if decision.get("operative_champion") != "V6":
        mismatches.append("OPERATIVE_CHAMPION")
    if decision.get("execution_authorized") is not False:
        mismatches.append("EXECUTION_AUTHORIZED")
    return {
        "intact": not mismatches,
        "mismatches": mismatches,
        "lock_sha256": sha256_file(lock_path),
        "operational_amendment": lock_path != settings.human_lock_path,
        "prospective_start_date": lock["prospective_start_date"],
        "v6_champion": True,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }


def freeze_operational_portability_amendment(
    settings: ProspectiveGen2Settings | None = None,
    *,
    now: datetime | None = None,
) -> dict:
    settings = settings or ProspectiveGen2Settings()
    now = now or datetime.now(timezone.utc)
    target = settings.human_dir / "experiments/008_operational_portability_fix"
    if target.exists() and any(target.iterdir()):
        raise RuntimeError("OPERATIONAL_PORTABILITY_AMENDMENT_ALREADY_EXISTS")
    original_lock = settings.human_lock_path
    if not original_lock.is_file() or not original_lock.with_suffix(original_lock.suffix + ".sha256").is_file():
        raise RuntimeError("ORIGINAL_HUMAN_LOCK_MISSING")
    original_hash = verify_immutable(original_lock)
    protocol = {
        "amendment_id": "GEN02-HUMAN-READJUDICATION-008-PORTABILITY",
        "classification": "OPERATIONAL_PATH_PORTABILITY_ONLY",
        "reason": "007 lock recorded __file__ as a Windows absolute path; clean checkout cannot resolve it",
        "human_decision_changed": False,
        "model_changed": False,
        "feature_policy_changed": False,
        "training_policy_changed": False,
        "portfolio_policy_changed": False,
        "cost_policy_changed": False,
        "prospective_start_date_changed": False,
        "provider_requests": 0,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    target.mkdir(parents=True, exist_ok=True)
    protocol_hash = write_immutable_json(target / "protocol_amendment.json", protocol)
    paths = [
        Path("stockpilot/research_challenger/prospective_gen2.py"),
        Path("tests/test_research_challenger_gen2_prospective.py"),
        settings.human_dir / "protocol.json",
        settings.human_dir / "audit.json",
        settings.human_dir / "decision.json",
        settings.human_dir / "challenger_spec.json",
        settings.human_dir / "artifact_manifest.json",
        original_lock,
        target / "protocol_amendment.json",
        settings.correctness_lock_path,
        settings.correctness_interpretation_lock_path,
        settings.v1r4_lock_path,
        settings.v6_lock_path,
        settings.calendar_path,
    ]
    lock = {
        "lock_id": "GEN02-HUMAN-READJUDICATION-008-PORTABILITY",
        "created_at_utc": _utc(now),
        "parent_007_lock_sha256": original_hash,
        "protocol_amendment_sha256": protocol_hash,
        "files": {path.as_posix(): sha256_file(path) for path in paths},
        "prospective_start_date": read_verified_json(settings.human_dir / "decision.json")["prospective_start_date"],
        "automatic_promotion_allowed": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    digest = write_immutable_json(target / "plan.lock.json", lock)
    return {
        "status": "OPERATIONAL_PORTABILITY_AMENDMENT_FROZEN",
        "lock_sha256": digest,
        "parent_007_lock_sha256": original_hash,
        "prospective_start_date": lock["prospective_start_date"],
    }


def _validate_prediction_date(
    target_date: str, now: datetime, settings: ProspectiveGen2Settings
) -> None:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    human = verify_human_freeze(settings)
    if not human["intact"]:
        raise RuntimeError("HUMAN_READJUDICATION_LOCK_INVALID")
    local_date = now.astimezone(SHANGHAI).date().isoformat()
    if target_date < human["prospective_start_date"]:
        raise RuntimeError("HISTORICAL_BACKFILL_FORBIDDEN")
    if target_date != local_date:
        raise RuntimeError("PREDICTION_DATE_MUST_EQUAL_CURRENT_SHANGHAI_DATE")
    calendar = load_verified_calendar(settings.calendar_path)
    if not calendar.is_session(target_date):
        raise RuntimeError("PREDICTION_DATE_NOT_VERIFIED_TRADING_SESSION")


def _default_train_and_score(
    target_date: str, settings: ProspectiveGen2Settings
) -> tuple[pd.DataFrame, dict]:
    base = ChallengerSettings(
        dataset_path=settings.dataset_path,
        dataset_manifest_path=settings.dataset_manifest_path,
    )
    verify_dataset_manifest(base)
    safe_features = tuple(base.factor_columns)
    assert_feature_columns_safe(safe_features)
    identity = [
        "date",
        "symbol",
        "broad_sector",
        "industry",
        "benchmark_weight",
        "benchmark_weight_rank",
    ]
    target = pd.Timestamp(target_date)
    current = pd.read_parquet(
        settings.dataset_path,
        columns=[*identity, *safe_features],
        filters=[("date", "==", target)],
    )
    if current.empty:
        raise RuntimeError("TARGET_DATE_PIT_FEATURES_NOT_AVAILABLE")
    current["date"] = pd.to_datetime(current["date"])
    if current["symbol"].astype(str).duplicated().any():
        raise RuntimeError("TARGET_DATE_DUPLICATE_SYMBOL")
    year_start = pd.Timestamp(target.year, 1, 1)
    validation_start = pd.Timestamp(target.year - 1, 1, 1)
    train_start = pd.Timestamp(target.year - settings.training_window_years - 1, 1, 1)
    training = pd.read_parquet(
        settings.dataset_path,
        columns=[
            *identity,
            *safe_features,
            "future_return_5d",
            "future_return_20d",
            "label_end_date_5d",
            "label_end_date_20d",
        ],
        filters=[
            ("date", ">=", train_start),
            ("date", "<", year_start),
            ("label_end_date_20d", "<", year_start),
        ],
    )
    training["date"] = pd.to_datetime(training["date"])
    training["label_end_date_5d"] = pd.to_datetime(training["label_end_date_5d"])
    training["label_end_date_20d"] = pd.to_datetime(training["label_end_date_20d"])
    training = add_research_targets(training)
    dates = pd.DatetimeIndex(training["date"].drop_duplicates().sort_values())
    before_validation = dates[dates < validation_start]
    before_year = dates[dates < year_start]
    if len(before_validation) <= settings.selection_purge_gap or len(before_year) <= settings.purge_gap_trading_days:
        raise RuntimeError("INSUFFICIENT_PURGED_TRAINING_DATES")
    selection_cutoff = before_validation[-(settings.selection_purge_gap + 1)]
    refit_cutoff = before_year[-(settings.purge_gap_trading_days + 1)]
    selection_train = training[
        training["date"].le(selection_cutoff)
        & training["label_end_date_5d"].lt(validation_start)
    ].copy()
    refit = training[
        training["date"].le(refit_cutoff)
        & training["label_end_date_20d"].lt(year_start)
    ].copy()
    selection = select_factors_train_only(selection_train, base)
    features = selection.selected
    target_column = "return_rank_20d"
    finite = pd.to_numeric(refit[target_column], errors="coerce")
    refit = refit[finite.notna() & np.isfinite(finite)].copy()
    sample = deterministic_full_date_sample(refit, base.training_row_cap)
    processor = TrainOnlyPreprocessor().fit(sample, features)
    model = LightGBMModel(
        "regression_l1", settings.lightgbm_rounds, settings.random_seed
    ).fit(
        processor.transform(sample, features),
        pd.to_numeric(sample[target_column], errors="raise").to_numpy(dtype=float),
    )
    scored = current[identity].copy()
    scored["score"] = model.predict(processor.transform(current, features))
    snapshot_columns = ["date", "symbol", target_column, *features]
    training_snapshot_hash = sha256_bytes(
        canonical_frame_bytes(sample[snapshot_columns], ["date", "symbol"])
    )
    return scored, {
        "model_signature": model.signature(),
        "training_snapshot_hash": training_snapshot_hash,
        "training_rows": int(len(sample)),
        "training_date_min": str(sample["date"].min().date()),
        "training_date_max": str(sample["date"].max().date()),
        "training_label_end_max": str(sample["label_end_date_20d"].max().date()),
        "selected_features": list(features),
        "selected_features_hash": _policy_hash({"features": list(features)}),
        "input_snapshot_hash": sha256_bytes(
            canonical_frame_bytes(current, ["date", "symbol"])
        ),
        "dataset_sha256": sha256_file(settings.dataset_path),
        "dataset_manifest_sha256": sha256_file(settings.dataset_manifest_path),
        "2026_realized_labels_read": False,
    }


def _sector_balanced_weights(scored: pd.DataFrame, top_k: int) -> pd.DataFrame:
    ranked = scored.sort_values(["score", "symbol"], ascending=[False, True]).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    quotas = _sector_quotas(ranked, top_k)
    selected = pd.concat(
        [ranked[ranked["broad_sector"].astype(str).eq(sector)].head(quota) for sector, quota in quotas.items()],
        ignore_index=False,
    ).sort_values(["score", "symbol"], ascending=[False, True]).head(top_k)
    ranked["selected_top20"] = ranked.index.isin(selected.index)
    ranked["portfolio_weight"] = 0.0
    if len(selected):
        ranked.loc[selected.index, "portfolio_weight"] = 1.0 / len(selected)
    return ranked


def generate_prediction(
    target_date: str,
    *,
    now: datetime | None = None,
    settings: ProspectiveGen2Settings | None = None,
    scorer: Callable[[str, ProspectiveGen2Settings], tuple[pd.DataFrame, dict]] | None = None,
) -> dict:
    settings = settings or ProspectiveGen2Settings()
    now = now or datetime.now(timezone.utc)
    _validate_prediction_date(target_date, now, settings)
    directory = settings.prediction_root / target_date
    if directory.exists():
        raise RuntimeError("IMMUTABLE_PREDICTION_ALREADY_EXISTS")
    scored, evidence = (scorer or _default_train_and_score)(target_date, settings)
    required = {"date", "symbol", "broad_sector", "score"}
    if required - set(scored):
        raise RuntimeError("PREDICTION_SCORE_SCHEMA_INVALID")
    if scored["symbol"].astype(str).duplicated().any() or not np.isfinite(pd.to_numeric(scored["score"], errors="coerce")).all():
        raise RuntimeError("PREDICTION_SCORE_INVALID")
    ranked = _sector_balanced_weights(scored, settings.top_k)
    spec = read_verified_json(settings.human_dir / "challenger_spec.json")
    expected_policies = {
        "model_spec_hash": _policy_hash(model_specification(settings)),
        "feature_policy_hash": _policy_hash(feature_policy(settings)),
        "training_policy_hash": _policy_hash(training_policy(settings)),
        "portfolio_policy_hash": _policy_hash(portfolio_policy(settings)),
        "cost_policy_hash": _policy_hash(cost_policy(settings)),
    }
    for name, expected in expected_policies.items():
        if spec.get(name) != expected:
            raise RuntimeError(f"FROZEN_SPEC_HASH_MISMATCH:{name}")
    maturity = label_end_session(target_date, settings)
    output = ranked[["date", "symbol", "broad_sector", "score", "rank", "selected_top20", "portfolio_weight"]].copy()
    output["prediction_date"] = target_date
    output["model_id"] = spec["model_id"]
    output["research_only"] = True
    output["production_prediction_ready"] = False
    output["execution_authorized"] = False
    csv_path = directory / "prediction.csv"
    csv_hash = write_immutable_frame(csv_path, output, ["prediction_date", "symbol"])
    previous = sorted(settings.prediction_root.glob("*/manifest.json"))
    previous_hash = verify_immutable(previous[-1]) if previous else None
    receipt = {
        "prediction_date": target_date,
        "created_at_utc": _utc(now),
        "source_commit": _git("rev-parse", "HEAD"),
        "model_id": spec["model_id"],
        "model_spec_hash": spec["model_spec_hash"],
        "feature_policy_hash": spec["feature_policy_hash"],
        "training_policy_hash": spec["training_policy_hash"],
        "portfolio_policy_hash": spec["portfolio_policy_hash"],
        "training_snapshot_hash": evidence["training_snapshot_hash"],
        "input_snapshot_hash": evidence["input_snapshot_hash"],
        "universe_snapshot_hash": _policy_hash({"symbols": sorted(output["symbol"].astype(str))}),
        "calendar_hash": sha256_file(settings.calendar_path),
        "prediction_csv_sha256": csv_hash,
        "rows": int(len(output)),
        "selected": int(output["selected_top20"].sum()),
        "cash_weight": float(1.0 - output["portfolio_weight"].sum()),
        "sector_balanced_policy_receipt": output[output["selected_top20"]].groupby("broad_sector")["portfolio_weight"].sum().to_dict(),
        "prediction_status": "RECORDED_RESEARCH_ONLY",
        "label_maturity_date": maturity,
        "expected_settlement_date": maturity,
        "maturity_status": "PENDING",
        "benchmark_status": "UNAPPROVED",
        "previous_prediction_manifest_hash": previous_hash,
        "training_evidence": evidence,
        "future_label_fields_present": False,
        "research_only": True,
        "automatic_promotion_allowed": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    receipt_hash = write_immutable_json(directory / "prediction.json", receipt)
    manifest = {
        "prediction.json": receipt_hash,
        "prediction.csv": csv_hash,
        "previous_prediction_manifest_hash": previous_hash,
    }
    manifest_hash = write_immutable_json(directory / "manifest.json", manifest)
    return receipt | {"manifest_sha256": manifest_hash}


def settle_prediction(
    prediction_date: str,
    market_path: Path,
    *,
    as_of_date: str,
    settings: ProspectiveGen2Settings | None = None,
    official_alpha_requested: bool = False,
) -> dict:
    settings = settings or ProspectiveGen2Settings()
    if official_alpha_requested:
        raise RuntimeError("OFFICIAL_ALPHA_BLOCKED_BENCHMARK_UNAPPROVED")
    directory = settings.prediction_root / prediction_date
    receipt = read_verified_json(directory / "prediction.json")
    manifest = read_verified_json(directory / "manifest.json")
    if verify_immutable(directory / "prediction.csv") != manifest["prediction.csv"]:
        raise RuntimeError("PREDICTION_MANIFEST_HASH_MISMATCH")
    maturity = receipt["label_maturity_date"]
    if pd.Timestamp(as_of_date) < pd.Timestamp(maturity):
        raise RuntimeError("20D_LABEL_NOT_MATURE")
    verify_immutable(market_path)
    market = pd.read_csv(market_path, dtype={"symbol": str})
    required = {"date", "symbol", "open"}
    if required - set(market):
        raise RuntimeError("SETTLEMENT_MARKET_SCHEMA_INVALID")
    market["date"] = pd.to_datetime(market["date"])
    market["symbol"] = market["symbol"].astype(str).str.zfill(6)
    predictions = pd.read_csv(directory / "prediction.csv", dtype={"symbol": str})
    calendar = load_verified_calendar(settings.calendar_path)
    later = calendar.sessions()[calendar.sessions() > pd.Timestamp(prediction_date)]
    entry_date, exit_date = later[0], later[settings.horizon]
    entry = market[market["date"].eq(entry_date)].set_index("symbol")["open"]
    exit_ = market[market["date"].eq(exit_date)].set_index("symbol")["open"]
    rows = predictions[["symbol", "score", "rank", "selected_top20", "portfolio_weight"]].copy()
    rows["prediction_date"] = prediction_date
    rows["entry_date"] = str(entry_date.date())
    rows["exit_date"] = str(exit_date.date())
    rows["entry_open"] = rows["symbol"].map(entry)
    rows["exit_open"] = rows["symbol"].map(exit_)
    rows["actual_return_20d"] = rows["exit_open"] / rows["entry_open"] - 1.0
    rows["settled"] = rows[["entry_open", "exit_open"]].notna().all(axis=1)
    rows["benchmark_status"] = "UNAPPROVED"
    rows["official_alpha_status"] = "PENDING_BENCHMARK_APPROVAL"
    rows["research_only"] = True
    target = settings.settlement_root / prediction_date / "settlement.csv"
    digest = write_immutable_frame(target, rows, ["prediction_date", "symbol"])
    valid = rows[rows["settled"]]
    rank_ic = valid["score"].corr(valid["actual_return_20d"], method="spearman") if len(valid) >= 3 else np.nan
    selected = valid[valid["selected_top20"].astype(bool)]
    summary = {
        "prediction_date": prediction_date,
        "maturity_date": maturity,
        "settlement_status": "SETTLED_RESEARCH_PROXY_ONLY",
        "settled_symbols": int(len(valid)),
        "rank_ic": None if pd.isna(rank_ic) else float(rank_ic),
        "top20_return": float(np.average(selected["actual_return_20d"], weights=selected["portfolio_weight"])) if len(selected) else None,
        "universe_return": float(valid["actual_return_20d"].mean()) if len(valid) else None,
        "research_proxy_spread": float(selected["actual_return_20d"].mean() - valid["actual_return_20d"].mean()) if len(selected) and len(valid) else None,
        "official_benchmark_status": "UNAPPROVED",
        "official_alpha_status": "PENDING_BENCHMARK_APPROVAL",
        "prediction_manifest_sha256": verify_immutable(directory / "manifest.json"),
        "market_source_sha256": sha256_file(market_path),
        "settlement_csv_sha256": digest,
        "prediction_recomputed": False,
        "automatic_promotion_allowed": False,
        "execution_authorized": False,
    }
    summary_hash = write_immutable_json(
        settings.settlement_root / prediction_date / "settlement.json", summary
    )
    return summary | {"settlement_json_sha256": summary_hash}


def review_checkpoint(settings: ProspectiveGen2Settings | None = None) -> dict:
    settings = settings or ProspectiveGen2Settings()
    prediction_dates = sorted(path.parent.name for path in settings.prediction_root.glob("*/manifest.json"))
    settled_dates = sorted(path.parent.name for path in settings.settlement_root.glob("*/settlement.json"))
    count = len(prediction_dates)
    if count < settings.minimum_pipeline_days:
        status = "PIPELINE_ONLY"
    elif count < settings.provisional_review_days:
        status = "PIPELINE_ONLY_COMPLETE_NO_MODEL_JUDGMENT"
    elif count < settings.evidence_review_days:
        status = "PROVISIONAL_RESEARCH_REVIEW_ONLY"
    else:
        status = "PROSPECTIVE_EVIDENCE_REVIEW_READY" if len(settled_dates) >= settings.provisional_review_days else "WAITING_FOR_MATURE_20D_EVIDENCE"
    return {
        "prediction_trading_days": count,
        "settled_20d_dates": len(settled_dates),
        "status": status,
        "human_review_required": True,
        "automatic_promotion_allowed": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frozen Gen2 prospective research-only chain")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze-human")
    sub.add_parser("freeze-operational-amendment")
    sub.add_parser("verify")
    predict = sub.add_parser("predict")
    predict.add_argument("--date", required=True)
    settle = sub.add_parser("settle")
    settle.add_argument("--date", required=True)
    settle.add_argument("--market", type=Path, required=True)
    settle.add_argument("--as-of", required=True)
    sub.add_parser("status")
    args = parser.parse_args(argv)
    if args.command == "freeze-human":
        result = freeze_human_readjudication()
    elif args.command == "freeze-operational-amendment":
        result = freeze_operational_portability_amendment()
    elif args.command == "verify":
        result = verify_human_freeze()
    elif args.command == "predict":
        result = generate_prediction(args.date)
    elif args.command == "settle":
        result = settle_prediction(args.date, args.market, as_of_date=args.as_of)
    else:
        result = review_checkpoint()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
