from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research_v20r2.config import V20R2Settings
from research_v20r2.ledger import Ledger, PriceBook
from research_v6.model import _sector_quotas
from stockpilot.prospective_r2.calendar import load_verified_calendar
from stockpilot.prospective_r2.integrity import (
    canonical_frame_bytes,
    canonical_json_bytes,
    read_verified_json,
    sha256_bytes,
    sha256_file,
    verify_immutable,
    write_atomic_reservation,
    write_immutable_frame,
    write_immutable_json,
)

from .config import ChallengerSettings
from .data import add_research_targets, assert_feature_columns_safe, verify_dataset_manifest
from .factors import select_factors_train_only
from .models import LightGBMModel, TrainOnlyPreprocessor, deterministic_full_date_sample
from .prospective_gen2 import (
    HUMAN_DIR,
    ProspectiveGen2Settings,
    _policy_hash,
    cost_policy,
    feature_policy,
    label_end_session,
    model_specification,
    portfolio_policy,
    review_checkpoint,
    training_policy,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")
AMENDMENT_008 = HUMAN_DIR / "experiments/008_operational_portability_fix/plan.lock.json"
AMENDMENT_009 = HUMAN_DIR / "experiments/009_prospective_runtime_hardening"
EXPECTED_008_LOCK = "1abb4bf9c65875b4e96918f931bfa78299ee5aa1938b4e02acd8fc2614a92022"
FORBIDDEN_RUNTIME_FEATURE_TOKENS = ("future_", "label", "entry_", "exit_", "actual_", "realized_")


@dataclass(frozen=True)
class RuntimeSettings(ProspectiveGen2Settings):
    input_seal_root: Path = Path("data/prospective_gen2/input_seals")
    reservation_root: Path = Path("data/prospective_gen2/_prediction_attempts")
    portfolio_root: Path = Path("data/prospective_gen2/portfolio")
    runtime_lock_path: Path = AMENDMENT_009 / "plan.lock.json"
    parent_008_lock_path: Path = AMENDMENT_008
    expected_parent_008_lock: str = EXPECTED_008_LOCK
    earliest_prediction_time: time = time(18, 30)
    rebalance_anchor_date: str = "2026-09-01"
    test_mode: bool = False
    factor_columns_override: tuple[str, ...] | None = None
    training_row_cap_override: int | None = None
    v1r4_input_evidence_root: Path = Path("data/prospective_alpha_v1r4/prediction_input_evidence")


def _utc(now: datetime) -> str:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(timezone.utc).isoformat()


def _require_parent(settings: RuntimeSettings) -> str:
    digest = verify_immutable(settings.parent_008_lock_path)
    if digest != settings.expected_parent_008_lock:
        raise RuntimeError("PARENT_008_LOCK_MISMATCH")
    return digest


def _verify_policy_hashes(settings: RuntimeSettings) -> dict:
    spec = read_verified_json(settings.human_dir / "challenger_spec.json")
    expected = {
        "model_spec_hash": _policy_hash(model_specification(settings)),
        "feature_policy_hash": _policy_hash(feature_policy(settings)),
        "training_policy_hash": _policy_hash(training_policy(settings)),
        "portfolio_policy_hash": _policy_hash(portfolio_policy(settings)),
        "cost_policy_hash": _policy_hash(cost_policy(settings)),
    }
    for key, value in expected.items():
        if spec.get(key) != value:
            raise RuntimeError(f"FROZEN_SPEC_HASH_MISMATCH:{key}")
    return spec


def _session_position(target_date: str, settings: RuntimeSettings) -> tuple[pd.DatetimeIndex, int]:
    sessions = load_verified_calendar(settings.calendar_path).sessions()
    matches = np.flatnonzero(sessions == pd.Timestamp(target_date))
    if len(matches) != 1:
        raise RuntimeError("PREDICTION_DATE_NOT_VERIFIED_TRADING_SESSION")
    return sessions, int(matches[0])


def _is_rebalance_day(target_date: str, settings: RuntimeSettings) -> tuple[bool, int]:
    sessions, target_pos = _session_position(target_date, settings)
    anchor = np.flatnonzero(sessions == pd.Timestamp(settings.rebalance_anchor_date))
    if len(anchor) != 1 or target_pos < int(anchor[0]):
        raise RuntimeError("REBALANCE_ANCHOR_INVALID")
    distance = target_pos - int(anchor[0])
    return distance % settings.rebalance_trading_days == 0, distance // settings.rebalance_trading_days


def _validate_clock(target_date: str, now: datetime, settings: RuntimeSettings) -> None:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    local = now.astimezone(SHANGHAI)
    start = read_verified_json(settings.human_dir / "decision.json")["prospective_start_date"]
    if target_date < start:
        raise RuntimeError("HISTORICAL_BACKFILL_FORBIDDEN")
    if target_date != local.date().isoformat():
        raise RuntimeError("PREDICTION_DATE_MUST_EQUAL_CURRENT_SHANGHAI_DATE")
    _session_position(target_date, settings)
    if local.timetz().replace(tzinfo=None) < settings.earliest_prediction_time:
        raise RuntimeError("GEN2_INPUT_NOT_READY:UPSTREAM_DATA_WINDOW_NOT_OPEN")


def _features(settings: RuntimeSettings) -> tuple[str, ...]:
    return settings.factor_columns_override or tuple(ChallengerSettings().factor_columns)


def _read_target_panel(target_date: str, settings: RuntimeSettings) -> tuple[pd.DataFrame, dict]:
    features = _features(settings)
    assert_feature_columns_safe(features)
    columns = [
        "date", "symbol", "eligible", "in_universe", "membership_snapshot_date",
        "available_date", "industry_effective_date", "industry", "broad_sector",
        "benchmark_weight", "benchmark_weight_rank", *features,
    ]
    try:
        panel = pd.read_parquet(
            settings.dataset_path,
            columns=list(dict.fromkeys(columns)),
            filters=[("date", "==", pd.Timestamp(target_date))],
        )
    except Exception as error:
        raise RuntimeError(f"TARGET_DATE_INPUT_READ_FAILED:{error}") from error
    if panel.empty:
        raise RuntimeError("TARGET_DATE_PIT_FEATURES_NOT_AVAILABLE")
    missing = set(columns) - set(panel.columns)
    if missing:
        raise RuntimeError(f"TARGET_DATE_SCHEMA_MISSING:{sorted(missing)}")
    panel["date"] = pd.to_datetime(panel["date"], errors="raise").dt.normalize()
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
    if panel.duplicated(["date", "symbol"]).any():
        raise RuntimeError("TARGET_DATE_DUPLICATE_SYMBOL")
    if not panel["date"].eq(pd.Timestamp(target_date)).all():
        raise RuntimeError("TARGET_DATE_MISMATCH")
    decision = panel["date"]
    membership = pd.to_datetime(panel["membership_snapshot_date"], errors="coerce")
    available = pd.to_datetime(panel["available_date"], errors="coerce")
    industry_date = pd.to_datetime(panel["industry_effective_date"], errors="coerce")
    checks = {
        "membership_pit": bool(membership.notna().all() and membership.le(decision).all()),
        "fundamental_availability_pit": bool(available.notna().all() and available.le(decision).all()),
        "industry_pit": bool(industry_date.notna().all() and industry_date.le(decision).all()),
        "eligible_boolean": bool(panel["eligible"].isin([True, False]).all()),
        "in_universe_boolean": bool(panel["in_universe"].isin([True, False]).all()),
        "industry_present": bool(panel[["industry", "broad_sector"]].notna().all().all()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"TARGET_DATE_PIT_GATE_FAILED:{checks}")
    eligible = panel["eligible"].eq(True)
    in_universe = panel["in_universe"].eq(True)
    final = panel.loc[eligible & in_universe].copy()
    if final.empty:
        raise RuntimeError("TARGET_DATE_ELIGIBLE_UNIVERSE_EMPTY")
    numeric = final[list(features)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError("TARGET_DATE_MODEL_FEATURES_NOT_FINITE")
    weights = pd.to_numeric(final["benchmark_weight"], errors="coerce")
    if not np.isfinite(weights).all() or weights.le(0).any():
        raise RuntimeError("TARGET_DATE_BENCHMARK_WEIGHT_INVALID")
    evidence = {
        "pit_checks": checks,
        "target_date_row_count": int(len(panel)),
        "eligible_count": int(eligible.sum()),
        "in_universe_count": int(in_universe.sum()),
        "final_prediction_rows": int(len(final)),
        "excluded_rows": int(len(panel) - len(final)),
        "symbol_count": int(final["symbol"].nunique()),
        "membership_snapshot_max": str(membership.max().date()),
        "fundamental_available_max": str(available.max().date()),
        "industry_effective_max": str(industry_date.max().date()),
    }
    return final.sort_values(["date", "symbol"]).reset_index(drop=True), evidence


def seal_inputs(
    target_date: str,
    *,
    now: datetime | None = None,
    settings: RuntimeSettings | None = None,
) -> dict:
    settings = settings or RuntimeSettings()
    now = now or datetime.now(timezone.utc)
    _require_parent(settings)
    _verify_policy_hashes(settings)
    _validate_clock(target_date, now, settings)
    if (settings.reservation_root / f"{target_date}.json").exists():
        raise RuntimeError("GEN2_PREDICTION_ATTEMPT_ALREADY_RESERVED")
    verify_dataset_manifest(
        ChallengerSettings(
            dataset_path=settings.dataset_path,
            dataset_manifest_path=settings.dataset_manifest_path,
        )
    )
    panel, evidence = _read_target_panel(target_date, settings)
    dataset_hash = sha256_file(settings.dataset_path)
    manifest_hash = sha256_file(settings.dataset_manifest_path)
    value = {
        "date": target_date,
        "created_at_utc": _utc(now),
        "source_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dataset_sha256": dataset_hash,
        "dataset_manifest_sha256": manifest_hash,
        "calendar_sha256": sha256_file(settings.calendar_path),
        **evidence,
        "input_snapshot_hash": sha256_bytes(canonical_frame_bytes(panel, ["date", "symbol"])),
        "source_hashes": json.loads(settings.dataset_manifest_path.read_text(encoding="utf-8"))["source_hashes"],
        "sealed": True,
        "provider_requests_made": 0,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    path = settings.input_seal_root / f"{target_date}.json"
    digest = write_immutable_json(path, value)
    return value | {"seal_path": path.as_posix(), "seal_sha256": digest}


def _verify_seal(target_date: str, settings: RuntimeSettings) -> tuple[pd.DataFrame, dict]:
    path = settings.input_seal_root / f"{target_date}.json"
    seal = read_verified_json(path)
    if seal.get("sealed") is not True or seal.get("date") != target_date:
        raise RuntimeError("GEN2_INPUT_SEAL_INVALID")
    if sha256_file(settings.dataset_path) != seal["dataset_sha256"]:
        raise RuntimeError("SEALED_INPUT_HASH_CHANGED")
    if sha256_file(settings.dataset_manifest_path) != seal["dataset_manifest_sha256"]:
        raise RuntimeError("SEALED_INPUT_MANIFEST_HASH_CHANGED")
    panel, evidence = _read_target_panel(target_date, settings)
    if sha256_bytes(canonical_frame_bytes(panel, ["date", "symbol"])) != seal["input_snapshot_hash"]:
        raise RuntimeError("SEALED_TARGET_PANEL_HASH_CHANGED")
    for key in ("pit_checks", "target_date_row_count", "eligible_count", "in_universe_count", "final_prediction_rows", "excluded_rows"):
        if seal.get(key) != evidence.get(key):
            raise RuntimeError(f"SEALED_INPUT_EVIDENCE_CHANGED:{key}")
    return panel, seal


def preflight(
    target_date: str,
    *,
    now: datetime | None = None,
    settings: RuntimeSettings | None = None,
) -> dict:
    settings = settings or RuntimeSettings()
    now = now or datetime.now(timezone.utc)
    failures: list[str] = []
    parent = policies = clock = seal_ok = False
    panel_rows = 0
    try:
        _require_parent(settings)
        parent = True
        _verify_policy_hashes(settings)
        policies = True
        _validate_clock(target_date, now, settings)
        clock = True
        panel, _ = _verify_seal(target_date, settings)
        panel_rows = len(panel)
        seal_ok = True
    except Exception as error:
        failures.append(f"{type(error).__name__}:{error}")
    prediction_exists = (settings.prediction_root / target_date).exists()
    reservation_exists = (settings.reservation_root / f"{target_date}.json").exists()
    try:
        rebalance, sequence = _is_rebalance_day(target_date, settings)
    except Exception:
        rebalance, sequence = False, None
    allowed = parent and policies and clock and seal_ok and not prediction_exists and not reservation_exists
    return {
        "target_date": target_date,
        "shanghai_now": now.astimezone(SHANGHAI).isoformat(),
        "human_freeze_intact": parent,
        "operational_lock_intact": parent,
        "time_gate_ready": clock,
        "prospective_start_valid": target_date >= "2026-09-01",
        "input_sealed": seal_ok,
        "input_hash_valid": seal_ok,
        "pit_valid": seal_ok,
        "universe_valid": seal_ok and panel_rows > 0,
        "model_policy_hashes_valid": policies,
        "calendar_valid": clock,
        "prediction_already_exists": prediction_exists,
        "reservation_exists": reservation_exists,
        "is_rebalance_day": rebalance,
        "rebalance_sequence": sequence,
        "settlement_maturity_pending_count": sum(1 for p in settings.prediction_root.glob("*/prediction.json") if not (settings.settlement_root / p.parent.name / "settlement.json").exists()),
        "daily_prediction_allowed": allowed,
        "status": "GEN2_DAILY_PREDICTION_ALLOWED" if allowed else "GEN2_INPUT_NOT_READY",
        "failures": failures,
        "provider_requests_made": 0,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }


def _default_train_and_score(target_date: str, settings: RuntimeSettings) -> tuple[pd.DataFrame, dict]:
    base = ChallengerSettings(
        dataset_path=settings.dataset_path,
        dataset_manifest_path=settings.dataset_manifest_path,
        factor_columns=_features(settings),
        training_row_cap=settings.training_row_cap_override or ChallengerSettings().training_row_cap,
    )
    verify_dataset_manifest(base)
    safe_features = tuple(base.factor_columns)
    assert_feature_columns_safe(safe_features)
    current, _ = _read_target_panel(target_date, settings)
    identity = ["date", "symbol", "broad_sector", "industry", "benchmark_weight", "benchmark_weight_rank"]
    target = pd.Timestamp(target_date)
    year_start = pd.Timestamp(target.year, 1, 1)
    validation_start = pd.Timestamp(target.year - 1, 1, 1)
    train_start = pd.Timestamp(target.year - settings.training_window_years - 1, 1, 1)
    training = pd.read_parquet(
        settings.dataset_path,
        columns=[*identity, *safe_features, "future_return_5d", "future_return_20d", "label_end_date_5d", "label_end_date_20d"],
        filters=[("date", ">=", train_start), ("date", "<", year_start), ("label_end_date_20d", "<", year_start)],
    )
    training["date"] = pd.to_datetime(training["date"])
    training["label_end_date_5d"] = pd.to_datetime(training["label_end_date_5d"])
    training["label_end_date_20d"] = pd.to_datetime(training["label_end_date_20d"])
    training = add_research_targets(training, (settings.selection_horizon, settings.horizon))
    dates = pd.DatetimeIndex(training["date"].drop_duplicates().sort_values())
    before_validation = dates[dates < validation_start]
    before_year = dates[dates < year_start]
    if len(before_validation) <= settings.selection_purge_gap or len(before_year) <= settings.purge_gap_trading_days:
        raise RuntimeError("INSUFFICIENT_PURGED_TRAINING_DATES")
    selection_cutoff = before_validation[-(settings.selection_purge_gap + 1)]
    refit_cutoff = before_year[-(settings.purge_gap_trading_days + 1)]
    selection_train = training[training["date"].le(selection_cutoff) & training["label_end_date_5d"].lt(validation_start)].copy()
    refit = training[training["date"].le(refit_cutoff) & training["label_end_date_20d"].lt(year_start)].copy()
    selection = select_factors_train_only(selection_train, base)
    features = selection.selected
    target_column = "return_rank_20d"
    finite = pd.to_numeric(refit[target_column], errors="coerce")
    refit = refit[finite.notna() & np.isfinite(finite)].copy()
    sample = deterministic_full_date_sample(refit, base.training_row_cap)
    processor = TrainOnlyPreprocessor().fit(sample, features)
    model = LightGBMModel("regression_l1", settings.lightgbm_rounds, settings.random_seed).fit(
        processor.transform(sample, features),
        pd.to_numeric(sample[target_column], errors="raise").to_numpy(dtype=float),
    )
    scored = current[identity].copy()
    scored["score"] = model.predict(processor.transform(current, features))
    label_end_max = pd.Timestamp(sample["label_end_date_20d"].max())
    if label_end_max >= year_start or label_end_max >= target:
        raise RuntimeError("TRAINING_LABEL_BOUNDARY_VIOLATION")
    return scored, {
        "model_signature": model.signature(),
        "training_snapshot_hash": sha256_bytes(canonical_frame_bytes(sample[["date", "symbol", target_column, *features]], ["date", "symbol"])),
        "training_rows": int(len(sample)),
        "training_date_min": str(sample["date"].min().date()),
        "training_date_max": str(sample["date"].max().date()),
        "training_label_date_min": str(sample["date"].min().date()),
        "training_label_date_max": str(sample["date"].max().date()),
        "training_label_years": sorted(int(v) for v in pd.DatetimeIndex(sample["date"]).year.unique()),
        "maximum_training_label_end": str(label_end_max.date()),
        "training_labels_all_mature_before_model_boundary": True,
        "labels_after_prediction_date_read": False,
        "current_prediction_outcome_read": False,
        "disqualified_2026_holdout_used_for_historical_confirmation": False,
        "historical_confirmation_attempted": False,
        "selected_features": list(features),
        "selected_features_hash": _policy_hash({"features": list(features)}),
        "input_snapshot_hash": sha256_bytes(canonical_frame_bytes(current, ["date", "symbol"])),
        "dataset_sha256": sha256_file(settings.dataset_path),
        "dataset_manifest_sha256": sha256_file(settings.dataset_manifest_path),
    }


def _rank_scores(scored: pd.DataFrame, settings: RuntimeSettings, rebalance: bool) -> pd.DataFrame:
    ranked = scored.sort_values(["score", "symbol"], ascending=[False, True]).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["selected_for_new_portfolio"] = False
    ranked["portfolio_weight"] = np.nan
    if rebalance:
        quotas = _sector_quotas(ranked, settings.top_k)
        selected = pd.concat(
            [ranked[ranked["broad_sector"].astype(str).eq(sector)].head(quota) for sector, quota in quotas.items()],
            ignore_index=False,
        ).sort_values(["score", "symbol"], ascending=[False, True]).head(settings.top_k)
        ranked.loc[selected.index, "selected_for_new_portfolio"] = True
        ranked.loc[selected.index, "portfolio_weight"] = 1.0 / len(selected)
    return ranked


def generate_prediction(
    target_date: str,
    *,
    now: datetime | None = None,
    settings: RuntimeSettings | None = None,
    scorer: Callable[[str, RuntimeSettings], tuple[pd.DataFrame, dict]] | None = None,
) -> dict:
    settings = settings or RuntimeSettings()
    now = now or datetime.now(timezone.utc)
    gate = preflight(target_date, now=now, settings=settings)
    if not gate["daily_prediction_allowed"]:
        raise RuntimeError(f"GEN2_INPUT_NOT_READY:{gate['failures']}")
    attempt_path = settings.reservation_root / f"{target_date}.json"
    write_atomic_reservation(attempt_path, {"date": target_date, "reserved_at_utc": _utc(now), "retry_allowed": False})
    scored, evidence = (scorer or _default_train_and_score)(target_date, settings)
    required = {"date", "symbol", "broad_sector", "industry", "benchmark_weight", "score"}
    if required - set(scored):
        raise RuntimeError("PREDICTION_SCORE_SCHEMA_INVALID")
    if scored["symbol"].astype(str).duplicated().any() or not np.isfinite(pd.to_numeric(scored["score"], errors="coerce")).all():
        raise RuntimeError("PREDICTION_SCORE_INVALID")
    is_rebalance, sequence = _is_rebalance_day(target_date, settings)
    ranked = _rank_scores(scored, settings, is_rebalance)
    spec = _verify_policy_hashes(settings)
    output = ranked[["date", "symbol", "industry", "broad_sector", "benchmark_weight", "score", "rank", "selected_for_new_portfolio", "portfolio_weight"]].copy()
    output["prediction_date"] = target_date
    output["portfolio_action"] = "REBALANCE" if is_rebalance else "HOLD"
    output["research_only"] = True
    output["production_prediction_ready"] = False
    output["execution_authorized"] = False
    directory = settings.prediction_root / target_date
    csv_hash = write_immutable_frame(directory / "prediction.csv", output, ["prediction_date", "symbol"])
    previous = sorted(settings.prediction_root.glob("*/manifest.json"))
    previous_hash = verify_immutable(previous[-1]) if previous else None
    seal_hash = verify_immutable(settings.input_seal_root / f"{target_date}.json")
    receipt = {
        "prediction_date": target_date,
        "created_at_utc": _utc(now),
        "model_id": spec["model_id"],
        "model_spec_hash": spec["model_spec_hash"],
        "feature_policy_hash": spec["feature_policy_hash"],
        "training_policy_hash": spec["training_policy_hash"],
        "portfolio_policy_hash": spec["portfolio_policy_hash"],
        "input_seal_sha256": seal_hash,
        "v6_comparison_reference": _v6_reference(target_date, settings),
        "training_evidence": evidence,
        "prediction_csv_sha256": csv_hash,
        "daily_score_rows": int(len(output)),
        "is_rebalance_day": is_rebalance,
        "rebalance_sequence": sequence,
        "rebalance_anchor_date": settings.rebalance_anchor_date,
        "portfolio_action": "REBALANCE" if is_rebalance else "HOLD",
        "active_portfolio_origin_date": target_date if is_rebalance else _latest_rebalance_date(settings),
        "selected_for_new_portfolio_count": int(output["selected_for_new_portfolio"].sum()),
        "label_maturity_date": label_end_session(target_date, settings),
        "maturity_status": "PENDING",
        "benchmark_status": "UNAPPROVED",
        "previous_prediction_manifest_hash": previous_hash,
        "future_label_fields_present": False,
        "research_only": True,
        "automatic_promotion_allowed": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    receipt_hash = write_immutable_json(directory / "prediction.json", receipt)
    manifest_hash = write_immutable_json(directory / "manifest.json", {"prediction.json": receipt_hash, "prediction.csv": csv_hash, "input_seal_sha256": seal_hash, "previous_prediction_manifest_hash": previous_hash})
    return receipt | {"manifest_sha256": manifest_hash}


def _latest_rebalance_date(settings: RuntimeSettings) -> str | None:
    dates = []
    for path in settings.prediction_root.glob("*/prediction.json"):
        value = read_verified_json(path)
        if value.get("portfolio_action") == "REBALANCE":
            dates.append(value["prediction_date"])
    return max(dates, default=None)


def _v6_reference(target_date: str, settings: RuntimeSettings) -> dict | None:
    """Cross-reference immutable V1r4 evidence without merging either chain."""
    path = settings.v1r4_input_evidence_root / f"{target_date}.json"
    if not path.is_file():
        return None
    value = read_verified_json(path)
    ranking = value.get("v6_ranking")
    if not isinstance(ranking, dict) or not ranking.get("sha256"):
        raise RuntimeError("V1R4_V6_REFERENCE_INVALID")
    return {
        "v1r4_input_evidence_sha256": verify_immutable(path),
        "v6_prediction_sha256": ranking["sha256"],
        "v6_prediction_date": ranking.get("prediction_date"),
        "evidence_chain_shared": False,
    }


def _market_witness(
    market_path: Path, effective_date: str
) -> tuple[pd.DataFrame, dict, list[dict]]:
    verify_immutable(market_path)
    witness_path = market_path.with_suffix(market_path.suffix + ".witness.json")
    witness = read_verified_json(witness_path)
    if witness.get("market_source_sha256") != sha256_file(market_path):
        raise RuntimeError("MARKET_WITNESS_HASH_MISMATCH")
    required = {"witnessed_at_utc", "source_created_at_utc", "acquisition_receipt_hash", "corporate_action_path", "corporate_action_sha256"}
    if required - set(witness):
        raise RuntimeError("SETTLEMENT_MARKET_WITNESS_INCOMPLETE")
    witnessed = pd.Timestamp(witness["witnessed_at_utc"])
    created = pd.Timestamp(witness["source_created_at_utc"])
    if witnessed.tzinfo is None or created.tzinfo is None or created > witnessed:
        raise RuntimeError("SETTLEMENT_MARKET_WITNESS_TIME_INVALID")
    if witnessed.tz_convert(SHANGHAI).date() > pd.Timestamp(effective_date).date():
        raise RuntimeError("SETTLEMENT_MARKET_WITNESS_FROM_FUTURE")
    action_path = Path(witness["corporate_action_path"])
    if sha256_file(action_path) != witness["corporate_action_sha256"]:
        raise RuntimeError("CORPORATE_ACTION_WITNESS_HASH_MISMATCH")
    events = json.loads(action_path.read_text(encoding="utf-8")).get("events", [])
    market = pd.read_csv(market_path, dtype={"symbol": str})
    needed = {"date", "symbol", "open", "close", "volume"}
    if needed - set(market):
        raise RuntimeError("SETTLEMENT_MARKET_SCHEMA_INVALID")
    market["date"] = pd.to_datetime(market["date"], errors="raise")
    market["symbol"] = market["symbol"].astype(str).str.zfill(6)
    if market.duplicated(["date", "symbol"]).any():
        raise RuntimeError("SETTLEMENT_MARKET_DUPLICATE")
    return market, witness, events


def _restore_ledger(book: PriceBook, state: dict | None, settings: RuntimeSettings) -> Ledger:
    ledger_settings = V20R2Settings(fee_rate=settings.commission, slippage=settings.slippage, stamp_duty=settings.sell_stamp_duty)
    ledger = Ledger(book, ledger_settings, charge_costs=True)
    if state:
        ledger.cash = float(state["cash"])
        ledger.units = {str(k): float(v) for k, v in state["units"].items()}
        ledger.applied = set(state.get("applied", []))
    return ledger


def settle_prediction(
    prediction_date: str,
    market_path: Path,
    *,
    now: datetime | None = None,
    test_as_of_override: str | None = None,
    settings: RuntimeSettings | None = None,
    official_alpha_requested: bool = False,
) -> dict:
    settings = settings or RuntimeSettings()
    now = now or datetime.now(timezone.utc)
    if official_alpha_requested:
        raise RuntimeError("OFFICIAL_ALPHA_BLOCKED_BENCHMARK_UNAPPROVED")
    if test_as_of_override is not None and not settings.test_mode:
        raise RuntimeError("PRODUCTION_AS_OF_OVERRIDE_FORBIDDEN")
    actual_date = now.astimezone(SHANGHAI).date().isoformat()
    effective_date = test_as_of_override if settings.test_mode and test_as_of_override else actual_date
    directory = settings.prediction_root / prediction_date
    receipt = read_verified_json(directory / "prediction.json")
    manifest = read_verified_json(directory / "manifest.json")
    if verify_immutable(directory / "prediction.csv") != manifest["prediction.csv"]:
        raise RuntimeError("PREDICTION_MANIFEST_HASH_MISMATCH")
    maturity = receipt["label_maturity_date"]
    if pd.Timestamp(actual_date) < pd.Timestamp(maturity) and not settings.test_mode:
        raise RuntimeError("20D_LABEL_NOT_MATURE")
    if pd.Timestamp(effective_date) < pd.Timestamp(maturity):
        raise RuntimeError("20D_LABEL_NOT_MATURE")
    market, witness, events = _market_witness(market_path, effective_date)
    calendar = load_verified_calendar(settings.calendar_path)
    later = calendar.sessions()[calendar.sessions() > pd.Timestamp(prediction_date)]
    entry_date, exit_date = later[0], later[settings.horizon]
    if market["date"].max() < exit_date:
        raise RuntimeError("SETTLEMENT_MARKET_NOT_MATURE")
    if market["date"].max() > pd.Timestamp(effective_date):
        raise RuntimeError("SETTLEMENT_MARKET_CONTAINS_UNWITNESSED_FUTURE")
    predictions = pd.read_csv(directory / "prediction.csv", dtype={"symbol": str})
    entry = market[market["date"].eq(entry_date)].set_index("symbol")["open"]
    exit_ = market[market["date"].eq(exit_date)].set_index("symbol")["open"]
    rows = predictions[["symbol", "score", "rank", "selected_for_new_portfolio", "portfolio_weight", "benchmark_weight"]].copy()
    rows["prediction_date"] = prediction_date
    rows["entry_date"] = str(entry_date.date())
    rows["exit_date"] = str(exit_date.date())
    rows["entry_open"] = rows["symbol"].map(entry)
    rows["exit_open"] = rows["symbol"].map(exit_)
    rows["actual_return_20d"] = rows["exit_open"] / rows["entry_open"] - 1.0
    rows["settled"] = rows[["entry_open", "exit_open"]].notna().all(axis=1)
    valid = rows[rows["settled"]].copy()
    weights = pd.to_numeric(valid["benchmark_weight"], errors="coerce")
    if not np.isfinite(weights).all() or weights.le(0).any():
        raise RuntimeError("RESEARCH_PROXY_BENCHMARK_WEIGHT_INVALID")
    research_proxy = float(np.average(valid["actual_return_20d"], weights=weights))
    portfolio_metrics = _settle_portfolio(
        receipt, predictions, market, events, entry_date, exit_date, research_proxy, settings
    )
    target = settings.settlement_root / prediction_date / "settlement.csv"
    csv_hash = write_immutable_frame(target, rows, ["prediction_date", "symbol"])
    rank_ic = valid["score"].corr(valid["actual_return_20d"], method="spearman") if len(valid) >= 3 else np.nan
    summary = {
        "prediction_date": prediction_date,
        "maturity_date": maturity,
        "settlement_status": "SETTLED_RESEARCH_PROXY_ONLY",
        "settled_symbols": int(len(valid)),
        "rank_ic": None if pd.isna(rank_ic) else float(rank_ic),
        "research_proxy_return": research_proxy,
        "research_proxy_semantics": "PIT_BENCHMARK_CONSTITUENT_WEIGHTED",
        **portfolio_metrics,
        "official_benchmark_status": "UNAPPROVED",
        "official_alpha_status": "PENDING_BENCHMARK_APPROVAL",
        "prediction_manifest_sha256": verify_immutable(directory / "manifest.json"),
        "market_source_sha256": sha256_file(market_path),
        "market_witness_sha256": verify_immutable(market_path.with_suffix(market_path.suffix + ".witness.json")),
        "settlement_csv_sha256": csv_hash,
        "prediction_recomputed": False,
        "automatic_promotion_allowed": False,
        "execution_authorized": False,
    }
    digest = write_immutable_json(settings.settlement_root / prediction_date / "settlement.json", summary)
    return summary | {"settlement_json_sha256": digest}


def _settle_portfolio(receipt: dict, predictions: pd.DataFrame, market: pd.DataFrame, events: list[dict], entry_date: pd.Timestamp, exit_date: pd.Timestamp, research_proxy: float, settings: RuntimeSettings) -> dict:
    if receipt["portfolio_action"] != "REBALANCE":
        return {"portfolio_action": "HOLD", "portfolio_metrics_status": "NO_NEW_REBALANCE"}
    book = PriceBook(market, events)
    entry_index, exit_index = book.index(entry_date), book.index(exit_date)
    prior_states = sorted(
        path
        for path in settings.portfolio_root.glob("*/state.json")
        if path.parent.name < receipt["prediction_date"]
    )
    prior = read_verified_json(prior_states[-1]) if prior_states else None
    ledger = _restore_ledger(book, prior, settings)
    gross_state = None
    if prior:
        gross_state = {
            "cash": prior.get("gross_cash", prior["cash"]),
            "units": prior.get("gross_units", prior["units"]),
            "applied": prior.get("gross_applied", prior.get("applied", [])),
        }
    gross = _restore_ledger(book, gross_state, settings)
    gross.charge_costs = False
    desired = predictions.loc[predictions["selected_for_new_portfolio"].astype(bool)].set_index("symbol")["portfolio_weight"].astype(float).to_dict()
    nav_before = ledger.nav(entry_index)
    gross_before = gross.nav(entry_index)
    trade = ledger.rebalance(desired, entry_index)
    gross_trade = gross.rebalance(desired, entry_index)
    ledger.advance(entry_index, exit_index)
    gross.advance(entry_index, exit_index)
    nav_after = ledger.nav(exit_index)
    gross_after = gross.nav(exit_index)
    state = {
        "prediction_date": receipt["prediction_date"],
        "valuation_date": str(exit_date.date()),
        "cash": ledger.cash,
        "units": ledger.units,
        "applied": sorted(ledger.applied),
        "gross_cash": gross.cash,
        "gross_units": gross.units,
        "gross_applied": sorted(gross.applied),
        "previous_state_sha256": verify_immutable(prior_states[-1]) if prior_states else None,
    }
    state_hash = write_immutable_json(settings.portfolio_root / receipt["prediction_date"] / "state.json", state)
    cash_weight = ledger.cash / nav_after if nav_after > 0 else 1.0
    return {
        "portfolio_action": "REBALANCE",
        "gross_portfolio_return": gross_after / gross_before - 1.0,
        "net_portfolio_return": nav_after / nav_before - 1.0,
        "buy_turnover": float(trade["buy_turnover"]),
        "sell_turnover": float(trade["sell_turnover"]),
        "transaction_cost_rate": float(trade["transaction_cost"]),
        "commission_slippage_stamp_duty_applied": True,
        "cash_weight": float(cash_weight),
        "blocked_sell_count": sum(v.get("side") == "sell" for v in trade["blocked"]),
        "blocked_buy_count": sum(v.get("side") == "buy" for v in trade["blocked"]),
        "gross_research_proxy_alpha": float(gross_after / gross_before - 1.0 - research_proxy),
        "net_research_proxy_alpha": float(nav_after / nav_before - 1.0 - research_proxy),
        "maximum_sector_weight": _maximum_sector_weight(ledger, book, exit_index, predictions),
        "portfolio_state_sha256": state_hash,
    }


def _maximum_sector_weight(
    ledger: Ledger, book: PriceBook, index: int, predictions: pd.DataFrame
) -> float:
    nav = ledger.nav(index)
    sector_by_symbol = predictions.set_index("symbol")["broad_sector"].astype(str).to_dict()
    totals: dict[str, float] = {}
    for symbol, units in ledger.units.items():
        sector = sector_by_symbol.get(symbol, "UNKNOWN")
        totals[sector] = totals.get(sector, 0.0) + units * book.mark(symbol, index) / nav
    return float(max(totals.values(), default=0.0))


def freeze_amendment(settings: RuntimeSettings | None = None, *, now: datetime | None = None) -> dict:
    settings = settings or RuntimeSettings()
    now = now or datetime.now(timezone.utc)
    if AMENDMENT_009.exists() and any(AMENDMENT_009.iterdir()):
        raise RuntimeError("AMENDMENT_009_ALREADY_EXISTS")
    parent = _require_parent(settings)
    AMENDMENT_009.mkdir(parents=True, exist_ok=True)
    protocol = {
        "amendment_id": "GEN02-PROSPECTIVE-RUNTIME-HARDENING-009",
        "classification": "PROSPECTIVE_RUNTIME_CORRECTNESS_ONLY",
        "parent_008_lock_sha256": parent,
        "human_decision_changed": False,
        "model_changed": False,
        "feature_policy_changed": False,
        "training_policy_changed": False,
        "portfolio_policy_changed": False,
        "cost_policy_changed": False,
        "prospective_start_date_changed": False,
        "historical_tuning_runs": 0,
        "provider_requests_during_build": 0,
        "execution_authorized": False,
        "production_prediction_ready": False,
    }
    audit = {
        "default_scorer_horizons": [settings.selection_horizon, settings.horizon],
        "input_flow": "INPUT_READY->SEAL->PREFLIGHT->RESERVE->PREDICT",
        "earliest_prediction_time_shanghai": settings.earliest_prediction_time.isoformat(timespec="minutes"),
        "rebalance_anchor_date": settings.rebalance_anchor_date,
        "portfolio_cadence_trading_days": settings.rebalance_trading_days,
        "research_proxy_semantics": "PIT_BENCHMARK_CONSTITUENT_WEIGHTED",
        "production_as_of_override_allowed": False,
        "first_real_prediction_generated": False,
        "provider_requests": {"market": 0, "financial": 0, "benchmark": 0},
    }
    before_after = {
        "default_scorer": {"before": "missing horizons argument", "after": "explicit 5D selection and 20D training targets"},
        "inputs": {"before": "live cache read", "after": "time-gated immutable seal bound to prediction"},
        "settlement": {"before": "caller as_of authoritative", "after": "actual Shanghai clock authoritative; override test-only"},
        "portfolio": {"before": "daily Top20 pseudo-portfolio", "after": "daily scores plus 20-session anchored stateful portfolio"},
        "proxy": {"before": "equal-weight universe mean", "after": "PIT benchmark constituent-weighted proxy"},
        "labels": {"before": "hard-coded 2026 flag", "after": "dynamic training label date provenance"},
    }
    hashes = {
        "protocol_amendment.json": write_immutable_json(AMENDMENT_009 / "protocol_amendment.json", protocol),
        "audit.json": write_immutable_json(AMENDMENT_009 / "audit.json", audit),
        "before_after.json": write_immutable_json(AMENDMENT_009 / "before_after.json", before_after),
    }
    files = [
        Path("stockpilot/research_challenger/prospective_gen2_runtime.py"),
        Path("tests/test_research_challenger_gen2_runtime_009.py"),
        settings.parent_008_lock_path,
        AMENDMENT_009 / "protocol_amendment.json",
        AMENDMENT_009 / "audit.json",
        AMENDMENT_009 / "before_after.json",
    ]
    lock = {
        "lock_id": "GEN02-PROSPECTIVE-RUNTIME-HARDENING-009",
        "created_at_utc": _utc(now),
        "parent_008_lock_sha256": parent,
        "files": {p.as_posix(): sha256_file(p) for p in files},
        "production_prediction_ready": False,
        "execution_authorized": False,
        "first_real_prediction_generated": False,
    }
    lock_hash = write_immutable_json(AMENDMENT_009 / "plan.lock.json", lock)
    manifest = {**hashes, "plan.lock.json": lock_hash}
    manifest_hash = write_immutable_json(AMENDMENT_009 / "artifact_manifest.json", manifest)
    return {"status": "PROSPECTIVE_RUNTIME_CORRECTNESS_ONLY_FROZEN", "lock_sha256": lock_hash, "artifact_manifest_sha256": manifest_hash}


def verify_amendment(settings: RuntimeSettings | None = None) -> dict:
    settings = settings or RuntimeSettings()
    lock = read_verified_json(settings.runtime_lock_path)
    mismatches = []
    for name, expected in lock["files"].items():
        path = Path(name)
        if not path.is_file() or sha256_file(path) != expected:
            mismatches.append(name)
    return {"intact": not mismatches, "mismatches": mismatches, "lock_sha256": verify_immutable(settings.runtime_lock_path), "v6_champion": True, "gen2_promoted": False, "production_prediction_ready": False, "execution_authorized": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gen2 prospective runtime hardening 009")
    sub = parser.add_subparsers(dest="command", required=True)
    seal = sub.add_parser("seal-inputs")
    seal.add_argument("--date", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--date", required=True)
    pred = sub.add_parser("predict")
    pred.add_argument("--date", required=True)
    settle = sub.add_parser("settle")
    settle.add_argument("--date", required=True)
    settle.add_argument("--market", required=True, type=Path)
    sub.add_parser("status")
    sub.add_parser("freeze-009")
    sub.add_parser("verify-009")
    args = parser.parse_args(argv)
    if args.command == "seal-inputs":
        result = seal_inputs(args.date)
    elif args.command == "preflight":
        result = preflight(args.date)
    elif args.command == "predict":
        result = generate_prediction(args.date)
    elif args.command == "settle":
        result = settle_prediction(args.date, args.market)
    elif args.command == "freeze-009":
        result = freeze_amendment()
    elif args.command == "verify-009":
        result = verify_amendment()
    else:
        result = review_checkpoint()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
