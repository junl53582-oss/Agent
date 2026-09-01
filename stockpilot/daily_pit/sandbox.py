"""Side-effect-free operational replay for the frozen DAILY PIT runtime.

This module is additive: it does not alter the production CLI or any 011-frozen
implementation.  Every mutable runtime root is bound beneath one unique sandbox
run directory before the production validation/scoring functions are called.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stockpilot.prospective_r2.integrity import (
    canonical_frame_bytes,
    canonical_json_bytes,
    read_verified_json,
    sha256_bytes,
    sha256_file,
    verify_immutable,
    write_immutable_bytes,
    write_immutable_json,
)
from stockpilot.research_challenger import prospective_gen2_runtime as runtime009
from stockpilot.research_challenger.config import ChallengerSettings
from stockpilot.research_challenger.data import (
    add_research_targets,
    assert_feature_columns_safe,
    verify_dataset_manifest,
)
from stockpilot.research_challenger.factors import select_factors_train_only
from stockpilot.research_challenger.models import (
    LightGBMModel,
    TrainOnlyPreprocessor,
    deterministic_full_date_sample,
)
from stockpilot.research_challenger.prospective_gen2 import _policy_hash

from . import runtime as daily_runtime
from .pipeline import (
    DAILY_FEATURE_COLUMNS,
    DailyPitSettings,
    _validate_daily_panel,
    acquire_market,
    policy_hashes,
    verify_daily_feature_partition,
)

SANDBOX_CONTRACT_DIR = Path(
    "artifacts/research_challenger/gen02/experiments/012_sandbox_replay_operational_contract"
)
SANDBOX_CONTRACT_LOCK = SANDBOX_CONTRACT_DIR / "plan.lock.json"
DEFAULT_SANDBOX_ROOT = Path("data/prospective_gen2_sandbox")
PRODUCTION_ROOTS = (
    Path("data/prospective_gen2/daily_inputs"),
    Path("data/prospective_gen2/input_seals"),
    Path("data/prospective_gen2/_prediction_attempts"),
    Path("data/prospective_gen2/predictions"),
    Path("data/prospective_gen2/settlements"),
)
REPLAY_FILES = (
    "market.csv",
    "panel.parquet",
    "historical_panel.parquet",
    "historical_manifest.json",
    "settlement_market.csv",
    "corporate_actions.json",
    "membership.csv",
)
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class SandboxSafetyError(RuntimeError):
    """Raised before a sandbox operation could cross an isolation boundary."""


class ReplayContractError(RuntimeError):
    """Raised when recorded replay evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class ReplayEvidence:
    root: Path
    manifest: dict[str, Any]
    manifest_sha256: str

    def path(self, name: str) -> Path:
        return self.root / name


@dataclass(frozen=True)
class SandboxRunPaths:
    sandbox_root: Path
    run_dir: Path
    daily_input_root: Path
    input_seal_root: Path
    reservation_root: Path
    prediction_root: Path
    settlement_root: Path
    portfolio_root: Path
    audit_root: Path


class ReplayMarketBackend:
    """The only acquisition backend available to the sandbox orchestrator."""

    def __init__(self, evidence: ReplayEvidence) -> None:
        self.evidence = evidence
        self.provider_requests = 0

    def fetch(self, requested, start, end, **kwargs):
        del start, kwargs
        if end != self.evidence.manifest["target_date"]:
            raise ReplayContractError("REPLAY_TARGET_DATE_MISMATCH")
        frame = pd.read_csv(self.evidence.path("market.csv"), dtype={"symbol": str})
        requested_symbols = {str(value).zfill(6) for value in requested}
        replay_symbols = set(frame["symbol"].astype(str).str.zfill(6))
        if not requested_symbols.issubset(replay_symbols):
            raise ReplayContractError("REPLAY_MARKET_SYMBOL_MISSING")
        return frame, []

    def attempt_real_provider(self, *args, **kwargs):
        del args, kwargs
        raise SandboxSafetyError("SANDBOX_REAL_PROVIDER_FORBIDDEN")


class ForbiddenBrokerAdapter:
    """A hard guard proving that sandbox execution cannot submit an order."""

    request_count = 0

    def submit(self, *args, **kwargs):
        del args, kwargs
        raise SandboxSafetyError("SANDBOX_BROKER_ADAPTER_FORBIDDEN")


def _resolved(path: Path) -> Path:
    return path.resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _production_roots() -> tuple[Path, ...]:
    return tuple(_resolved(path) for path in PRODUCTION_ROOTS)


def validate_sandbox_root(path: str | Path) -> Path:
    candidate = _resolved(Path(path))
    if candidate == _resolved(Path.cwd()):
        raise SandboxSafetyError("SANDBOX_ROOT_MUST_NOT_BE_REPOSITORY_ROOT")
    for production in _production_roots():
        if candidate == production or _is_within(candidate, production):
            raise SandboxSafetyError(f"SANDBOX_PRODUCTION_ROOT_FORBIDDEN:{production}")
        if _is_within(production, candidate):
            raise SandboxSafetyError("SANDBOX_ROOT_MUST_NOT_CONTAIN_PRODUCTION_ROOTS")
    for ancestor in (candidate, *candidate.parents):
        if ancestor.exists() and ancestor.is_symlink():
            raise SandboxSafetyError(f"SANDBOX_SYMLINK_ROOT_FORBIDDEN:{ancestor}")
    return candidate


def _validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise SandboxSafetyError("SANDBOX_RUN_ID_INVALID")
    return run_id


def _paths(root: Path, run_id: str) -> SandboxRunPaths:
    run_dir = root / run_id
    values = SandboxRunPaths(
        sandbox_root=root,
        run_dir=run_dir,
        daily_input_root=run_dir / "daily_inputs",
        input_seal_root=run_dir / "input_seals",
        reservation_root=run_dir / "attempts",
        prediction_root=run_dir / "predictions",
        settlement_root=run_dir / "settlements",
        portfolio_root=run_dir / "portfolio",
        audit_root=run_dir / "audit",
    )
    for value in values.__dict__.values():
        resolved = _resolved(value)
        if value != root and not _is_within(resolved, root):
            raise SandboxSafetyError(f"SANDBOX_PATH_ESCAPE:{value}")
        if any(resolved == item or _is_within(resolved, item) for item in _production_roots()):
            raise SandboxSafetyError(f"SANDBOX_PRODUCTION_PATH_FORBIDDEN:{value}")
    return values


def verify_sandbox_contract() -> dict[str, Any]:
    digest = verify_immutable(SANDBOX_CONTRACT_LOCK)
    payload = read_verified_json(SANDBOX_CONTRACT_LOCK)
    if payload.get("lock_id") != "GEN02-SIDE-EFFECT-FREE-SANDBOX-REPLAY-012":
        raise SandboxSafetyError("SANDBOX_CONTRACT_LOCK_ID_INVALID")
    mismatches = []
    for name, expected in payload.get("files", {}).items():
        path = Path(name)
        if not path.is_file() or sha256_file(path) != expected:
            mismatches.append(name)
    if mismatches:
        raise SandboxSafetyError(f"SANDBOX_CONTRACT_MISMATCH:{mismatches}")
    if payload.get("parent_011_lock_sha256") != verify_immutable(daily_runtime.ACTIVATION_LOCK):
        raise SandboxSafetyError("SANDBOX_PARENT_011_MISMATCH")
    return {
        "sandbox_012_lock_intact": True,
        "sandbox_012_lock_sha256": digest,
        "parent_011_lock_sha256": payload["parent_011_lock_sha256"],
    }


def _validate_relative_file(name: str) -> None:
    value = Path(name)
    if value.is_absolute() or ".." in value.parts or value.as_posix() != name:
        raise ReplayContractError(f"REPLAY_FILE_PATH_INVALID:{name}")


def load_replay_evidence(source: str | Path, target_date: str) -> ReplayEvidence:
    root = _resolved(Path(source))
    if not root.is_dir():
        raise ReplayContractError(f"REPLAY_SOURCE_NOT_DIRECTORY:{root}")
    if any(root == value or _is_within(root, value) for value in _production_roots()):
        raise ReplayContractError("REPLAY_SOURCE_MUST_NOT_BE_PRODUCTION_STATE")
    manifest_path = root / "replay_manifest.json"
    try:
        manifest_sha256 = verify_immutable(manifest_path)
        manifest = read_verified_json(manifest_path)
    except Exception as error:
        raise ReplayContractError(f"REPLAY_MANIFEST_INVALID:{error}") from error
    if manifest.get("contract_version") != "DAILY_PIT_SANDBOX_REPLAY_V1":
        raise ReplayContractError("REPLAY_CONTRACT_VERSION_INVALID")
    if manifest.get("mode") != "SANDBOX_REPLAY_ONLY":
        raise ReplayContractError("REPLAY_MODE_INVALID")
    if manifest.get("target_date") != target_date:
        raise ReplayContractError("REPLAY_TARGET_DATE_MISMATCH")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(REPLAY_FILES):
        raise ReplayContractError("REPLAY_FILE_SET_INVALID")
    for name, expected in files.items():
        _validate_relative_file(name)
        path = root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ReplayContractError(f"REPLAY_HASH_MISMATCH:{name}")
    as_of = pd.Timestamp(manifest.get("as_of_timestamp"))
    if as_of.tzinfo is None or str(as_of.tz_convert("Asia/Shanghai").date()) != target_date:
        raise ReplayContractError("REPLAY_AS_OF_INVALID")
    if as_of.tz_convert("Asia/Shanghai").time() < time(18, 30):
        raise ReplayContractError("REPLAY_AS_OF_BEFORE_DATA_WINDOW")
    market = pd.read_csv(root / "market.csv", dtype={"symbol": str})
    dates = pd.to_datetime(market["date"], errors="raise").dt.normalize()
    if market.empty or dates.max() > pd.Timestamp(target_date) or not dates.eq(target_date).any():
        raise ReplayContractError("REPLAY_MARKET_PIT_VIOLATION")
    panel = pd.read_parquet(root / "panel.parquet")
    try:
        _validate_daily_panel(panel, target_date)
    except Exception as error:
        raise ReplayContractError(f"REPLAY_FEATURE_INVALID:{error}") from error
    settlement = pd.read_csv(root / "settlement_market.csv", dtype={"symbol": str})
    settlement_dates = pd.to_datetime(settlement["date"], errors="raise")
    settlement_as_of = pd.Timestamp(manifest.get("settlement_as_of_timestamp"))
    if settlement_as_of.tzinfo is None:
        raise ReplayContractError("REPLAY_SETTLEMENT_AS_OF_INVALID")
    if settlement_dates.max() > settlement_as_of.tz_convert("Asia/Shanghai").tz_localize(None):
        raise ReplayContractError("REPLAY_SETTLEMENT_FUTURE_WITNESS")
    return ReplayEvidence(root=root, manifest=manifest, manifest_sha256=manifest_sha256)


def _tree_snapshot(root: Path) -> dict[str, str]:
    if not root.exists():
        return {"__state__": "ABSENT"}
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _production_snapshot() -> dict[str, dict[str, str]]:
    return {path.as_posix(): _tree_snapshot(path) for path in PRODUCTION_ROOTS}


def _write_stage(paths: SandboxRunPaths, stage: str, value: dict[str, Any]) -> str:
    return write_immutable_json(paths.audit_root / "stages" / f"{stage}.json", value)


def _market_stage(
    evidence: ReplayEvidence,
    paths: SandboxRunPaths,
    target_date: str,
    now: datetime,
    run_id: str,
) -> tuple[dict[str, Any], DailyPitSettings]:
    settings = DailyPitSettings(
        root=paths.daily_input_root,
        calendar_path=daily_runtime.DailyRuntimeSettings().calendar_path,
        membership_path=evidence.path("membership.csv"),
        provider_cache_dir=paths.run_dir / "provider_cache_forbidden",
    )
    backend = ReplayMarketBackend(evidence)
    market = pd.read_csv(evidence.path("market.csv"), dtype={"symbol": str})
    symbols = sorted(set(market["symbol"].astype(str).str.zfill(6)))
    result = acquire_market(
        target_date,
        symbols,
        now=now,
        settings=settings,
        fetcher=backend.fetch,
    )
    if backend.provider_requests != 0 or result.get("provider_requests_made") != 0:
        raise SandboxSafetyError("SANDBOX_PROVIDER_REQUEST_DETECTED")
    stage = {
        "run_id": run_id,
        "stage": "SANDBOX_MARKET_EVIDENCE",
        "replay_manifest_sha256": evidence.manifest_sha256,
        "market_manifest_sha256": result["market_manifest_sha256"],
        "provider_requests": 0,
        "sandbox_only": True,
    }
    stage["stage_manifest_sha256"] = _write_stage(paths, "01_market", stage)
    return result | stage, settings


def _materialize_replayed_features(
    evidence: ReplayEvidence,
    paths: SandboxRunPaths,
    settings: DailyPitSettings,
    target_date: str,
    run_id: str,
    market_result: dict[str, Any],
) -> dict[str, Any]:
    directory = settings.date_dir(target_date)
    panel_path = directory / "panel.parquet"
    panel = pd.read_parquet(evidence.path("panel.parquet"))
    _validate_daily_panel(panel, target_date)
    panel_hash = write_immutable_bytes(panel_path, evidence.path("panel.parquet").read_bytes())
    replay_source_hashes = {
        evidence.path("market.csv").as_posix(): sha256_file(evidence.path("market.csv")),
        evidence.path("panel.parquet").as_posix(): sha256_file(evidence.path("panel.parquet")),
    }
    manifest = {
        "manifest_version": "DAILY_PIT_FEATURES_V1",
        "sandbox_contract_version": "DAILY_PIT_SANDBOX_REPLAY_V1",
        "target_date": target_date,
        "panel_sha256": panel_hash,
        "rows": len(panel),
        "symbols": int(panel["symbol"].nunique()),
        "columns": DAILY_FEATURE_COLUMNS,
        "column_count": len(DAILY_FEATURE_COLUMNS),
        "feature_count": len(DAILY_FEATURE_COLUMNS) - 10,
        "source_hashes": replay_source_hashes,
        "sandbox_market_manifest_sha256": market_result["market_manifest_sha256"],
        "replay_manifest_sha256": evidence.manifest_sha256,
        "membership_not_future": True,
        "fundamental_not_future": True,
        "industry_not_future": True,
        "future_market_used": False,
        "previous_day_substituted": False,
        "historical_training_parquet_modified": False,
        "prediction_created": False,
        "reservation_created": False,
        **policy_hashes(settings),
    }
    manifest_hash = write_immutable_json(directory / "manifest.json", manifest)
    verified = verify_daily_feature_partition(target_date, settings=settings)
    stage = {
        "run_id": run_id,
        "stage": "SANDBOX_FEATURE_MATERIALIZATION",
        "panel_sha256": panel_hash,
        "feature_manifest_sha256": manifest_hash,
        "rows": verified["rows"],
        "columns": verified["column_count"],
        "sandbox_only": True,
    }
    stage["stage_manifest_sha256"] = _write_stage(paths, "02_features", stage)
    return verified | stage


def _runtime_settings(
    evidence: ReplayEvidence, paths: SandboxRunPaths
) -> daily_runtime.DailyRuntimeSettings:
    settings = replace(
        daily_runtime.DailyRuntimeSettings(),
        data_root=paths.run_dir / "state",
        daily_input_root=paths.daily_input_root,
        input_seal_root=paths.input_seal_root,
        reservation_root=paths.reservation_root,
        prediction_root=paths.prediction_root,
        settlement_root=paths.settlement_root,
        portfolio_root=paths.portfolio_root,
        historical_dataset_path=evidence.path("historical_panel.parquet"),
        historical_dataset_manifest_path=evidence.path("historical_manifest.json"),
        test_mode=True,
    )
    _assert_runtime_roots(settings, paths)
    return settings


def _sandbox_train_and_score(
    target_date: str,
    daily_settings: runtime009.RuntimeSettings,
    settings: daily_runtime.DailyRuntimeSettings,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the frozen 011 split-store scorer with a deduplicated parquet projection.

    ``benchmark_weight_rank`` is both an identity field and a frozen candidate
    feature.  The 011 implementation requests it twice from historical parquet,
    which produces duplicate DataFrame labels on pandas.  This 012-only adapter
    keeps the complete feature set and all model semantics while projecting each
    physical parquet column once.  The frozen production module is untouched.
    """

    base = ChallengerSettings(
        dataset_path=settings.historical_dataset_path,
        dataset_manifest_path=settings.historical_dataset_manifest_path,
        factor_columns=runtime009._features(settings),
        training_row_cap=settings.training_row_cap_override
        or ChallengerSettings().training_row_cap,
    )
    training_manifest = verify_dataset_manifest(base)
    safe_features = tuple(base.factor_columns)
    assert_feature_columns_safe(safe_features)
    current, _ = runtime009._read_target_panel(target_date, daily_settings)
    identity = [
        "date",
        "symbol",
        "broad_sector",
        "industry",
        "benchmark_weight",
        "benchmark_weight_rank",
    ]
    target = pd.Timestamp(target_date)
    year_start = pd.Timestamp(target.year, 1, 1)
    validation_start = pd.Timestamp(target.year - 1, 1, 1)
    train_start = pd.Timestamp(target.year - settings.training_window_years - 1, 1, 1)
    projection = list(
        dict.fromkeys(
            [
                *identity,
                *safe_features,
                "future_return_5d",
                "future_return_20d",
                "label_end_date_5d",
                "label_end_date_20d",
            ]
        )
    )
    training = pd.read_parquet(
        settings.historical_dataset_path,
        columns=projection,
        filters=[
            ("date", ">=", train_start),
            ("date", "<", year_start),
            ("label_end_date_20d", "<", year_start),
        ],
    )
    training["date"] = pd.to_datetime(training["date"])
    training["label_end_date_5d"] = pd.to_datetime(training["label_end_date_5d"])
    training["label_end_date_20d"] = pd.to_datetime(training["label_end_date_20d"])
    training = add_research_targets(training, (settings.selection_horizon, settings.horizon))
    dates = pd.DatetimeIndex(training["date"].drop_duplicates().sort_values())
    before_validation = dates[dates < validation_start]
    before_year = dates[dates < year_start]
    if (
        len(before_validation) <= settings.selection_purge_gap
        or len(before_year) <= settings.purge_gap_trading_days
    ):
        raise RuntimeError("INSUFFICIENT_PURGED_TRAINING_DATES")
    selection_cutoff = before_validation[-(settings.selection_purge_gap + 1)]
    refit_cutoff = before_year[-(settings.purge_gap_trading_days + 1)]
    selection_train = training[
        training["date"].le(selection_cutoff) & training["label_end_date_5d"].lt(validation_start)
    ].copy()
    refit = training[
        training["date"].le(refit_cutoff) & training["label_end_date_20d"].lt(year_start)
    ].copy()
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
        "training_snapshot_hash": sha256_bytes(
            canonical_frame_bytes(
                sample[["date", "symbol", target_column, *features]],
                ["date", "symbol"],
            )
        ),
        "training_rows": len(sample),
        "training_date_min": str(sample["date"].min().date()),
        "training_date_max": str(sample["date"].max().date()),
        "maximum_training_label_end": str(label_end_max.date()),
        "training_labels_all_mature_before_model_boundary": True,
        "labels_after_prediction_date_read": False,
        "current_prediction_outcome_read": False,
        "disqualified_2026_holdout_used_for_historical_confirmation": False,
        "historical_confirmation_attempted": False,
        "selected_features": list(features),
        "selected_features_hash": _policy_hash({"features": list(features)}),
        "input_snapshot_hash": sha256_bytes(canonical_frame_bytes(current, ["date", "symbol"])),
        "historical_dataset_sha256": training_manifest["dataset_sha256"],
        "historical_dataset_manifest_sha256": training_manifest["manifest_sha256"],
        "daily_dataset_sha256": sha256_file(daily_settings.dataset_path),
        "daily_dataset_manifest_sha256": sha256_file(daily_settings.dataset_manifest_path),
        "split_store_adapter": "DAILY_PIT_INPUT_PIPELINE_V1_SANDBOX_012",
        "deduplicated_parquet_projection": True,
    }


def _generate_sandbox_prediction(
    target_date: str,
    now: datetime,
    settings: daily_runtime.DailyRuntimeSettings,
) -> dict[str, Any]:
    daily_runtime._guard(settings)
    if target_date in settings.pit_settings().permanently_blocked_prediction_dates:
        raise RuntimeError("HISTORICAL_BACKFILL_FORBIDDEN:PERMANENTLY_BLOCKED")
    daily_settings = daily_runtime._daily_009_settings(target_date, settings)
    return runtime009.generate_prediction(
        target_date,
        now=now,
        settings=daily_settings,
        scorer=lambda date, active: _sandbox_train_and_score(date, active, settings),
    )


def _assert_runtime_roots(
    settings: daily_runtime.DailyRuntimeSettings, paths: SandboxRunPaths
) -> None:
    mutable = {
        "daily_input_root": settings.daily_input_root,
        "input_seal_root": settings.input_seal_root,
        "reservation_root": settings.reservation_root,
        "prediction_root": settings.prediction_root,
        "settlement_root": settings.settlement_root,
        "portfolio_root": settings.portfolio_root,
        "data_root": settings.data_root,
    }
    for name, value in mutable.items():
        resolved = _resolved(Path(value))
        if not _is_within(resolved, paths.run_dir):
            raise SandboxSafetyError(f"SANDBOX_RUNTIME_ROOT_ESCAPE:{name}:{value}")
        if any(resolved == item or _is_within(resolved, item) for item in _production_roots()):
            raise SandboxSafetyError(f"SANDBOX_RUNTIME_PRODUCTION_ROOT:{name}:{value}")


def _candidate_gate_decision(
    prediction: pd.DataFrame,
    receipt: dict[str, Any],
    prediction_manifest_sha256: str,
    lock: dict[str, Any],
    decision_timestamp: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    scores = pd.to_numeric(prediction["score"], errors="coerce")
    ranks = pd.to_numeric(prediction["rank"], errors="coerce")
    eligible = scores.notna() & np.isfinite(scores) & ranks.notna() & np.isfinite(ranks)
    candidate_rows = []
    for index, row in prediction.sort_values(["rank", "symbol"]).iterrows():
        candidate_id = sha256_bytes(
            canonical_json_bytes(
                {
                    "prediction_manifest_sha256": prediction_manifest_sha256,
                    "symbol": str(row["symbol"]).zfill(6),
                }
            )
        )
        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "symbol": str(row["symbol"]).zfill(6),
                "rank": int(row["rank"]),
                "score": float(row["score"]),
                "eligible": bool(eligible.loc[index]),
                "selected": bool(row["selected_for_new_portfolio"]),
            }
        )
    failures = []
    if not bool(eligible.all()):
        failures.append("INVALID_CANDIDATE_SCORE_OR_RANK")
    if receipt.get("research_only") is not True:
        failures.append("RESEARCH_ONLY_REQUIRED")
    if receipt.get("execution_authorized") is not False:
        failures.append("EXECUTION_AUTHORIZATION_MUST_BE_FALSE")
    gate_state = "ACCEPTED_RESEARCH_ONLY" if not failures else "REJECTED_FAIL_CLOSED"
    candidate = {
        "count": len(candidate_rows),
        "ids": [row["candidate_id"] for row in candidate_rows],
        "rows": candidate_rows,
        "eligible_count": sum(row["eligible"] for row in candidate_rows),
        "selected_count": sum(row["selected"] for row in candidate_rows),
    }
    gate = {
        "state": gate_state,
        "accepted": not failures,
        "rejected": bool(failures),
        "reasons": failures or ["FROZEN_RESEARCH_ONLY_GATES_SATISFIED"],
        "thresholds": {"top_k": 20, "finite_scores_required": True},
        "lock_identity": lock["sandbox_012_lock_sha256"],
    }
    if failures:
        raise SandboxSafetyError(f"SANDBOX_CANDIDATE_GATE_REJECTED:{failures}")
    selected = [row for row in candidate_rows if row["selected"]]
    decision_body = {
        "action": receipt["portfolio_action"],
        "selected_candidate_ids": [row["candidate_id"] for row in selected],
        "weights": {
            str(row["symbol"]).zfill(6): float(row["portfolio_weight"])
            for _, row in prediction[prediction["selected_for_new_portfolio"]].iterrows()
        },
        "decision_timestamp": decision_timestamp,
        "source_prediction_hash": prediction_manifest_sha256,
        "gate_state": gate_state,
        "sandbox_only": True,
        "execution_authorized": False,
    }
    decision = decision_body | {"decision_id": sha256_bytes(canonical_json_bytes(decision_body))}
    return candidate, gate, decision


def _simulate_execution(decision: dict[str, Any]) -> dict[str, Any]:
    orders = [
        {
            "candidate_id": candidate_id,
            "direction": "BUY",
            "hypothetical_weight": weight,
            "quantity": None,
            "risk_result": "ELIGIBLE_RESEARCH_ONLY",
        }
        for candidate_id, weight in zip(
            decision["selected_candidate_ids"], decision["weights"].values()
        )
    ]
    return {
        "sandbox_only": True,
        "evaluated": True,
        "action": decision["action"],
        "hypothetical_orders": orders,
        "hypothetical_order_count": len(orders),
        "risk_checks_passed": True,
        "broker_requests": 0,
        "real_order_submitted": False,
        "real_trade_executed": False,
        "execution_authorized": False,
        "reject_reason": "SANDBOX_EXECUTION_FORBIDDEN",
    }


def _sandbox_settlement(
    evidence: ReplayEvidence,
    paths: SandboxRunPaths,
    settings: daily_runtime.DailyRuntimeSettings,
    target_date: str,
) -> tuple[dict[str, Any], str]:
    market_path = paths.run_dir / "settlement_input" / "market.csv"
    write_immutable_bytes(market_path, evidence.path("settlement_market.csv").read_bytes())
    actions_path = paths.run_dir / "settlement_input" / "corporate_actions.json"
    write_immutable_bytes(actions_path, evidence.path("corporate_actions.json").read_bytes())
    settlement_as_of = pd.Timestamp(evidence.manifest["settlement_as_of_timestamp"])
    source_created = settlement_as_of - pd.Timedelta(hours=1)
    witness = {
        "market_source_sha256": sha256_file(market_path),
        "witnessed_at_utc": settlement_as_of.tz_convert("UTC").isoformat(),
        "source_created_at_utc": source_created.tz_convert("UTC").isoformat(),
        "acquisition_receipt_hash": evidence.manifest_sha256,
        "corporate_action_path": actions_path.as_posix(),
        "corporate_action_sha256": sha256_file(actions_path),
        "sandbox_only": True,
    }
    write_immutable_json(market_path.with_suffix(market_path.suffix + ".witness.json"), witness)
    result = runtime009.settle_prediction(
        target_date,
        market_path,
        now=settlement_as_of.to_pydatetime(),
        test_as_of_override=str(settlement_as_of.tz_convert("Asia/Shanghai").date()),
        settings=settings,
    )
    semantic = {
        key: result.get(key)
        for key in (
            "prediction_date",
            "maturity_date",
            "settlement_status",
            "settled_symbols",
            "rank_ic",
            "research_proxy_return",
            "portfolio_action",
            "execution_authorized",
        )
    }
    return (
        {
            "sandbox_only": True,
            "eligibility_evaluated": True,
            "production_settlement_created": False,
            "duplicate_protection": "IMMUTABLE_RUN_LOCAL_ARTIFACT",
            "result": result,
            "semantic_hash": sha256_bytes(canonical_json_bytes(semantic)),
        },
        verify_immutable(settings.settlement_root / target_date / "settlement.json"),
    )


def run_sandbox_replay(
    target_date: str,
    replay_source: str | Path,
    *,
    sandbox_root: str | Path = DEFAULT_SANDBOX_ROOT,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Execute the complete replay lifecycle with no reachable production writer."""

    contract = verify_sandbox_contract()
    effective = daily_runtime.verify_effective_daily_runtime_freeze()
    if effective.get("effective_daily_input_lock_intact") is not True:
        raise SandboxSafetyError(f"SANDBOX_EFFECTIVE_LOCK_INVALID:{effective.get('failures')}")
    evidence = load_replay_evidence(replay_source, target_date)
    root = validate_sandbox_root(sandbox_root)
    identity = run_id or (f"{target_date}-{evidence.manifest_sha256[:12]}-{uuid.uuid4().hex[:12]}")
    identity = _validate_run_id(identity)
    paths = _paths(root, identity)
    if paths.run_dir.exists():
        raise SandboxSafetyError(f"SANDBOX_RUN_ALREADY_EXISTS:{identity}")
    before = _production_snapshot()
    paths.run_dir.mkdir(parents=True, exist_ok=False)
    now = pd.Timestamp(evidence.manifest["as_of_timestamp"]).to_pydatetime()

    market, pit_settings = _market_stage(evidence, paths, target_date, now, identity)
    features = _materialize_replayed_features(
        evidence, paths, pit_settings, target_date, identity, market
    )
    settings = _runtime_settings(evidence, paths)

    seal = daily_runtime.seal_inputs(target_date, now=now, settings=settings)
    seal_stage = {
        "run_id": identity,
        "stage": "SANDBOX_INPUT_SEAL",
        "seal_sha256": seal["seal_sha256"],
        "input_snapshot_hash": seal["input_snapshot_hash"],
        "effective_daily_input_lock_intact": seal["effective_daily_input_lock_intact"],
        "sandbox_only": True,
    }
    seal_stage["stage_manifest_sha256"] = _write_stage(paths, "03_seal", seal_stage)

    preflight = daily_runtime.preflight(target_date, now=now, settings=settings)
    if preflight.get("daily_prediction_allowed") is not True:
        raise SandboxSafetyError(f"SANDBOX_PREFLIGHT_REJECTED:{preflight.get('failures')}")
    prediction = _generate_sandbox_prediction(target_date, now, settings)
    prediction_manifest = settings.prediction_root / target_date / "manifest.json"
    prediction_hash = verify_immutable(prediction_manifest)
    prediction_frame = pd.read_csv(
        settings.prediction_root / target_date / "prediction.csv", dtype={"symbol": str}
    )
    prediction_stage = {
        "run_id": identity,
        "stage": "SANDBOX_PREDICTION",
        "prediction_manifest_sha256": prediction_hash,
        "prediction_rows": len(prediction_frame),
        "model_id": prediction["model_id"],
        "model_spec_hash": prediction["model_spec_hash"],
        "sandbox_attempt_sha256": sha256_file(settings.reservation_root / f"{target_date}.json"),
        "production_reservation_created": False,
        "sandbox_only": True,
        "execution_authorized": False,
    }
    prediction_stage["stage_manifest_sha256"] = _write_stage(
        paths, "04_prediction", prediction_stage
    )

    candidate, gate, decision = _candidate_gate_decision(
        prediction_frame,
        prediction,
        prediction_hash,
        contract,
        evidence.manifest["as_of_timestamp"],
    )
    for name, value in (("05_candidates", candidate), ("06_gate", gate), ("07_decision", decision)):
        value["run_id"] = identity
        value["sandbox_only"] = True
        value["stage_manifest_sha256"] = _write_stage(paths, name, value)

    execution = _simulate_execution(decision)
    execution["run_id"] = identity
    execution["stage_manifest_sha256"] = _write_stage(paths, "08_execution", execution)

    settlement, settlement_hash = _sandbox_settlement(evidence, paths, settings, target_date)
    settlement["run_id"] = identity
    settlement["sandbox_settlement_sha256"] = settlement_hash
    settlement["stage_manifest_sha256"] = _write_stage(paths, "09_settlement", settlement)

    after = _production_snapshot()
    if before != after:
        raise SandboxSafetyError("SANDBOX_PRODUCTION_STATE_CHANGED")
    baseline = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    normalized = {
        "input_hash": seal["input_snapshot_hash"],
        "feature_hash": features["panel_sha256"],
        "seal_hash": seal["seal_sha256"],
        "prediction_hash": prediction_hash,
        "prediction_scores_ranks_hash": sha256_bytes(
            prediction_frame[["symbol", "score", "rank"]]
            .sort_values("symbol")
            .to_csv(index=False, lineterminator="\n", float_format="%.12g")
            .encode("utf-8")
        ),
        "decision_hash": decision["decision_id"],
        "settlement_semantic_hash": settlement["semantic_hash"],
    }
    audit = {
        "manifest_version": "DAILY_PIT_SANDBOX_RUN_V1",
        "run_id": identity,
        "mode": "SIDE_EFFECT_FREE_SANDBOX_REPLAY",
        "target_date": target_date,
        "as_of_timestamp": evidence.manifest["as_of_timestamp"],
        "pit_cutoff": target_date,
        "baseline_git_sha": baseline,
        "effective_lock": {
            "daily_011_lock_sha256": effective["daily_011_lock_sha256"],
            **contract,
        },
        "replay_source": evidence.root.as_posix(),
        "replay_hash": evidence.manifest_sha256,
        "market": market,
        "features": features,
        "seal": seal_stage,
        "model": {
            "model_id": prediction["model_id"],
            "model_spec_hash": prediction["model_spec_hash"],
            "feature_policy_hash": prediction["feature_policy_hash"],
            "training_evidence": prediction["training_evidence"],
        },
        "prediction": prediction_stage,
        "candidate": candidate,
        "gate": gate,
        "decision": decision,
        "execution_simulation": execution,
        "settlement_simulation": settlement,
        "deterministic_outputs": normalized,
        "side_effects": {
            "provider_requests": 0,
            "production_write_count": 0,
            "production_daily_input_modified": False,
            "production_seal_created": False,
            "production_reservation_created": False,
            "production_prediction_created": False,
            "production_settlement_created": False,
            "broker_requests": 0,
            "real_order_submitted": False,
            "real_trade_executed": False,
            "real_promotion": False,
            "execution_authorized": False,
        },
        "final_status": "OPERATIONAL_DRY_RUN_PASSED",
    }
    audit_path = paths.audit_root / "sandbox_run_manifest.json"
    audit_hash = write_immutable_json(audit_path, audit)
    return {
        "status": "OPERATIONAL_DRY_RUN_PASSED",
        "run_id": identity,
        "sandbox_root": root.as_posix(),
        "sandbox_run_manifest": audit_path.as_posix(),
        "sandbox_run_manifest_sha256": audit_hash,
        "deterministic_outputs": normalized,
        "side_effects": audit["side_effects"],
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="StockPilot side-effect-free DAILY PIT sandbox replay"
    )
    value.add_argument("target_date")
    value.add_argument("--replay-source", required=True, type=Path)
    value.add_argument("--sandbox-root", type=Path, default=DEFAULT_SANDBOX_ROOT)
    value.add_argument("--run-id")
    return value


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    result = run_sandbox_replay(
        args.target_date,
        args.replay_source,
        sandbox_root=args.sandbox_root,
        run_id=args.run_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
