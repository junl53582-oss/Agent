"""011 activation wrapper binding immutable daily PIT partitions to frozen Gen2 009."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stockpilot.prospective_r2.integrity import (
    canonical_frame_bytes,
    read_verified_json,
    sha256_bytes,
    sha256_file,
    verify_immutable,
)
from stockpilot.research_challenger import prospective_gen2_runtime as runtime009
from stockpilot.research_challenger import prospective_gen2_runtime_locked as runtime010r3
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
from stockpilot.research_challenger.prospective_gen2 import (
    CORRECTNESS_INTERPRETATION_LOCK,
    V1R4_LOCK,
    V6_LOCK,
    _policy_hash,
)

from .pipeline import DailyPitSettings, policy_hashes, verify_daily_feature_partition

ACTIVATION_DIR = runtime009.AMENDMENT_009 / "experiments/011_daily_pit_feature_activation"
ACTIVATION_LOCK = ACTIVATION_DIR / "plan.lock.json"


class DailyActivationLockError(RuntimeError):
    """Raised before any operational side effect when 011 is not fully intact."""


@dataclass(frozen=True)
class DailyRuntimeSettings(runtime009.RuntimeSettings):
    daily_input_root: Path = Path("data/prospective_gen2/daily_inputs")
    daily_activation_lock_path: Path = ACTIVATION_LOCK
    historical_dataset_path: Path = Path("artifacts/prediction_v30/cache/eligible_panel.parquet")
    historical_dataset_manifest_path: Path = Path("artifacts/prediction_v30/cache/manifest.json")

    def pit_settings(self) -> DailyPitSettings:
        return DailyPitSettings(
            root=self.daily_input_root,
            calendar_path=self.calendar_path,
        )


def _verify_lock_surface(path: Path) -> dict:
    digest = verify_immutable(path)
    payload = read_verified_json(path)
    mismatches: list[str] = []
    for name, expected in payload.get("files", {}).items():
        candidate = Path(name)
        if not candidate.is_file() or sha256_file(candidate) != expected:
            mismatches.append(name)
    return {
        "intact": not mismatches,
        "mismatches": mismatches,
        "sha256": digest,
        "payload": payload,
    }


def verify_daily_activation(settings: DailyRuntimeSettings | None = None) -> dict:
    settings = settings or DailyRuntimeSettings()
    result = _verify_lock_surface(settings.daily_activation_lock_path)
    payload = result.pop("payload")
    mismatches = result["mismatches"]
    if payload.get("lock_id") != "GEN02-DAILY-PIT-FEATURE-ACTIVATION-011":
        mismatches.append("LOCK_ID")
    parent_010 = verify_immutable(runtime010r3.ACTIVATION_LOCK)
    parent_009 = verify_immutable(settings.runtime_lock_path)
    parent_008 = verify_immutable(settings.parent_008_lock_path)
    parent_007 = verify_immutable(settings.human_lock_path)
    correctness = verify_immutable(runtime010r3.CORRECTNESS_LOCK)
    interpretation = verify_immutable(Path(CORRECTNESS_INTERPRETATION_LOCK))
    v1r4 = verify_immutable(Path(V1R4_LOCK))
    v6 = sha256_file(Path(V6_LOCK))
    expected_parents = {
        "human_007_lock_sha256": parent_007,
        "runtime_010r3_lock_sha256": parent_010,
        "runtime_009_lock_sha256": parent_009,
        "parent_008_lock_sha256": parent_008,
        "correctness_lock_sha256": correctness,
        "interpretation_lock_sha256": interpretation,
        "v1r4_lock_sha256": v1r4,
        "v6_lock_sha256": v6,
    }
    for key, expected in expected_parents.items():
        if payload.get(key) != expected:
            mismatches.append(key)
    expected_policies = policy_hashes(settings.pit_settings())
    for key, expected in expected_policies.items():
        if payload.get(key) != expected:
            mismatches.append(key)
    result.update(
        {
            "intact": not mismatches,
            "daily_011_lock_intact": not mismatches,
            "daily_011_lock_sha256": result["sha256"],
            "provider_requests_made": 0,
            "production_prediction_ready": False,
            "execution_authorized": False,
        }
    )
    return result


def verify_effective_daily_runtime_freeze(
    settings: DailyRuntimeSettings | None = None,
) -> dict:
    settings = settings or DailyRuntimeSettings()
    base = runtime010r3.verify_effective_runtime_freeze(settings)
    daily = verify_daily_activation(settings)
    failures = list(base.get("failures", []))
    if daily["intact"] is not True:
        failures.append(f"011:{daily['mismatches']}")
    effective = base.get("effective_operational_lock_intact") is True and not failures
    return {
        **base,
        **daily,
        "effective_daily_input_lock_intact": effective,
        "effective_operational_lock_intact": effective,
        "operational_lock_intact": effective,
        "failures": failures,
        "provider_requests_made": 0,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }


def _guard(
    settings: DailyRuntimeSettings,
    verifier: Callable[[DailyRuntimeSettings], dict] | None = None,
) -> dict:
    try:
        result = (verifier or verify_effective_daily_runtime_freeze)(settings)
    except Exception as error:
        raise DailyActivationLockError(
            f"GEN2_EFFECTIVE_DAILY_INPUT_LOCK_INVALID:{type(error).__name__}:{error}"
        ) from error
    if result.get("effective_daily_input_lock_intact") is not True:
        raise DailyActivationLockError(
            f"GEN2_EFFECTIVE_DAILY_INPUT_LOCK_INVALID:{result.get('failures', [])}"
        )
    return result


def _daily_009_settings(
    target_date: str, settings: DailyRuntimeSettings
) -> runtime009.RuntimeSettings:
    verify_daily_feature_partition(target_date, settings=settings.pit_settings())
    directory = settings.daily_input_root / target_date
    return replace(
        settings,
        dataset_path=directory / "panel.parquet",
        dataset_manifest_path=directory / "manifest.json",
    )


def seal_inputs(
    target_date: str,
    *,
    now: datetime | None = None,
    settings: DailyRuntimeSettings | None = None,
    effective_verifier: Callable[[DailyRuntimeSettings], dict] | None = None,
) -> dict:
    settings = settings or DailyRuntimeSettings()
    lock = _guard(settings, effective_verifier)
    result = runtime009.seal_inputs(
        target_date, now=now, settings=_daily_009_settings(target_date, settings)
    )
    return {**result, **lock, "daily_011_verified": True}


def preflight(
    target_date: str,
    *,
    now: datetime | None = None,
    settings: DailyRuntimeSettings | None = None,
    effective_verifier: Callable[[DailyRuntimeSettings], dict] | None = None,
) -> dict:
    settings = settings or DailyRuntimeSettings()
    try:
        lock = _guard(settings, effective_verifier)
        daily_settings = _daily_009_settings(target_date, settings)
    except Exception as error:  # noqa: BLE001 - preflight must return diagnostics
        return {
            "target_date": target_date,
            "daily_prediction_allowed": False,
            "status": "GEN2_EFFECTIVE_DAILY_INPUT_LOCK_INVALID",
            "failures": [f"{type(error).__name__}:{error}"],
            "provider_requests_made": 0,
            "production_prediction_ready": False,
            "execution_authorized": False,
        }
    return {**runtime009.preflight(target_date, now=now, settings=daily_settings), **lock}


def _daily_train_and_score(
    target_date: str,
    daily_settings: runtime009.RuntimeSettings,
    settings: DailyRuntimeSettings,
) -> tuple[pd.DataFrame, dict]:
    """Frozen 009 algorithm with split historical-training and sealed daily-current stores."""
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
    training = pd.read_parquet(
        settings.historical_dataset_path,
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
                sample[["date", "symbol", target_column, *features]], ["date", "symbol"]
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
        "split_store_adapter": "DAILY_PIT_INPUT_PIPELINE_V1",
    }


def generate_prediction(
    target_date: str,
    *,
    now: datetime | None = None,
    settings: DailyRuntimeSettings | None = None,
    effective_verifier: Callable[[DailyRuntimeSettings], dict] | None = None,
) -> dict:
    settings = settings or DailyRuntimeSettings()
    _guard(settings, effective_verifier)
    if target_date in settings.pit_settings().permanently_blocked_prediction_dates:
        raise RuntimeError("HISTORICAL_BACKFILL_FORBIDDEN:2026-09-01_PERMANENTLY_BLOCKED")
    daily_settings = _daily_009_settings(target_date, settings)
    return runtime009.generate_prediction(
        target_date,
        now=now,
        settings=daily_settings,
        scorer=lambda date, active: _daily_train_and_score(date, active, settings),
    )
