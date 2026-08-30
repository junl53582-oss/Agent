from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from stockpilot.prospective import ledger as parent
from stockpilot.prospective.immutable import sha256_file, write_new_json


@dataclass(frozen=True)
class SourceCapture(parent.SourceCapture):
    required_value_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class LedgerSettings(parent.LedgerSettings):
    data_root: Path = Path("data/prospective_alpha_v1r1/observations")
    lock_path: Path = Path("artifacts/prospective_alpha_v1r1/plan.lock.json")


def _attempt_path(settings: LedgerSettings, target_date: str) -> Path:
    return settings.data_root / "_attempts" / f"{target_date}.json"


def validate_observation_request(
    target_date: str,
    observed_at: datetime,
    trading_calendar: set[str],
    settings: LedgerSettings | None = None,
) -> None:
    settings = settings or LedgerSettings()
    parent.validate_observation_request(target_date, observed_at, trading_calendar, settings)
    if _attempt_path(settings, target_date).exists():
        raise RuntimeError(f"observation attempt already reserved for {target_date}")


def reserve_observation_request(
    target_date: str,
    observed_at: datetime,
    trading_calendar: set[str],
    settings: LedgerSettings | None = None,
) -> dict:
    """Atomically reserve a date before any provider callable can run."""
    settings = settings or LedgerSettings()
    validate_observation_request(target_date, observed_at, trading_calendar, settings)
    observation_id = observed_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    receipt = {
        "observation_id": observation_id,
        "target_date": target_date,
        "observed_at": observed_at.astimezone(timezone.utc).isoformat(),
        "status": "ATTEMPT_RESERVED_BEFORE_NETWORK",
        "automatic_retry": False,
        "model_training_ready": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    try:
        write_new_json(_attempt_path(settings, target_date), receipt)
    except FileExistsError as error:
        raise RuntimeError(f"observation attempt already reserved for {target_date}") from error
    return receipt


def _validate_capture(capture: SourceCapture) -> None:
    missing = set(capture.required_value_columns) - set(capture.normalized.columns)
    if missing:
        raise ValueError(f"normalized required value columns missing: {sorted(missing)}")
    if len(capture.normalized) and capture.required_value_columns:
        values = capture.normalized[list(capture.required_value_columns)]
        if values.notna().sum().sum() == 0:
            raise ValueError("non-empty normalized data has no available required values")


def observe_sources(
    *,
    target_date: str,
    observed_at: datetime,
    trading_calendar: set[str],
    universe: set[str],
    source_fetchers: dict[str, tuple[dict, Callable[[], SourceCapture]]],
    membership_snapshot_hash: str,
    industry_mapping_hash: str,
    settings: LedgerSettings | None = None,
) -> dict:
    settings = settings or LedgerSettings()
    attempt = reserve_observation_request(target_date, observed_at, trading_calendar, settings)
    observation_id = attempt["observation_id"]
    directory = settings.data_root / observation_id
    directory.mkdir(parents=True, exist_ok=False)
    receipts: dict[str, dict] = {}
    for source, (params, fetcher) in source_fetchers.items():
        try:
            capture = fetcher()
            if capture.source != source:
                raise ValueError("source identity changed; fallback is forbidden")
            _validate_capture(capture)
            receipts[source] = parent._record_source_success(directory, capture, universe)
        except BaseException as error:
            receipts[source] = parent._record_source_failure(directory, source, params, error)
    successful = [name for name, item in receipts.items() if item["success"]]
    failed = [name for name, item in receipts.items() if not item["success"]]
    attempt_path = _attempt_path(settings, target_date)
    record = {
        "observation_id": observation_id,
        "target_date": target_date,
        "observed_at": observed_at.astimezone(timezone.utc).isoformat(),
        "qualifying_trading_observation": True,
        "attempt_receipt_path": attempt_path.as_posix(),
        "attempt_receipt_sha256": sha256_file(attempt_path),
        "sources": receipts,
        "status": "SUCCESS" if not failed else ("PARTIAL" if successful else "FAILED"),
        "universe_hash": parent._universe_hash(universe),
        "pit_membership_snapshot_hash": membership_snapshot_hash,
        "pit_industry_mapping_hash": industry_mapping_hash,
        "universe_size": len(universe),
        "code_commit_sha": parent._commit_sha(),
        "lock_sha256": sha256_file(settings.lock_path),
        "model_training_ready": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    write_new_json(directory / "observation.json", record)
    return record


def load_observations(settings: LedgerSettings | None = None) -> list[dict]:
    settings = settings or LedgerSettings()
    return parent.load_observations(settings)


def import_verified_baseline(**kwargs) -> dict:
    kwargs.setdefault("settings", LedgerSettings())
    return parent.import_verified_baseline(**kwargs)
