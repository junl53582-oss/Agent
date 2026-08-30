from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import OperationalSettings
from .integrity import (
    canonical_json_bytes,
    read_verified_json,
    sha256_bytes,
    sha256_file,
    verify_immutable,
    write_atomic_reservation,
    write_immutable_bytes,
    write_immutable_frame,
    write_immutable_json,
)


SOURCE_STATUSES = {
    "SUCCESS",
    "EMPTY_SUCCESS",
    "REQUEST_FAILED",
    "SOURCE_UNAVAILABLE",
    "VALIDATION_FAILED",
    "HASH_VERIFICATION_FAILED",
    "PARTIAL_SOURCE_FAILURE",
    "SCHEMA_MISMATCH",
    "CONFLICTING_DUPLICATE",
}


class SourceUnavailableError(RuntimeError):
    pass


class PartialSourceFailureError(RuntimeError):
    pass


class ConflictingDuplicateError(ValueError):
    pass


@dataclass(frozen=True)
class SourceCapture:
    source: str
    request_parameters: dict
    raw_payloads: tuple[bytes, ...]
    normalized: pd.DataFrame
    normalized_keys: tuple[str, ...] = ("symbol",)
    required_value_columns: tuple[str, ...] = ()
    duplicate_count: int = 0
    conflicting_duplicate_count: int = 0
    missing_symbols: tuple[str, ...] = ()
    confirmed_symbols: tuple[str, ...] = ()
    network_request_count: int = 0


def git_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN"


def universe_hash(symbols: set[str]) -> str:
    return sha256_bytes("\n".join(sorted(symbols)).encode("ascii"))


def reserve_daily_attempt(
    target_date: str,
    observed_at: datetime,
    *,
    parent_lock_sha256: str,
    settings: OperationalSettings,
) -> dict:
    if observed_at.tzinfo is None:
        raise ValueError("observed_at must be timezone-aware")
    attempt_id = observed_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    receipt = {
        "target_date": target_date,
        "reserved_at": observed_at.astimezone(timezone.utc).isoformat(),
        "observation_attempt_id": attempt_id,
        "git_commit": git_commit_sha(),
        "parent_lock_sha256": parent_lock_sha256,
        "status": "ATTEMPT_RESERVED_BEFORE_NETWORK",
        "retry_allowed": False,
        "automatic_retry": False,
        "manual_retry": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    path = settings.attempts_root / f"{target_date}.json"
    try:
        receipt["reservation_sha256"] = write_atomic_reservation(path, receipt)
    except FileExistsError as error:
        raise RuntimeError(f"daily attempt already reserved for {target_date}") from error
    return receipt


def _deduplicate(capture: SourceCapture) -> pd.DataFrame:
    frame = capture.normalized.copy()
    keys = list(capture.normalized_keys)
    missing = set(keys) - set(frame.columns)
    if not keys or missing:
        raise ValueError(f"normalized identity keys missing: {sorted(missing)}")
    duplicate_mask = frame.duplicated(keys, keep=False)
    actual_duplicate_rows = int(duplicate_mask.sum() - frame.loc[duplicate_mask, keys].drop_duplicates().shape[0])
    if not duplicate_mask.any():
        if capture.conflicting_duplicate_count:
            raise ConflictingDuplicateError("conflicting duplicate count has no matching rows")
        return frame
    conflicts = 0
    for _, group in frame.loc[duplicate_mask].groupby(keys, dropna=False):
        records = {sha256_bytes(canonical_json_bytes(row)) for row in group.to_dict("records")}
        conflicts += int(len(records) > 1)
    if conflicts or capture.conflicting_duplicate_count:
        raise ConflictingDuplicateError("conflicting duplicate provider records")
    if capture.duplicate_count not in (0, actual_duplicate_rows):
        raise ValueError("duplicate count disagrees with normalized rows")
    return frame.drop_duplicates(keys, keep="first")


def _validate_capture(capture: SourceCapture) -> pd.DataFrame:
    frame = _deduplicate(capture)
    missing = set(capture.required_value_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"normalized required value columns missing: {sorted(missing)}")
    if len(frame) and capture.required_value_columns:
        if frame[list(capture.required_value_columns)].notna().sum().sum() == 0:
            raise ValueError("non-empty normalized data has no available required values")
    if capture.confirmed_symbols:
        confirmed = {str(value).zfill(6) for value in capture.confirmed_symbols}
        unexpected = set(frame["symbol"].astype(str).str.zfill(6)) - confirmed
        if unexpected:
            raise ValueError("normalized rows contain symbols outside confirmed queries")
    return frame


def classify_failure(error: Exception) -> str:
    if isinstance(error, SourceUnavailableError):
        return "SOURCE_UNAVAILABLE"
    if isinstance(error, PartialSourceFailureError):
        return "PARTIAL_SOURCE_FAILURE"
    if isinstance(error, ConflictingDuplicateError):
        return "CONFLICTING_DUPLICATE"
    if isinstance(error, (KeyError, TypeError)):
        return "SCHEMA_MISMATCH"
    if "hash" in str(error).lower():
        return "HASH_VERIFICATION_FAILED"
    if isinstance(error, (ValueError, AssertionError)):
        return "VALIDATION_FAILED"
    return "REQUEST_FAILED"


def _write_source_success(
    directory: Path,
    capture: SourceCapture,
    universe: set[str],
) -> dict:
    frame = _validate_capture(capture)
    source_dir = directory / "sources" / capture.source
    raw_hashes = [
        write_immutable_bytes(source_dir / "raw" / f"page_{index:04d}.bin", payload)
        for index, payload in enumerate(capture.raw_payloads, 1)
    ]
    normalized_path = source_dir / "normalized.csv"
    normalized_hash = write_immutable_frame(
        normalized_path, frame, list(capture.normalized_keys)
    )
    covered = (
        set(frame["symbol"].astype(str).str.zfill(6)) if "symbol" in frame.columns else set()
    )
    confirmed = {str(value).zfill(6) for value in capture.confirmed_symbols}
    coverage_set = confirmed if confirmed else covered
    receipt = {
        "source": capture.source,
        "source_status": "SUCCESS" if len(frame) else "EMPTY_SUCCESS",
        "request_parameters": capture.request_parameters,
        "raw_response_sha256": raw_hashes,
        "raw_paths": [
            (source_dir / "raw" / f"page_{index:04d}.bin").as_posix()
            for index in range(1, len(raw_hashes) + 1)
        ],
        "normalized_data_sha256": normalized_hash,
        "normalized_path": normalized_path.as_posix(),
        "row_count": int(len(frame)),
        "universe_coverage": float(len(coverage_set & universe) / len(universe)) if universe else None,
        "confirmed_symbol_count": len(confirmed),
        "duplicate_count": int(capture.duplicate_count),
        "conflicting_duplicate_count": int(capture.conflicting_duplicate_count),
        "missing_symbols": sorted(universe - coverage_set),
        "success": True,
        "failure_reason": None,
        "silent_fallback_used": False,
        "network_request_count": int(capture.network_request_count),
        "hashes_verified": True,
    }
    receipt_path = source_dir / "receipt.json"
    write_immutable_json(receipt_path, receipt)
    return receipt | {"receipt_path": receipt_path.as_posix(), "receipt_sha256": verify_immutable(receipt_path)}


def _write_source_failure(
    directory: Path,
    source: str,
    params: dict,
    error: Exception,
) -> dict:
    receipt = {
        "source": source,
        "source_status": classify_failure(error),
        "request_parameters": params,
        "raw_response_sha256": [],
        "normalized_data_sha256": None,
        "normalized_path": None,
        "row_count": 0,
        "universe_coverage": None,
        "duplicate_count": 0,
        "conflicting_duplicate_count": 0,
        "missing_symbols": [],
        "success": False,
        "failure_reason": f"{type(error).__name__}: {error}",
        "silent_fallback_used": False,
        "network_request_count": int(getattr(error, "network_request_count", 0)),
        "hashes_verified": False,
    }
    path = directory / "sources" / source / "failure.json"
    write_immutable_json(path, receipt)
    return receipt | {"receipt_path": path.as_posix(), "receipt_sha256": verify_immutable(path)}


def verify_source_receipt(receipt: dict) -> bool:
    if receipt.get("source_status") not in SOURCE_STATUSES:
        return False
    if not receipt.get("success"):
        return True
    try:
        if verify_immutable(receipt["receipt_path"]) != receipt["receipt_sha256"]:
            return False
        if verify_immutable(receipt["normalized_path"]) != receipt["normalized_data_sha256"]:
            return False
        for path, digest in zip(receipt["raw_paths"], receipt["raw_response_sha256"]):
            if verify_immutable(path) != digest:
                return False
    except (OSError, KeyError, RuntimeError):
        return False
    return True


def capture_sources_once(
    *,
    attempt: dict,
    target_date: str,
    observed_at: datetime,
    universe: set[str],
    source_fetchers: dict[str, tuple[dict, Callable[[], SourceCapture]]],
    membership_snapshot_hash: str,
    industry_mapping_hash: str,
    trading_calendar_hash: str,
    settings: OperationalSettings,
) -> dict:
    observation_id = attempt["observation_attempt_id"]
    directory = settings.observations_root / observation_id
    directory.mkdir(parents=True, exist_ok=False)
    receipts: dict[str, dict] = {}
    for source, (params, fetcher) in source_fetchers.items():
        try:
            capture = fetcher()
            if capture.source != source:
                raise ValueError("source identity changed; silent fallback is forbidden")
            receipts[source] = _write_source_success(directory, capture, universe)
        except Exception as error:
            receipts[source] = _write_source_failure(directory, source, params, error)
    successful = [name for name, item in receipts.items() if item["success"]]
    failed = [name for name, item in receipts.items() if not item["success"]]
    record = {
        "observation_id": observation_id,
        "target_date": target_date,
        "observed_at": observed_at.astimezone(timezone.utc).isoformat(),
        "attempt_path": (settings.attempts_root / f"{target_date}.json").as_posix(),
        "attempt_sha256": sha256_file(settings.attempts_root / f"{target_date}.json"),
        "sources": receipts,
        "status": "SUCCESS" if not failed else ("PARTIAL" if successful else "FAILED"),
        "universe_hash": universe_hash(universe),
        "pit_membership_snapshot_hash": membership_snapshot_hash,
        "pit_industry_mapping_hash": industry_mapping_hash,
        "trading_calendar_hash": trading_calendar_hash,
        "verified_shanghai_trading_date": True,
        "observation_immutable_verified": True,
        "universe_size": len(universe),
        "code_commit_sha": git_commit_sha(),
        "lock_sha256": sha256_file(settings.plan_lock_path) if settings.plan_lock_path.exists() else None,
        "network_request_count": sum(item.get("network_request_count", 0) for item in receipts.values()),
        "automatic_retry": False,
        "manual_retry": False,
        "model_training_ready": False,
        "replacement_evaluation_ready": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    path = directory / "observation.json"
    write_immutable_json(path, record)
    verified = read_verified_json(path)
    verified["observation_path"] = path.as_posix()
    verified["observation_sha256"] = verify_immutable(path)
    return verified


def load_verified_observations(settings: OperationalSettings) -> list[dict]:
    records: list[dict] = []
    for path in sorted(settings.observations_root.glob("*/observation.json")):
        record = read_verified_json(path)
        record["observation_path"] = path.as_posix()
        record["observation_sha256"] = verify_immutable(path)
        records.append(record)
    return records
