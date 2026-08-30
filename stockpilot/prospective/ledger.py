from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

from .immutable import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    write_new_bytes,
    write_new_frame,
    write_new_json,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class SourceCapture:
    source: str
    request_parameters: dict
    raw_payloads: tuple[bytes, ...]
    normalized: pd.DataFrame
    normalized_keys: tuple[str, ...] = ("symbol",)
    duplicate_count: int = 0
    conflicting_duplicate_count: int = 0
    missing_symbols: tuple[str, ...] = ()


@dataclass(frozen=True)
class LedgerSettings:
    data_root: Path = Path("data/prospective_alpha_v1/observations")
    lock_path: Path = Path("artifacts/prospective_alpha_v1/plan.lock.json")
    membership_path: Path = Path("data/universes/000300/history_v10.csv")
    industry_path: Path = Path("data/industry_history_v10.csv")


def _commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN"


def _observation_files(settings: LedgerSettings) -> list[Path]:
    return sorted(settings.data_root.glob("*/observation.json"))


def load_observations(settings: LedgerSettings | None = None) -> list[dict]:
    settings = settings or LedgerSettings()
    return [json.loads(path.read_text(encoding="utf-8")) for path in _observation_files(settings)]


def validate_observation_request(
    target_date: str,
    observed_at: datetime,
    trading_calendar: set[str],
    settings: LedgerSettings | None = None,
) -> None:
    settings = settings or LedgerSettings()
    if observed_at.tzinfo is None:
        raise ValueError("observed_at must be timezone-aware")
    actual_date = observed_at.astimezone(SHANGHAI).date().isoformat()
    if target_date != actual_date:
        raise ValueError("historical backfill is forbidden")
    if target_date not in trading_calendar:
        raise ValueError("target date is not a verified Shanghai trading date")
    if any(item["target_date"] == target_date for item in load_observations(settings)):
        raise RuntimeError(f"observation attempt already exists for {target_date}")


def _universe_hash(symbols: set[str]) -> str:
    return sha256_bytes("\n".join(sorted(symbols)).encode("ascii"))


def _deduplicate(capture: SourceCapture) -> pd.DataFrame:
    frame = capture.normalized.copy()
    keys = list(capture.normalized_keys)
    if not keys or any(key not in frame for key in keys):
        raise ValueError("normalized identity keys are missing")
    duplicates = frame.duplicated(keys, keep=False)
    if not duplicates.any():
        if capture.conflicting_duplicate_count:
            raise ValueError("conflicting duplicate count disagrees with normalized rows")
        return frame
    conflicts = 0
    for _, group in frame.loc[duplicates].groupby(keys, dropna=False):
        records = {
            sha256_bytes(canonical_json_bytes(row))
            for row in group.to_dict(orient="records")
        }
        conflicts += int(len(records) > 1)
    if conflicts or capture.conflicting_duplicate_count:
        raise ValueError("conflicting duplicate provider records")
    return frame.drop_duplicates(keys, keep="first")


def _record_source_success(
    directory: Path,
    capture: SourceCapture,
    universe: set[str],
) -> dict:
    frame = _deduplicate(capture)
    source_dir = directory / "sources" / capture.source
    raw_hashes: list[str] = []
    for index, payload in enumerate(capture.raw_payloads, 1):
        raw_hashes.append(write_new_bytes(source_dir / "raw" / f"page_{index:04d}.bin", payload))
    normalized_path = source_dir / "normalized.csv"
    normalized_hash = write_new_frame(normalized_path, frame, list(capture.normalized_keys))
    covered = set(frame["symbol"].astype(str).str.zfill(6)) if "symbol" in frame else set()
    missing = sorted(universe - covered) if universe else sorted(capture.missing_symbols)
    return {
        "source": capture.source,
        "source_status": "SUCCESS" if len(frame) else "EMPTY_SUCCESS",
        "request_parameters": capture.request_parameters,
        "raw_response_sha256": raw_hashes,
        "normalized_data_sha256": normalized_hash,
        "normalized_path": normalized_path.as_posix(),
        "row_count": int(len(frame)),
        "universe_coverage": float(len(covered) / len(universe)) if universe else None,
        "duplicate_count": int(capture.duplicate_count),
        "conflicting_duplicate_count": int(capture.conflicting_duplicate_count),
        "missing_symbols": missing,
        "success": True,
        "failure_reason": None,
    }


def _record_source_failure(directory: Path, source: str, params: dict, error: BaseException) -> dict:
    receipt = {
        "source": source,
        "source_status": "REQUEST_FAILED",
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
    }
    write_new_json(directory / "sources" / source / "failure.json", receipt)
    return receipt


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
    """Capture each source independently after all no-network guards pass."""
    settings = settings or LedgerSettings()
    validate_observation_request(target_date, observed_at, trading_calendar, settings)
    observation_id = observed_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = settings.data_root / observation_id
    directory.mkdir(parents=True, exist_ok=False)
    receipts: dict[str, dict] = {}
    for source, (params, fetcher) in source_fetchers.items():
        try:
            capture = fetcher()
            if capture.source != source:
                raise ValueError("source identity changed; fallback is forbidden")
            receipts[source] = _record_source_success(directory, capture, universe)
        except BaseException as error:
            receipts[source] = _record_source_failure(directory, source, params, error)
    successful = [name for name, item in receipts.items() if item["success"]]
    failed = [name for name, item in receipts.items() if not item["success"]]
    record = {
        "observation_id": observation_id,
        "target_date": target_date,
        "observed_at": observed_at.astimezone(timezone.utc).isoformat(),
        "qualifying_trading_observation": True,
        "sources": receipts,
        "status": "SUCCESS" if not failed else ("PARTIAL" if successful else "FAILED"),
        "universe_hash": _universe_hash(universe),
        "pit_membership_snapshot_hash": membership_snapshot_hash,
        "pit_industry_mapping_hash": industry_mapping_hash,
        "universe_size": len(universe),
        "code_commit_sha": _commit_sha(),
        "lock_sha256": sha256_file(settings.lock_path),
        "model_training_ready": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    write_new_json(directory / "observation.json", record)
    return record


def import_verified_baseline(
    *,
    baseline_manifest: Path,
    expectation_path: Path,
    announcement_receipt: Path,
    settings: LedgerSettings | None = None,
) -> dict:
    """Index existing immutable evidence without pretending to recapture it."""
    settings = settings or LedgerSettings()
    source = json.loads(baseline_manifest.read_text(encoding="utf-8"))
    observation_id = "inherited-" + source["observation_id"]
    target = settings.data_root / observation_id / "observation.json"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing["inherited_manifest_sha256"] != sha256_file(baseline_manifest):
            raise RuntimeError("inherited baseline changed")
        return existing
    announcement = json.loads(announcement_receipt.read_text(encoding="utf-8"))
    expectation = source["sources"]["earnings_expectations"]
    flow = source["sources"]["fund_flows"]
    record = {
        "observation_id": observation_id,
        "target_date": source["observed_date_shanghai"],
        "observed_at": source["observed_at_utc"],
        "qualifying_trading_observation": False,
        "qualification_reason": "weekend baseline evidence; not a Shanghai trading sample",
        "source_mode": "INHERITED_VERIFIED_BASELINE_NO_NETWORK",
        "sources": {
            "earnings_expectations": {
                "source_status": "SUCCESS",
                "row_count": expectation["rows"],
                "universe_coverage": expectation["coverage"],
                "duplicate_count": expectation["duplicate_audit"]["duplicate_rows_removed"],
                "conflicting_duplicate_count": expectation["duplicate_audit"]["conflicting_duplicate_symbols"],
                "missing_symbols": expectation["duplicate_audit"]["missing_watchlist_symbols"],
                "raw_response_sha256": expectation["duplicate_audit"]["raw_page_sha256"],
                "normalized_data_sha256": sha256_file(expectation_path),
                "success": True,
            },
            "fund_flows": {
                "source_status": "SOURCE_UNAVAILABLE",
                "success": False,
                "failure_reason": flow["error"],
                "silent_fallback_used": False,
            },
            "announcements": {
                "source_status": "EMPTY_SUCCESS",
                "success": True,
                "row_count": announcement["records_returned"],
                "raw_response_sha256": announcement["raw_page_sha256"],
            },
        },
        "inherited_manifest_path": baseline_manifest.as_posix(),
        "inherited_manifest_sha256": sha256_file(baseline_manifest),
        "expectation_snapshot_path": expectation_path.as_posix(),
        "announcement_receipt_sha256": sha256_file(announcement_receipt),
        "code_commit_sha": _commit_sha(),
        "lock_sha256": sha256_file(settings.lock_path),
        "model_training_ready": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    write_new_json(target, record)
    return record
