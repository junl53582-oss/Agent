"""Isolated Tencent-HFQ candidate acquisition for DAILY PIT lineage repair.

This module is additive.  It never changes the locked Eastmoney-first DAILY PIT
router and never overwrites the operational daily market partition.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from research_v10.features import V10_FEATURES
from research_v10.history_data import _normalize_hfq
from stockpilot.daily_pit import pipeline as daily_pit_pipeline
from stockpilot.daily_pit.pipeline import (
    DAILY_FEATURE_COLUMNS,
    DailyPitError,
    DailyPitSettings,
    policy_hashes,
)
from stockpilot.data import REQUIRED_COLUMNS, load_panel, validate_panel
from stockpilot.membership import load_membership_history
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

TARGET_PROVIDER = "akshare-tencent"
CANDIDATE_VERSION = "DAILY_PIT_TENCENT_HFQ_LINEAGE_CANDIDATE_V1"
PRIORITY_POLICY_VERSION = "DAILY_PIT_PROVIDER_PRIORITY_TENCENT_FIRST_V1"


class ProviderLineageAlignmentError(RuntimeError):
    """Raised before publishing an incomplete or mixed-provider candidate."""


@dataclass(frozen=True)
class ProviderLineageAlignmentSettings:
    candidate_root: Path = Path(
        "data/prospective_gen2/provider_lineage_candidates/tencent_hfq/daily_inputs"
    )
    cache_root: Path = Path(
        "data/prospective_gen2/provider_lineage_candidates/tencent_hfq/provider_cache"
    )
    production_root: Path = Path("data/prospective_gen2/daily_inputs")
    membership_path: Path = Path("data/universes/000300/history_v10.csv")
    overlap_calendar_days: int = 45
    workers: int = 8

    def candidate_dir(self, target_date: str) -> Path:
        return self.candidate_root / target_date

    def production_dir(self, target_date: str) -> Path:
        return self.production_root / target_date


@dataclass(frozen=True)
class ProviderPrioritySettings:
    lineage_evidence_path: Path = Path(
        "artifacts/daily_predictions/gen2/hfq_overlap_diagnostic.json"
    )
    cache_root: Path = Path("data/prospective_gen2/provider_priority_cache")
    workers: int = 8


def _market_symbol(symbol: str) -> str:
    return ("sh" if symbol.startswith(("5", "6", "9")) else "sz") + symbol


def _current_members(target_date: str, settings: ProviderLineageAlignmentSettings) -> list[str]:
    history = load_membership_history(settings.membership_path)
    eligible = history[pd.to_datetime(history["snapshot_date"]).le(pd.Timestamp(target_date))]
    if eligible.empty:
        raise ProviderLineageAlignmentError("NO_PIT_MEMBERSHIP_SNAPSHOT")
    snapshot = pd.to_datetime(eligible["snapshot_date"]).max()
    return sorted(
        eligible.loc[pd.to_datetime(eligible["snapshot_date"]).eq(snapshot), "symbol"]
        .astype(str)
        .str.zfill(6)
        .unique()
        .tolist()
    )


def _atomic_cache_write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_frame_bytes(frame, ["date", "symbol"])
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    os.replace(temporary, path)


def fetch_tencent_hfq_candidate(
    symbols: list[str],
    start_date: str,
    end_date: str,
    *,
    cache_root: Path,
    workers: int = 8,
    provider: Callable[..., pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, str]], dict[str, int]]:
    """Fetch only Tencent HFQ bars and normalize with the frozen V10 adapter."""

    if provider is None:
        try:
            import akshare as ak
        except ImportError as error:  # pragma: no cover
            raise ProviderLineageAlignmentError("AKSHARE_NOT_INSTALLED") from error
        provider = ak.stock_zh_a_hist_tx
    normalized = sorted({str(symbol).zfill(6) for symbol in symbols})
    compact_start = start_date.replace("-", "")
    compact_end = end_date.replace("-", "")

    def load_one(symbol: str) -> tuple[str, pd.DataFrame | None, str, str | None]:
        target = cache_root / f"{symbol}_{start_date}_{end_date}_tencent_hfq.csv"
        if target.is_file():
            try:
                cached = validate_panel(pd.read_csv(target, dtype={"symbol": str}))
                return symbol, cached, "cache", None
            except Exception as error:  # noqa: BLE001
                return symbol, None, "cache", type(error).__name__
        try:
            raw = provider(
                symbol=_market_symbol(symbol),
                start_date=compact_start,
                end_date=compact_end,
                adjust="hfq",
                timeout=30,
            )
            if raw is None or raw.empty:
                return symbol, None, "tencent", "EMPTY"
            frame = _normalize_hfq(raw, symbol, "tencent")
            _atomic_cache_write(target, frame)
            return symbol, frame, "tencent", None
        except Exception as error:  # noqa: BLE001
            return symbol, None, "tencent", f"{type(error).__name__}:{error}"

    results = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as executor:
        futures = [executor.submit(load_one, symbol) for symbol in normalized]
        for future in as_completed(futures):
            results.append(future.result())
    frames = [frame for _, frame, _, _ in results if frame is not None and not frame.empty]
    failures = [
        {"symbol": symbol, "source": source, "error": error or "UNKNOWN"}
        for symbol, frame, source, error in results
        if frame is None or frame.empty
    ]
    if not frames:
        raise ProviderLineageAlignmentError("NO_TENCENT_HFQ_ROWS")
    panel = validate_panel(pd.concat(frames, ignore_index=True))
    source_counts = pd.Series(
        [source for _, frame, source, _ in results if frame is not None]
    ).value_counts()
    return panel, failures, {str(key): int(value) for key, value in source_counts.items()}


def resolve_historical_provider(settings: ProviderPrioritySettings | None = None) -> str:
    """Resolve the frozen history lineage from immutable provenance evidence."""

    settings = settings or ProviderPrioritySettings()
    evidence = read_verified_json(settings.lineage_evidence_path)
    provider = (
        evidence.get("HFQ_MISMATCH_ROOT_CAUSE", {})
        .get("historical_canonical_provider", "")
        .lower()
    )
    if provider != "tencent":
        raise ProviderLineageAlignmentError(
            f"FROZEN_HISTORICAL_LINEAGE_UNSUPPORTED:{provider or 'MISSING'}"
        )
    return TARGET_PROVIDER


def fetch_tencent_first_hfq(
    symbols: list[str],
    start_date: str,
    end_date: str,
    *,
    cache_root: Path,
    workers: int = 8,
    tencent_provider: Callable[..., pd.DataFrame] | None = None,
    eastmoney_provider: Callable[..., pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, str]], dict[str, int], int]:
    """Fetch Tencent first and use Eastmoney only for unavailable symbols."""

    if tencent_provider is None or eastmoney_provider is None:
        try:
            import akshare as ak
        except ImportError as error:  # pragma: no cover
            raise ProviderLineageAlignmentError("AKSHARE_NOT_INSTALLED") from error
        tencent_provider = tencent_provider or ak.stock_zh_a_hist_tx
        eastmoney_provider = eastmoney_provider or ak.stock_zh_a_hist
    normalized = sorted({str(symbol).zfill(6) for symbol in symbols})
    compact_start = start_date.replace("-", "")
    compact_end = end_date.replace("-", "")

    def load_one(symbol: str) -> tuple[str, pd.DataFrame | None, str, str | None, int]:
        tencent_cache = cache_root / f"{symbol}_{start_date}_{end_date}_tencent_hfq.csv"
        eastmoney_cache = cache_root / f"{symbol}_{start_date}_{end_date}_eastmoney_hfq.csv"
        tencent_cache_error: str | None = None
        if tencent_cache.is_file():
            try:
                frame = validate_panel(pd.read_csv(tencent_cache, dtype={"symbol": str}))
                return symbol, frame, "tencent", None, 0
            except Exception as error:  # noqa: BLE001
                tencent_cache_error = type(error).__name__
        requests = 1
        tencent_error = (
            f"CACHE_INVALID:{tencent_cache_error};EMPTY" if tencent_cache_error else "EMPTY"
        )
        try:
            raw = tencent_provider(
                symbol=_market_symbol(symbol),
                start_date=compact_start,
                end_date=compact_end,
                adjust="hfq",
                timeout=30,
            )
            if raw is not None and not raw.empty:
                frame = _normalize_hfq(raw, symbol, "tencent")
                _atomic_cache_write(tencent_cache, frame)
                return symbol, frame, "tencent", None, requests
        except Exception as error:  # noqa: BLE001
            tencent_error = type(error).__name__
        eastmoney_cache_error: str | None = None
        if eastmoney_cache.is_file():
            try:
                frame = validate_panel(pd.read_csv(eastmoney_cache, dtype={"symbol": str}))
                return symbol, frame, "eastmoney", None, requests
            except Exception as error:  # noqa: BLE001
                eastmoney_cache_error = type(error).__name__
        requests += 1
        try:
            raw = eastmoney_provider(
                symbol=symbol,
                period="daily",
                start_date=compact_start,
                end_date=compact_end,
                adjust="hfq",
                timeout=20,
            )
            if raw is not None and not raw.empty:
                frame = _normalize_hfq(raw, symbol, "eastmoney")
                _atomic_cache_write(eastmoney_cache, frame)
                return symbol, frame, "eastmoney", None, requests
            eastmoney_error = (
                f"CACHE_INVALID:{eastmoney_cache_error};EMPTY"
                if eastmoney_cache_error
                else "EMPTY"
            )
        except Exception as error:  # noqa: BLE001
            eastmoney_error = type(error).__name__
        return (
            symbol,
            None,
            "failed",
            f"tencent:{tencent_error};eastmoney:{eastmoney_error}",
            requests,
        )

    results = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as executor:
        futures = [executor.submit(load_one, symbol) for symbol in normalized]
        for future in as_completed(futures):
            results.append(future.result())
    frames = [frame for _, frame, _, _, _ in results if frame is not None and not frame.empty]
    failures = [
        {"symbol": symbol, "source": source, "error": error or "UNKNOWN"}
        for symbol, frame, source, error, _ in results
        if frame is None or frame.empty
    ]
    if not frames:
        raise ProviderLineageAlignmentError("NO_HFQ_ROWS")
    panel = validate_panel(pd.concat(frames, ignore_index=True))
    counts = pd.Series(
        [source for _, frame, source, _, _ in results if frame is not None]
    ).value_counts()
    return (
        panel,
        failures,
        {str(key): int(value) for key, value in counts.items()},
        sum(requests for *_, requests in results),
    )


def validate_routed_hfq_lineage(
    frozen: pd.DataFrame,
    incremental: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    target_date: str,
    settings: DailyPitSettings,
) -> dict[str, Any]:
    """Run the unchanged HFQ overlap validator before publishing routed data."""

    cutoff = pd.to_datetime(frozen["date"]).max()
    try:
        _, audit = daily_pit_pipeline.stitch_hfq_market(
            frozen,
            incremental,
            membership,
            cutoff=cutoff,
            as_of=target_date,
            settings=settings.forward_settings(),
        )
    except Exception as error:
        raise DailyPitError("HFQ_LINEAGE_FALLBACK_BLOCKED", str(error)) from error
    return audit


PriorityFetcher = Callable[
    [list[str], str, str],
    tuple[pd.DataFrame, list[dict[str, str]], dict[str, int], int],
]


def acquire_lineage_aligned_market(
    target_date: str,
    requested_symbols: list[str] | tuple[str, ...],
    *,
    now: datetime,
    settings: DailyPitSettings | None = None,
    priority_settings: ProviderPrioritySettings | None = None,
    fetcher: Callable[..., tuple[pd.DataFrame, list[dict[str, str]], dict[str, int], int]] = (
        fetch_tencent_first_hfq
    ),
) -> dict[str, Any]:
    """Acquire Tencent-first DAILY PIT data with pre-publication overlap validation."""

    settings = settings or DailyPitSettings()
    priority_settings = priority_settings or ProviderPrioritySettings()
    existing = daily_pit_pipeline._verify_existing_market(target_date, settings)
    if existing is not None:
        return existing
    daily_pit_pipeline._session_guard(target_date, now, settings)
    historical_provider = resolve_historical_provider(priority_settings)
    snapshot, required = daily_pit_pipeline._current_members(target_date, settings)
    requested = sorted({str(symbol).zfill(6) for symbol in requested_symbols} | required)
    target = pd.Timestamp(target_date)
    start = (target - pd.Timedelta(days=settings.overlap_calendar_days)).date().isoformat()
    try:
        market, failures, sources, provider_requests = fetcher(
            requested,
            start,
            target_date,
            cache_root=priority_settings.cache_root,
            workers=priority_settings.workers,
        )
    except Exception as error:
        raise DailyPitError("MARKET_DATA_NOT_READY", str(error)) from error
    market = validate_panel(market)
    market = market[pd.to_datetime(market["date"]).le(target)].copy()
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()
    market["symbol"] = market["symbol"].astype(str).str.zfill(6)
    target_rows = market[market["date"].eq(target)]
    covered = required.intersection(set(target_rows["symbol"]))
    coverage = len(covered) / len(required)
    if target_rows.empty:
        raise DailyPitError("MARKET_DATA_NOT_READY", "TARGET_DATE_BAR_MISSING")
    if coverage < settings.minimum_universe_coverage:
        raise DailyPitError(
            "MARKET_COVERAGE_INSUFFICIENT",
            f"{coverage:.6f}<{settings.minimum_universe_coverage:.6f}",
        )
    frozen = load_panel(settings.frozen_market_path)
    membership = load_membership_history(settings.membership_path)
    overlap_audit = validate_routed_hfq_lineage(
        frozen, market, membership, target_date=target_date, settings=settings
    )
    fallback_used = int(sources.get("eastmoney", 0)) > 0
    priority_policy = {
        "version": PRIORITY_POLICY_VERSION,
        "historical_lineage": historical_provider,
        "primary": TARGET_PROVIDER,
        "fallback": "akshare-eastmoney",
        "fallback_condition": "tencent_unavailable",
        "fallback_requires_hfq_overlap_validation": True,
    }
    directory = settings.date_dir(target_date)
    market = market.sort_values(["date", "symbol"]).reset_index(drop=True)
    market_hash = write_immutable_bytes(
        directory / "market.csv", canonical_frame_bytes(market, ["date", "symbol"])
    )
    failure_frame = pd.DataFrame(failures, columns=["symbol", "source", "error"])
    failure_hash = write_immutable_bytes(
        directory / "market_failures.csv",
        canonical_frame_bytes(failure_frame, ["symbol"])
        if not failure_frame.empty
        else b"\xef\xbb\xbfsymbol,source,error\n",
    )
    receipt = {
        "target_date": target_date,
        "acquired_at_utc": now.astimezone(timezone.utc).isoformat(),
        "request_start_date": start,
        "request_end_date": target_date,
        "requested_symbols": len(requested),
        "historical_provider": historical_provider,
        "provider_sources": sources,
        "provider_request_count": provider_requests,
        "provider_fallback_order": [TARGET_PROVIDER, "akshare-eastmoney"],
        "fallback_used": fallback_used,
        "lineage_warning": "EASTMONEY_FALLBACK_USED" if fallback_used else None,
        "overlap_validation": "PASS",
        "overlap_audit_sha256": sha256_bytes(canonical_json_bytes(overlap_audit)),
        "provider_priority_policy_sha256": sha256_bytes(
            canonical_json_bytes(priority_policy)
        ),
        "provider_failures": len(failures),
        "target_rows": len(target_rows),
        "target_symbols": int(target_rows["symbol"].nunique()),
        "required_membership_snapshot": snapshot,
        "required_membership_symbols": len(required),
        "required_membership_covered": len(covered),
        "required_membership_coverage": coverage,
        "maximum_market_date": str(market["date"].max().date()),
        "future_market_used": False,
        "previous_day_substituted": False,
        "prediction_created": False,
        "reservation_created": False,
        **policy_hashes(settings),
    }
    receipt_hash = write_immutable_json(directory / "source_receipt.json", receipt)
    manifest = {
        "manifest_version": PRIORITY_POLICY_VERSION,
        "target_date": target_date,
        "files": {
            "market.csv": market_hash,
            "market_failures.csv": failure_hash,
            "source_receipt.json": receipt_hash,
        },
        "market_rows": len(market),
        "market_symbols": int(market["symbol"].nunique()),
        "target_rows": len(target_rows),
        "target_symbols": int(target_rows["symbol"].nunique()),
        "provider_requests_made": provider_requests,
        "provider_priority_policy_sha256": receipt["provider_priority_policy_sha256"],
        **policy_hashes(settings),
    }
    manifest_hash = write_immutable_json(directory / "market_manifest.json", manifest)
    return manifest | {"market_manifest_sha256": manifest_hash, "idempotent": False}


def verify_candidate(
    target_date: str, settings: ProviderLineageAlignmentSettings | None = None
) -> dict[str, Any]:
    settings = settings or ProviderLineageAlignmentSettings()
    directory = settings.candidate_dir(target_date)
    manifest = read_verified_json(directory / "market_manifest.json")
    for name, expected in manifest["files"].items():
        if verify_immutable(directory / name) != expected:
            raise ProviderLineageAlignmentError(f"CANDIDATE_HASH_MISMATCH:{name}")
    receipt = read_verified_json(directory / "source_receipt.json")
    if receipt.get("provider") != TARGET_PROVIDER or receipt.get("mixed_provider") is not False:
        raise ProviderLineageAlignmentError("CANDIDATE_PROVIDER_LINEAGE_INVALID")
    market = validate_panel(pd.read_csv(directory / "market.csv", dtype={"symbol": str}))
    if list(market.columns) != REQUIRED_COLUMNS:
        raise ProviderLineageAlignmentError("CANDIDATE_SCHEMA_INVALID")
    return manifest | {
        "market_manifest_sha256": verify_immutable(directory / "market_manifest.json"),
        "market_rows": len(market),
        "market_symbols": int(market["symbol"].nunique()),
        "idempotent": True,
        "provider_requests_made": 0,
    }


def acquire_tencent_candidate(
    target_date: str,
    *,
    now: datetime | None = None,
    settings: ProviderLineageAlignmentSettings | None = None,
    provider: Callable[..., pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Publish a complete Tencent-only candidate beside the operational partition."""

    settings = settings or ProviderLineageAlignmentSettings()
    now = now or datetime.now(timezone.utc)
    directory = settings.candidate_dir(target_date)
    if (directory / "market_manifest.json").is_file():
        return verify_candidate(target_date, settings)
    production = settings.production_dir(target_date)
    production_files = ["market.csv", "market_manifest.json", "source_receipt.json"]
    before = {name: sha256_file(production / name) for name in production_files}
    symbols = _current_members(target_date, settings)
    target = pd.Timestamp(target_date)
    start_date = (target - pd.Timedelta(days=settings.overlap_calendar_days)).date().isoformat()
    market, failures, sources = fetch_tencent_hfq_candidate(
        symbols,
        start_date,
        target_date,
        cache_root=settings.cache_root,
        workers=settings.workers,
        provider=provider,
    )
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()
    market = market[market["date"].le(target)].sort_values(["date", "symbol"]).reset_index(drop=True)
    target_symbols = set(market.loc[market["date"].eq(target), "symbol"].astype(str).str.zfill(6))
    missing = sorted(set(symbols).difference(target_symbols))
    if failures or missing or len(target_symbols) != len(symbols):
        raise ProviderLineageAlignmentError(
            "TENCENT_CANDIDATE_INCOMPLETE:"
            + json.dumps({"failures": failures, "missing_target_symbols": missing}, ensure_ascii=False)
        )
    if set(sources).difference({"tencent", "cache"}):
        raise ProviderLineageAlignmentError(f"MIXED_PROVIDER_CANDIDATE:{sources}")
    after = {name: sha256_file(production / name) for name in production_files}
    if before != after:
        raise ProviderLineageAlignmentError("PRODUCTION_PARTITION_CHANGED")
    market_hash = write_immutable_bytes(
        directory / "market.csv", canonical_frame_bytes(market, ["date", "symbol"])
    )
    failures_hash = write_immutable_bytes(
        directory / "market_failures.csv", b"\xef\xbb\xbfsymbol,source,error\n"
    )
    alignment_policy = {
        "candidate_only": True,
        "provider": TARGET_PROVIDER,
        "adjustment": "hfq",
        "fallback_allowed": False,
        "normalizer": "research_v10.history_data._normalize_hfq(source=tencent)",
        "production_partition_overwrite_allowed": False,
    }
    receipt = {
        "candidate_version": CANDIDATE_VERSION,
        "target_date": target_date,
        "acquired_at_utc": now.astimezone(timezone.utc).isoformat(),
        "provider": TARGET_PROVIDER,
        "adjustment": "hfq",
        "mixed_provider": False,
        "fallback_used": False,
        "normalizer": alignment_policy["normalizer"],
        "requested_symbols": len(symbols),
        "target_symbols": len(target_symbols),
        "market_rows": len(market),
        "request_start_date": start_date,
        "request_end_date": target_date,
        "maximum_market_date": str(market["date"].max().date()),
        "provider_sources": sources,
        "provider_requests_made": int(sources.get("tencent", 0)),
        "provider_failures": 0,
        "future_market_used": False,
        "previous_day_substituted": False,
        "production_partition_modified": False,
        "production_hashes_before": before,
        "production_hashes_after": after,
        "alignment_policy_sha256": sha256_bytes(canonical_json_bytes(alignment_policy)),
    }
    receipt_hash = write_immutable_json(directory / "source_receipt.json", receipt)
    manifest = {
        "manifest_version": CANDIDATE_VERSION,
        "target_date": target_date,
        "provider": TARGET_PROVIDER,
        "adjustment": "hfq",
        "mixed_provider": False,
        "files": {
            "market.csv": market_hash,
            "market_failures.csv": failures_hash,
            "source_receipt.json": receipt_hash,
        },
        "market_rows": len(market),
        "market_symbols": int(market["symbol"].nunique()),
        "target_rows": len(target_symbols),
        "target_symbols": len(target_symbols),
        "provider_requests_made": int(sources.get("tencent", 0)),
        "candidate_only": True,
        "alignment_policy_sha256": receipt["alignment_policy_sha256"],
        **policy_hashes(DailyPitSettings()),
    }
    manifest_hash = write_immutable_json(directory / "market_manifest.json", manifest)
    return manifest | {"market_manifest_sha256": manifest_hash, "idempotent": False}


def _assemble_candidate_panel(metadata: pd.DataFrame, reduced: pd.DataFrame) -> pd.DataFrame:
    """Join locked builder values without dropping its existing sector metadata."""

    keys = ["date", "symbol"]
    sector_values = reduced[[*keys, "broad_sector"]].copy()
    sector_values["date"] = pd.to_datetime(sector_values["date"]).dt.normalize()
    metadata = metadata.drop(columns=["broad_sector"], errors="ignore").merge(
        sector_values, on=keys, how="inner", validate="one_to_one"
    )
    feature_values = reduced[[*keys, *V10_FEATURES]].copy()
    feature_values["date"] = pd.to_datetime(feature_values["date"]).dt.normalize()
    panel = metadata.merge(feature_values, on=keys, how="inner", validate="one_to_one")
    return panel[DAILY_FEATURE_COLUMNS].sort_values(keys).reset_index(drop=True)


def materialize_aligned_candidate_features(
    target_date: str,
    *,
    settings: DailyPitSettings,
) -> dict[str, Any]:
    """Materialize the Tencent candidate with unchanged locked feature semantics.

    The locked builder already emits ``broad_sector``.  The locked DAILY PIT
    materializer accidentally drops that metadata column before its final schema
    selection.  This candidate-only adapter carries the emitted value through;
    it does not infer, transform, or replace any model feature.
    """

    directory = settings.date_dir(target_date)
    panel_path = directory / "panel.parquet"
    manifest_path = directory / "manifest.json"
    if panel_path.exists() or manifest_path.exists():
        return daily_pit_pipeline.verify_daily_feature_partition(
            target_date, settings=settings
        ) | {"idempotent": True}
    try:
        market_manifest = read_verified_json(directory / "market_manifest.json")
        market_manifest_hash = verify_immutable(directory / "market_manifest.json")
        market_hash = verify_immutable(directory / "market.csv")
    except Exception as error:
        raise DailyPitError("MARKET_DATA_NOT_READY", str(error)) from error
    if market_manifest.get("target_date") != target_date:
        raise DailyPitError("DAILY_FEATURE_MANIFEST_INVALID", "MARKET_TARGET_DATE")
    try:
        frozen = load_panel(settings.frozen_market_path)
        incremental = load_panel(directory / "market.csv")
        membership = load_membership_history(settings.membership_path)
        cutoff = pd.to_datetime(frozen["date"]).max()
        if cutoff >= pd.Timestamp(target_date):
            combined = frozen[
                pd.to_datetime(frozen["date"]).le(pd.Timestamp(target_date))
            ].copy()
            stitch_audit: dict[str, Any] = {"passed": True, "mode": "frozen_contains_target"}
        else:
            combined, stitch_audit = daily_pit_pipeline.stitch_hfq_market(
                frozen,
                incremental,
                membership,
                cutoff=cutoff,
                as_of=target_date,
                settings=settings.forward_settings(),
            )
        reduced, build_audit = daily_pit_pipeline.build_latest_pit_feature_panel(
            combined, target_date, settings=settings.forward_settings()
        )
        metadata = daily_pit_pipeline._metadata_for_current(
            combined, target_date, reduced["symbol"], settings
        )
        panel = _assemble_candidate_panel(metadata, reduced)
        daily_pit_pipeline._validate_daily_panel(panel, target_date)
    except DailyPitError:
        raise
    except Exception as error:
        raise DailyPitError(
            "TARGET_DATE_FEATURE_MATERIALIZATION_FAILED", str(error)
        ) from error
    panel_hash = write_immutable_bytes(panel_path, daily_pit_pipeline._parquet_bytes(panel))
    source_hashes = {
        str(directory / "market_manifest.json"): market_manifest_hash,
        str(directory / "market.csv"): market_hash,
        str(settings.frozen_market_path): sha256_file(settings.frozen_market_path),
        str(settings.membership_path): sha256_file(settings.membership_path),
        str(settings.fundamental_path): sha256_file(settings.fundamental_path),
        str(settings.industry_path): sha256_file(settings.industry_path),
    }
    manifest = {
        "manifest_version": "DAILY_PIT_FEATURES_V1",
        "target_date": target_date,
        "panel_sha256": panel_hash,
        "rows": len(panel),
        "symbols": int(panel["symbol"].nunique()),
        "columns": DAILY_FEATURE_COLUMNS,
        "column_count": len(DAILY_FEATURE_COLUMNS),
        "feature_count": len(V10_FEATURES),
        "source_hashes": source_hashes,
        "stitch_audit_sha256": sha256_bytes(canonical_json_bytes(stitch_audit)),
        "builder_audit_sha256": sha256_bytes(canonical_json_bytes(build_audit)),
        "provider_lineage_adapter": "CARRY_LOCKED_BUILDER_BROAD_SECTOR_V1",
        "broad_sector_source": "build_latest_pit_feature_panel output",
        "feature_semantics_changed": False,
        "membership_not_future": True,
        "fundamental_not_future": True,
        "industry_not_future": True,
        "future_market_used": False,
        "previous_day_substituted": False,
        "historical_training_parquet_modified": False,
        "prediction_created": False,
        "reservation_created": False,
        "prediction_backfill_2026_09_01": False,
        **policy_hashes(settings),
    }
    digest = write_immutable_json(manifest_path, manifest)
    return manifest | {"manifest_sha256": digest, "idempotent": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Acquire isolated Tencent HFQ DAILY PIT candidate")
    parser.add_argument("target_date")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    result = verify_candidate(args.target_date) if args.verify else acquire_tencent_candidate(args.target_date)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
