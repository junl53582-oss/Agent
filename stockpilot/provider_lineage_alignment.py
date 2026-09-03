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

from research_v10.history_data import _normalize_hfq
from stockpilot.daily_pit.pipeline import DailyPitSettings, policy_hashes
from stockpilot.data import REQUIRED_COLUMNS, validate_panel
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
