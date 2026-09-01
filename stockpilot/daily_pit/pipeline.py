from __future__ import annotations

import io
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research_v9.data import attach_industry_asof, attach_membership_weight, load_industry_history
from research_v10.features import V10_FEATURES
from research_v10.fundamentals import (
    attach_extended_fundamentals_asof,
    load_extended_fundamentals,
)
from research_v10.history_data import fetch_hfq_history
from stockpilot.data import load_panel, validate_panel
from stockpilot.membership import attach_point_in_time_membership, load_membership_history
from stockpilot.prediction_forward import (
    ForwardPredictionSettings,
    build_latest_pit_feature_panel,
    stitch_hfq_market,
)
from stockpilot.prospective_r2.calendar import load_verified_calendar
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

SHANGHAI = ZoneInfo("Asia/Shanghai")
META_COLUMNS = [
    "date",
    "symbol",
    "eligible",
    "in_universe",
    "membership_snapshot_date",
    "available_date",
    "industry_effective_date",
    "industry",
    "broad_sector",
    "benchmark_weight",
]
DAILY_FEATURE_COLUMNS = list(dict.fromkeys([*META_COLUMNS, *V10_FEATURES]))
if len(DAILY_FEATURE_COLUMNS) != 71:  # frozen-contract assertion at import time
    raise RuntimeError(f"DAILY_FEATURE_SCHEMA_INVALID:{len(DAILY_FEATURE_COLUMNS)}")


class DailyPitError(RuntimeError):
    """Fail-closed error carrying one stable operational failure code."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


@dataclass(frozen=True)
class DailyPitSettings:
    root: Path = Path("data/prospective_gen2/daily_inputs")
    calendar_path: Path = Path("artifacts/prospective_alpha_v1r2/trading_calendar_2026.json")
    frozen_market_path: Path = Path("data/market_history_v10_hfq.csv")
    membership_path: Path = Path("data/universes/000300/history_v10.csv")
    fundamental_path: Path = Path("data/fundamentals_pit_v10_extended.csv")
    industry_path: Path = Path("data/industry_history_v10.csv")
    names_path: Path = Path("data/stock_names.csv")
    provider_cache_dir: Path = Path("data/raw_v10_hfq")
    earliest_ready_time: time = time(18, 30)
    minimum_universe_coverage: float = 0.95
    overlap_calendar_days: int = 45
    continuity_only_dates: tuple[str, ...] = ("2026-08-31", "2026-09-01")
    permanently_blocked_prediction_dates: tuple[str, ...] = ("2026-09-01",)

    def date_dir(self, target_date: str) -> Path:
        return self.root / target_date

    def forward_settings(self) -> ForwardPredictionSettings:
        return ForwardPredictionSettings(
            frozen_market_path=self.frozen_market_path,
            membership_path=self.membership_path,
            fundamental_path=self.fundamental_path,
            industry_path=self.industry_path,
            names_path=self.names_path,
            minimum_current_coverage=self.minimum_universe_coverage,
        )


def _utc(now: datetime) -> str:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(timezone.utc).isoformat()


def _policy_payload(settings: DailyPitSettings) -> dict:
    return {
        "market_acquisition": {
            "adjustment": "hfq",
            "primary": "akshare-eastmoney",
            "fallback": "akshare-tencent",
            "fallback_order_explicit": True,
            "request_end_is_target_date": True,
            "future_market_forbidden": True,
            "previous_day_substitution_forbidden": True,
            "minimum_universe_coverage": settings.minimum_universe_coverage,
            "overlap_calendar_days": settings.overlap_calendar_days,
        },
        "feature_materialization": {
            "builder": "stockpilot.prediction_forward.build_latest_pit_feature_panel",
            "features": list(V10_FEATURES),
            "membership_join": "latest snapshot_date <= target_date",
            "fundamental_join": "latest available_date <= target_date",
            "industry_join": "latest effective_date <= target_date",
            "decision_time": "target close after 18:30 Asia/Shanghai",
            "entry_time": "next verified trading session open",
        },
    }


def policy_hashes(settings: DailyPitSettings) -> dict[str, str]:
    payload = _policy_payload(settings)
    return {
        "market_acquisition_policy_sha256": sha256_bytes(
            canonical_json_bytes(payload["market_acquisition"])
        ),
        "feature_materialization_policy_sha256": sha256_bytes(
            canonical_json_bytes(payload["feature_materialization"])
        ),
        "daily_feature_schema_sha256": sha256_bytes(canonical_json_bytes(DAILY_FEATURE_COLUMNS)),
    }


def _session_guard(target_date: str, now: datetime, settings: DailyPitSettings) -> None:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    target = pd.Timestamp(target_date).normalize()
    try:
        sessions = load_verified_calendar(settings.calendar_path).sessions()
    except Exception as error:
        raise DailyPitError("MARKET_DATA_NOT_READY", f"CALENDAR:{error}") from error
    if target not in sessions:
        raise DailyPitError("MARKET_DATA_NOT_READY", "NOT_VERIFIED_TRADING_SESSION")
    local = now.astimezone(SHANGHAI)
    if target_date not in settings.continuity_only_dates:
        if target_date != local.date().isoformat():
            raise DailyPitError("MARKET_DATA_NOT_READY", "TARGET_DATE_MUST_BE_TODAY")
        if local.timetz().replace(tzinfo=None) < settings.earliest_ready_time:
            raise DailyPitError("MARKET_DATA_NOT_READY", "DATA_WINDOW_NOT_OPEN")
    elif target > pd.Timestamp(local.date()):
        raise DailyPitError("MARKET_DATA_NOT_READY", "CONTINUITY_DATE_IS_FUTURE")


def _current_members(target_date: str, settings: DailyPitSettings) -> tuple[str, set[str]]:
    try:
        history = load_membership_history(settings.membership_path)
    except Exception as error:
        raise DailyPitError("PIT_JOIN_INVALID", f"MEMBERSHIP:{error}") from error
    history["snapshot_date"] = pd.to_datetime(history["snapshot_date"]).dt.normalize()
    legal = history[history["snapshot_date"].le(pd.Timestamp(target_date))]
    if legal.empty:
        raise DailyPitError("PIT_JOIN_INVALID", "MEMBERSHIP_SNAPSHOT_MISSING")
    snapshot = legal["snapshot_date"].max()
    symbols = set(legal.loc[legal["snapshot_date"].eq(snapshot), "symbol"].astype(str).str.zfill(6))
    if not symbols:
        raise DailyPitError("PIT_JOIN_INVALID", "MEMBERSHIP_UNIVERSE_EMPTY")
    return str(snapshot.date()), symbols


def _verify_existing_market(target_date: str, settings: DailyPitSettings) -> dict | None:
    directory = settings.date_dir(target_date)
    manifest_path = directory / "market_manifest.json"
    if not directory.exists():
        return None
    expected = {
        "market.csv",
        "market.csv.sha256",
        "market_failures.csv",
        "market_failures.csv.sha256",
        "source_receipt.json",
        "source_receipt.json.sha256",
        "market_manifest.json",
        "market_manifest.json.sha256",
    }
    present = {path.name for path in directory.iterdir() if path.is_file()}
    if not expected.issubset(present):
        raise DailyPitError("DAILY_FEATURE_MANIFEST_INVALID", "PARTIAL_MARKET_PARTITION")
    manifest = read_verified_json(manifest_path)
    for name, digest in manifest.get("files", {}).items():
        if verify_immutable(directory / name) != digest:
            raise DailyPitError("DAILY_FEATURE_MANIFEST_INVALID", f"HASH:{name}")
    if manifest.get("target_date") != target_date:
        raise DailyPitError("DAILY_FEATURE_MANIFEST_INVALID", "TARGET_DATE")
    return manifest | {"idempotent": True, "provider_requests_made": 0}


Fetcher = Callable[..., tuple[pd.DataFrame, list[dict]]]


def acquire_market(
    target_date: str,
    requested_symbols: list[str] | tuple[str, ...],
    *,
    now: datetime,
    settings: DailyPitSettings | None = None,
    fetcher: Fetcher = fetch_hfq_history,
) -> dict:
    """Acquire and immutably bind <=target HFQ evidence; never predicts or reserves."""
    settings = settings or DailyPitSettings()
    existing = _verify_existing_market(target_date, settings)
    if existing is not None:
        return existing
    _session_guard(target_date, now, settings)
    snapshot, required = _current_members(target_date, settings)
    requested = sorted({str(symbol).zfill(6) for symbol in requested_symbols} | required)
    target = pd.Timestamp(target_date)
    start = (target - pd.Timedelta(days=settings.overlap_calendar_days)).date().isoformat()
    with tempfile.TemporaryDirectory(prefix="stockpilot-daily-pit-") as temporary:
        temporary_path = Path(temporary)
        try:
            market, failures = fetcher(
                requested,
                start,
                target_date,
                output_path=temporary_path / "provider_market.csv",
                cache_dir=settings.provider_cache_dir,
                workers=8,
            )
        except Exception as error:
            raise DailyPitError("MARKET_DATA_NOT_READY", str(error)) from error
        provider_manifest_path = temporary_path / "provider_market.manifest.json"
        provider_manifest = (
            json.loads(provider_manifest_path.read_text(encoding="utf-8-sig"))
            if provider_manifest_path.is_file()
            else {}
        )
    market = validate_panel(market)
    market = market[pd.to_datetime(market["date"]).le(target)].copy()
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()
    market["symbol"] = market["symbol"].astype(str).str.zfill(6)
    if market.empty or market["date"].max() > target:
        raise DailyPitError("MARKET_DATA_NOT_READY", "NO_LEGAL_MARKET_ROWS")
    target_rows = market[market["date"].eq(target)]
    if target_rows.empty:
        raise DailyPitError("MARKET_DATA_NOT_READY", "TARGET_DATE_BAR_MISSING")
    covered = required.intersection(set(target_rows["symbol"]))
    coverage = len(covered) / len(required)
    if coverage < settings.minimum_universe_coverage:
        raise DailyPitError(
            "MARKET_COVERAGE_INSUFFICIENT",
            f"{coverage:.6f}<{settings.minimum_universe_coverage:.6f}",
        )
    market = market.sort_values(["date", "symbol"]).reset_index(drop=True)
    failure_frame = pd.DataFrame(failures, columns=["symbol", "source", "error"])
    sources = provider_manifest.get("sources", {})
    provider_request_count = (
        int(sources.get("eastmoney", 0)) + 2 * int(sources.get("tencent", 0)) + 2 * len(failures)
    )
    directory = settings.date_dir(target_date)
    market_hash = write_immutable_bytes(
        directory / "market.csv", canonical_frame_bytes(market, ["date", "symbol"])
    )
    failure_hash = write_immutable_bytes(
        directory / "market_failures.csv",
        canonical_frame_bytes(failure_frame, ["symbol"])
        if not failure_frame.empty
        else (b"\xef\xbb\xbfsymbol,source,error\n"),
    )
    receipt = {
        "target_date": target_date,
        "acquired_at_utc": _utc(now),
        "request_start_date": start,
        "request_end_date": target_date,
        "requested_symbols": len(requested),
        "provider_sources": sources or {"injected_or_unreported": len(requested)},
        "provider_request_count": provider_request_count,
        "provider_fallback_order": ["akshare-eastmoney", "akshare-tencent"],
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
        "manifest_version": "DAILY_PIT_MARKET_V1",
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
        "provider_requests_made_during_manifest_write": 0,
        "provider_requests_made": provider_request_count,
        **policy_hashes(settings),
    }
    manifest_hash = write_immutable_json(directory / "market_manifest.json", manifest)
    return manifest | {"market_manifest_sha256": manifest_hash, "idempotent": False}


def _metadata_for_current(
    market: pd.DataFrame,
    target_date: str,
    symbols: pd.Series,
    settings: DailyPitSettings,
) -> pd.DataFrame:
    target = pd.Timestamp(target_date)
    panel = market[pd.to_datetime(market["date"]).eq(target)].copy()
    membership = load_membership_history(settings.membership_path)
    panel = attach_point_in_time_membership(panel, membership)
    panel = attach_membership_weight(panel, membership)
    panel = attach_extended_fundamentals_asof(
        panel, load_extended_fundamentals(settings.fundamental_path)
    )
    panel = attach_industry_asof(panel, load_industry_history(settings.industry_path))
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
    wanted = set(symbols.astype(str).str.zfill(6))
    panel = panel[panel["symbol"].isin(wanted)].copy()
    panel["eligible"] = True
    return panel


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    stream = io.BytesIO()
    frame.to_parquet(stream, index=False, engine="pyarrow", compression="zstd")
    return stream.getvalue()


def materialize_features(
    target_date: str,
    *,
    settings: DailyPitSettings | None = None,
) -> dict:
    """Create one exact-schema PIT feature partition from immutable market evidence."""
    settings = settings or DailyPitSettings()
    if target_date in settings.permanently_blocked_prediction_dates:
        # Continuity evidence may exist for this date, but its prediction remains permanently absent.
        pass
    directory = settings.date_dir(target_date)
    panel_path = directory / "panel.parquet"
    manifest_path = directory / "manifest.json"
    if panel_path.exists() or manifest_path.exists():
        return verify_daily_feature_partition(target_date, settings=settings) | {"idempotent": True}
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
            combined = frozen[pd.to_datetime(frozen["date"]).le(pd.Timestamp(target_date))].copy()
            stitch_audit = {"passed": True, "mode": "frozen_contains_target"}
        else:
            combined, stitch_audit = stitch_hfq_market(
                frozen,
                incremental,
                membership,
                cutoff=cutoff,
                as_of=target_date,
                settings=settings.forward_settings(),
            )
        reduced, build_audit = build_latest_pit_feature_panel(
            combined, target_date, settings=settings.forward_settings()
        )
        metadata = _metadata_for_current(combined, target_date, reduced["symbol"], settings)
        feature_values = reduced[["date", "symbol", *V10_FEATURES]].copy()
        feature_values["date"] = pd.to_datetime(feature_values["date"]).dt.normalize()
        panel = metadata.merge(
            feature_values, on=["date", "symbol"], how="inner", validate="one_to_one"
        )
        panel = panel[DAILY_FEATURE_COLUMNS].sort_values(["date", "symbol"]).reset_index(drop=True)
    except DailyPitError:
        raise
    except Exception as error:
        text = str(error)
        code = (
            "PIT_JOIN_INVALID"
            if any(
                token in text.lower() for token in ("membership", "fundamental", "industry", "pit")
            )
            else "TARGET_DATE_FEATURE_MATERIALIZATION_FAILED"
        )
        raise DailyPitError(code, text) from error
    try:
        _validate_daily_panel(panel, target_date)
    except Exception as error:
        raise DailyPitError("PIT_JOIN_INVALID", str(error)) from error
    panel_hash = write_immutable_bytes(panel_path, _parquet_bytes(panel))
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


def _validate_daily_panel(panel: pd.DataFrame, target_date: str) -> None:
    if list(panel.columns) != DAILY_FEATURE_COLUMNS or len(set(panel.columns)) != 71:
        raise ValueError("EXACT_71_COLUMN_SCHEMA_REQUIRED")
    if panel.empty or panel.duplicated(["date", "symbol"]).any():
        raise ValueError("EMPTY_OR_DUPLICATE_TARGET_PANEL")
    decision = pd.to_datetime(panel["date"], errors="raise").dt.normalize()
    if not decision.eq(pd.Timestamp(target_date)).all():
        raise ValueError("TARGET_DATE_MISMATCH")
    joins = {
        "membership": pd.to_datetime(panel["membership_snapshot_date"], errors="coerce"),
        "fundamental": pd.to_datetime(panel["available_date"], errors="coerce"),
        "industry": pd.to_datetime(panel["industry_effective_date"], errors="coerce"),
    }
    if any(values.isna().any() or not values.le(decision).all() for values in joins.values()):
        raise ValueError("PIT_ASOF_VIOLATION")
    if (
        not panel["eligible"].isin([True, False]).all()
        or not panel["in_universe"].isin([True, False]).all()
    ):
        raise ValueError("BOOLEAN_GATE_INVALID")
    numeric = panel[V10_FEATURES].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("MODEL_FEATURES_NOT_FINITE")


def verify_daily_feature_partition(
    target_date: str, *, settings: DailyPitSettings | None = None
) -> dict:
    settings = settings or DailyPitSettings()
    directory = settings.date_dir(target_date)
    try:
        manifest = read_verified_json(directory / "manifest.json")
        manifest_hash = verify_immutable(directory / "manifest.json")
        panel_hash = verify_immutable(directory / "panel.parquet")
    except Exception as error:
        raise DailyPitError("DAILY_FEATURE_MANIFEST_INVALID", str(error)) from error
    if manifest.get("target_date") != target_date or manifest.get("panel_sha256") != panel_hash:
        raise DailyPitError("DAILY_FEATURE_MANIFEST_INVALID", "TARGET_OR_PANEL_HASH")
    expected_hashes = policy_hashes(settings)
    if any(manifest.get(key) != value for key, value in expected_hashes.items()):
        raise DailyPitError("DAILY_FEATURE_MANIFEST_INVALID", "POLICY_OR_SCHEMA_HASH")
    for name, expected in manifest.get("source_hashes", {}).items():
        path = Path(name)
        actual = sha256_file(path) if path.is_file() else "MISSING"
        if actual != expected:
            raise DailyPitError("DAILY_FEATURE_MANIFEST_INVALID", f"SOURCE_HASH:{name}")
    try:
        panel = pd.read_parquet(directory / "panel.parquet")
        _validate_daily_panel(panel, target_date)
    except Exception as error:
        raise DailyPitError("DAILY_FEATURE_MANIFEST_INVALID", str(error)) from error
    if len(panel) != manifest.get("rows"):
        raise DailyPitError("DAILY_FEATURE_MANIFEST_INVALID", "ROW_COUNT")
    return manifest | {
        "manifest_sha256": manifest_hash,
        "verified": True,
        "provider_requests_made": 0,
    }
