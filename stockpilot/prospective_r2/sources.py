from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from announcement_body_v5r2 import source as announcement_source
from pit_data_v1 import core as pit_parent
from pit_data_v1r1.source import fetch_flow_pages
from pit_data_v1r3.core import normalize_exact_duplicate_expectations

from .config import OperationalSettings
from .integrity import canonical_frame_bytes, sha256_bytes, sha256_file
from .observation import SourceCapture, SourceUnavailableError


def load_pit_context(
    target_date: str, settings: OperationalSettings
) -> tuple[pd.DataFrame, dict]:
    snapshot_date, symbols = pit_parent.load_watchlist(settings.membership_path, target_date)
    industry = pd.read_csv(settings.industry_path, dtype={"symbol": str})
    required = {"symbol", "industry", "industry_effective_date"}
    missing = required - set(industry.columns)
    if missing:
        raise ValueError(f"PIT industry history missing {sorted(missing)}")
    industry["symbol"] = industry["symbol"].str.zfill(6)
    industry["industry_effective_date"] = pd.to_datetime(
        industry["industry_effective_date"]
    ).dt.normalize()
    cutoff = pd.Timestamp(target_date).normalize()
    eligible = industry[
        industry["symbol"].isin(symbols)
        & industry["industry_effective_date"].le(cutoff)
    ].sort_values(["symbol", "industry_effective_date"])
    latest = eligible.groupby("symbol", as_index=False).tail(1)
    panel = pd.DataFrame({"symbol": sorted(symbols)})
    panel = panel.merge(
        latest[["symbol", "industry", "industry_effective_date"]],
        on="symbol",
        how="left",
        validate="one_to_one",
    )
    if panel["industry"].isna().any():
        raise ValueError("PIT industry mapping is incomplete")
    panel["universe_member"] = True
    membership_rows = pd.read_csv(settings.membership_path, dtype={"symbol": str})
    membership_rows["symbol"] = membership_rows["symbol"].str.zfill(6)
    snapshot_rows = membership_rows[
        membership_rows["snapshot_date"].astype(str).eq(snapshot_date)
    ].copy()
    membership_hash = sha256_bytes(
        canonical_frame_bytes(snapshot_rows, ["snapshot_date", "symbol"])
    )
    industry_hash = sha256_bytes(
        canonical_frame_bytes(
            panel[["symbol", "industry", "industry_effective_date"]], ["symbol"]
        )
    )
    proof = {
        "membership_snapshot_date": snapshot_date,
        "membership_source_path": settings.membership_path.as_posix(),
        "membership_source_sha256": sha256_file(settings.membership_path),
        "membership_snapshot_sha256": membership_hash,
        "industry_source_path": settings.industry_path.as_posix(),
        "industry_source_sha256": sha256_file(settings.industry_path),
        "industry_mapping_sha256": industry_hash,
        "universe_size": len(panel),
    }
    return panel, proof


def earnings_capture(
    universe: set[str],
    target_date: str,
    observed_at: datetime,
    settings: OperationalSettings,
) -> SourceCapture:
    pages = pit_parent.fetch_expectation_pages()
    frame, duplicate_audit = normalize_exact_duplicate_expectations(
        pages, universe, observed_at
    )
    frame = pit_parent.attach_pit_industry(
        frame, settings.industry_path, target_date
    )
    return SourceCapture(
        source="earnings_expectations",
        request_parameters={
            "endpoint": pit_parent.EXPECTATION_URL,
            "target_date": target_date,
            "automatic_retries": 0,
        },
        raw_payloads=tuple(raw for raw, _ in pages),
        normalized=frame,
        required_value_columns=("forecast_eps_1",),
        duplicate_count=int(duplicate_audit["duplicate_rows_removed"]),
        conflicting_duplicate_count=int(
            duplicate_audit["conflicting_duplicate_symbols"]
        ),
        missing_symbols=tuple(duplicate_audit["missing_watchlist_symbols"]),
        network_request_count=len(pages),
    )


def announcement_capture(
    universe: set[str], target_date: str, settings: OperationalSettings
) -> SourceCapture:
    org_ids = announcement_source.load_org_ids(
        settings.announcement_org_metadata_path, sorted(universe)
    )
    raw_pages, manifest = announcement_source.fetch_watchlist(
        target_date, sorted(universe), org_ids
    )
    partitions = {item["symbol"]: item for item in manifest["partitions"]}
    if set(partitions) != universe:
        raise ValueError("announcement query did not confirm every PIT symbol")
    frame = pd.DataFrame(
        {
            "symbol": sorted(universe),
            "announcement_event_count": [
                int(partitions[symbol]["reported_total"]) for symbol in sorted(universe)
            ],
            "announcement_available": True,
        }
    )
    return SourceCapture(
        source="announcements",
        request_parameters={
            "endpoint": manifest["endpoint"],
            "target_date": target_date,
            "partition": manifest["partition"],
            "automatic_retries": 0,
        },
        raw_payloads=tuple(raw_pages),
        normalized=frame,
        required_value_columns=("announcement_event_count",),
        confirmed_symbols=tuple(sorted(universe)),
        network_request_count=sum(int(item["pages"]) for item in manifest["partitions"]),
    )


def fund_flow_capture(
    universe: set[str], observed_at: datetime
) -> SourceCapture:
    try:
        pages = fetch_flow_pages()
    except Exception as error:
        # This is the approved provider only.  No alternate provider is attempted.
        unavailable = SourceUnavailableError(str(error))
        unavailable.network_request_count = int(getattr(error, "network_request_count", 0))
        raise unavailable from error
    frame = pit_parent.normalize_flows(pages, universe, observed_at)
    return SourceCapture(
        source="fund_flows",
        request_parameters={
            "endpoint": pit_parent.FLOW_URL,
            "automatic_retries": 0,
            "fallback_provider": None,
        },
        raw_payloads=tuple(raw for raw, _ in pages),
        normalized=frame,
        required_value_columns=("main_net_inflow", "main_net_inflow_ratio"),
        network_request_count=len(pages),
    )


def production_source_fetchers(
    universe: set[str],
    target_date: str,
    observed_at: datetime,
    settings: OperationalSettings,
) -> dict:
    return {
        "earnings_expectations": (
            {"endpoint": pit_parent.EXPECTATION_URL, "target_date": target_date},
            lambda: earnings_capture(universe, target_date, observed_at, settings),
        ),
        "announcements": (
            {"endpoint": announcement_source.ENDPOINT, "target_date": target_date},
            lambda: announcement_capture(universe, target_date, settings),
        ),
        "fund_flows": (
            {"endpoint": pit_parent.FLOW_URL, "target_date": target_date},
            lambda: fund_flow_capture(universe, observed_at),
        ),
    }


def load_normalized_source(receipt: dict) -> pd.DataFrame | None:
    if not receipt.get("success"):
        return None
    path = Path(receipt["normalized_path"])
    from .integrity import verify_immutable

    if verify_immutable(path) != receipt["normalized_data_sha256"]:
        raise RuntimeError("normalized source hash changed")
    return pd.read_csv(path, dtype={"symbol": str})
