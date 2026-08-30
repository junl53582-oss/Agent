from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pit_data_v1 import core as parent


@dataclass(frozen=True)
class AdmissionSettings(parent.ObservationSettings):
    version: str = "pit-data-v1r3"
    data_root: Path = Path("data/pit_observations_v1r3")
    artifact_root: Path = Path("artifacts/pit_data_v1r3")
    parent_data_root: Path = Path("data/pit_observations_v1r2")
    parent_artifact_root: Path = Path("artifacts/pit_data_v1r2")
    parent_observation_id: str = "20260830T051947935413Z"


def load_raw_pages(directory: str | Path) -> list[tuple[bytes, dict]]:
    paths = sorted(Path(directory).glob("page_*.json"))
    if not paths:
        raise ValueError("no frozen expectation pages are available")
    pages: list[tuple[bytes, dict]] = []
    for path in paths:
        raw = path.read_bytes()
        body = json.loads(raw)
        if not isinstance(body.get("result", {}).get("data"), list):
            raise ValueError(f"invalid expectation page schema: {path.name}")
        pages.append((raw, body))
    return pages


def normalize_exact_duplicate_expectations(
    pages: list[tuple[bytes, dict]], watchlist: set[str], observed_at: datetime
) -> tuple[pd.DataFrame, dict]:
    """Admit only byte-semantically identical provider duplicates.

    A symbol with two different full provider records is ambiguous and fails closed.
    Every accepted row retains all source page hashes and its full provider-record hash.
    """
    grouped: dict[str, list[tuple[dict, str, str]]] = {}
    raw_rows = 0
    for raw_page, body in pages:
        page_hash = parent.sha256_bytes(raw_page)
        for record in body["result"]["data"]:
            raw_rows += 1
            symbol = str(record.get("SECURITY_CODE") or "").zfill(6)
            if symbol not in watchlist:
                continue
            record_hash = parent.canonical_hash(record)
            grouped.setdefault(symbol, []).append((record, page_hash, record_hash))
    if not grouped:
        raise ValueError("expectation snapshot has no PIT-watchlist intersection")

    rows: list[dict] = []
    duplicate_symbols = 0
    duplicate_rows_removed = 0
    for symbol, records in sorted(grouped.items()):
        record_hashes = {item[2] for item in records}
        if len(record_hashes) != 1:
            raise ValueError(f"conflicting expectation records for symbol {symbol}")
        if len(records) > 1:
            duplicate_symbols += 1
            duplicate_rows_removed += len(records) - 1
        record = records[0][0]
        page_hashes = sorted({item[1] for item in records})
        item = {
            "symbol": symbol,
            "name": record.get("SECURITY_NAME_ABBR"),
            "rating_org_count": record.get("RATING_ORG_NUM"),
            "rating_buy_count": record.get("RATING_BUY_NUM"),
            "rating_add_count": record.get("RATING_ADD_NUM"),
            "rating_neutral_count": record.get("RATING_NEUTRAL_NUM"),
            "rating_reduce_count": record.get("RATING_REDUCE_NUM"),
            "rating_sell_count": record.get("RATING_SALE_NUM"),
            "forecast_year_1": record.get("YEAR1"),
            "forecast_eps_1": record.get("EPS1"),
            "forecast_year_2": record.get("YEAR2"),
            "forecast_eps_2": record.get("EPS2"),
            "forecast_year_3": record.get("YEAR3"),
            "forecast_eps_3": record.get("EPS3"),
            "forecast_year_4": record.get("YEAR4"),
            "forecast_eps_4": record.get("EPS4"),
            "target_price_min": record.get("DEC_AIMPRICEMIN"),
            "target_price_max": record.get("DEC_AIMPRICEMAX"),
            "provider_industry": record.get("INDUSTRY_BOARD"),
            "raw_page_sha256": ";".join(page_hashes),
            "raw_page_count": len(page_hashes),
            "raw_duplicate_count": len(records) - 1,
            "provider_record_sha256": records[0][2],
        }
        item["identity_sha256"] = parent.canonical_hash(item)
        rows.append(item)

    result = pd.DataFrame(rows)
    if result["symbol"].duplicated().any():
        raise AssertionError("duplicate admission invariant failed")
    result.insert(0, "observed_at_utc", observed_at.astimezone(timezone.utc).isoformat())
    audit = {
        "raw_rows": raw_rows,
        "watchlist_rows_before_deduplication": sum(len(value) for value in grouped.values()),
        "accepted_unique_symbols": len(result),
        "duplicate_symbols": duplicate_symbols,
        "duplicate_rows_removed": duplicate_rows_removed,
        "conflicting_duplicate_symbols": 0,
        "missing_watchlist_symbols": sorted(watchlist - set(grouped)),
        "raw_page_sha256": [parent.sha256_bytes(raw) for raw, _ in pages],
    }
    return result.sort_values("symbol").reset_index(drop=True), audit


def _write_outputs(destination: Path, expectations: pd.DataFrame, prosperity: pd.DataFrame, manifest: dict) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    expectations.to_csv(destination / "expectations.csv", index=False, encoding="utf-8-sig")
    prosperity.to_csv(destination / "industry_prosperity.csv", index=False, encoding="utf-8-sig")
    parent._write_new(
        destination / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
    )


def admit_parent_expectations(settings: AdmissionSettings | None = None) -> dict:
    """Normalize a frozen prospective raw capture without contacting any provider."""
    from .freeze import verify_lock

    settings = settings or AdmissionSettings()
    lock = verify_lock(settings)
    parent_manifest_path = settings.parent_data_root / settings.parent_observation_id / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent_manifest["observation_id"] != settings.parent_observation_id:
        raise RuntimeError("parent observation identity changed")
    if parent_manifest["sources"]["expectations"]["status"] != "failed":
        raise RuntimeError("parent expectation failure evidence changed")
    observed_at = datetime.fromisoformat(parent_manifest["observed_at_utc"])
    target_date = parent_manifest["observed_date_shanghai"]
    snapshot, watchlist = parent.load_watchlist(settings.membership_path, target_date)
    if snapshot != parent_manifest["membership_snapshot"]:
        raise RuntimeError("parent PIT membership snapshot changed")

    pages = load_raw_pages(
        settings.parent_data_root / settings.parent_observation_id / "raw" / "expectations"
    )
    expectations, duplicate_audit = normalize_exact_duplicate_expectations(pages, watchlist, observed_at)
    expectations = parent.attach_pit_industry(expectations, settings.industry_path, target_date)
    prosperity = parent.industry_prosperity(expectations, previous=None)
    coverage = expectations["symbol"].nunique() / len(watchlist)
    destination = settings.data_root / settings.parent_observation_id
    if destination.exists():
        raise RuntimeError("V1r3 repaired observation is immutable and already exists")

    sources = {
        "earnings_expectations": {
            "status": "complete",
            "rows": len(expectations),
            "coverage": coverage,
            "source": parent.EXPECTATION_URL,
            "prospective_capture_verified": coverage >= settings.minimum_expectation_coverage,
            "duplicate_audit": duplicate_audit,
            "model_training_ready": False,
        },
        "industry_prosperity": {
            "status": "complete",
            "rows": len(prosperity),
            "method": "PIT-industry aggregation of prospectively captured forecast levels",
            "has_prior_snapshot": False,
            "revision_rows": 0,
            "effective_date_verified": True,
            "model_training_ready": False,
        },
        "fund_flows": {
            **parent_manifest["sources"]["fund_flows"],
            "carried_from_parent_observation": settings.parent_observation_id,
            "retried": False,
        },
    }
    manifest = {
        "version": settings.version,
        "observation_id": settings.parent_observation_id,
        "observed_at_utc": parent_manifest["observed_at_utc"],
        "admitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "observed_date_shanghai": target_date,
        "membership_snapshot": snapshot,
        "watchlist_size": len(watchlist),
        "status": "partial",
        "completed_sources": ["earnings_expectations", "industry_prosperity"],
        "failed_sources": ["fund_flows"],
        "sources": sources,
        "parent_observation_preserved": True,
        "offline_repair_only": True,
        "network_requests": 0,
        "partial_capture_preserved": True,
        "prospective_expectations_verified": sources["earnings_expectations"]["prospective_capture_verified"],
        "prospective_all_sources_verified": False,
        "historical_pit_verified": False,
        "labels_matured": False,
        "minimum_observations_required": settings.minimum_training_observations,
        "completed_prospective_observations": 1,
        "model_training_ready": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
        "frozen_inputs_intact": True,
        "lock_sha256": lock["lock_sha256"],
    }
    _write_outputs(destination, expectations, prosperity, manifest)
    report = settings.artifact_root / "observations" / f"{settings.parent_observation_id}.json"
    parent._write_new(report, json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    return manifest
