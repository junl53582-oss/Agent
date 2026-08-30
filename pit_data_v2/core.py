from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pit_data_v1 import core as parent
from pit_data_v1r1.source import fetch_flow_pages
from pit_data_v1r2.core import _source_failure
from pit_data_v1r3.core import normalize_exact_duplicate_expectations


@dataclass(frozen=True)
class ObservationSettings(parent.ObservationSettings):
    version: str = "pit-data-v2"
    data_root: Path = Path("data/pit_observations_v2")
    artifact_root: Path = Path("artifacts/pit_data_v2")
    baseline_root: Path = Path("data/pit_observations_v1r3")


def _successful_expectation_manifests(settings: ObservationSettings) -> list[Path]:
    output: list[Path] = []
    for root in (settings.baseline_root, settings.data_root):
        for manifest_path in sorted(root.glob("*/manifest.json")):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source = manifest.get("sources", {}).get("earnings_expectations") or manifest.get("sources", {}).get("expectations")
            if source and source.get("status") == "complete" and (manifest_path.parent / "expectations.csv").exists():
                output.append(manifest_path)
    return sorted(output, key=lambda path: json.loads(path.read_text(encoding="utf-8"))["observed_at_utc"])


def _previous_expectations(settings: ObservationSettings) -> pd.DataFrame | None:
    manifests = _successful_expectation_manifests(settings)
    if not manifests:
        return None
    return pd.read_csv(manifests[-1].parent / "expectations.csv", dtype={"symbol": str})


def _observed_dates(settings: ObservationSettings) -> set[str]:
    dates: set[str] = set()
    for root in (settings.baseline_root, settings.data_root):
        for path in root.glob("*/manifest.json"):
            dates.add(json.loads(path.read_text(encoding="utf-8"))["observed_date_shanghai"])
    return dates


def capture_sources(
    destination: Path,
    *,
    target_date: str,
    now: datetime,
    watchlist: set[str],
    settings: ObservationSettings,
    expectation_fetcher=parent.fetch_expectation_pages,
    flow_fetcher=fetch_flow_pages,
    previous_expectations: pd.DataFrame | None = None,
) -> dict:
    sources: dict[str, dict] = {}
    try:
        pages = expectation_fetcher()
        for index, (raw, _) in enumerate(pages, 1):
            parent._write_new(destination / "raw" / "expectations" / f"page_{index:04d}.json", raw)
        expectations, duplicate_audit = normalize_exact_duplicate_expectations(pages, watchlist, now)
        expectations = parent.attach_pit_industry(expectations, settings.industry_path, target_date)
        prosperity = parent.industry_prosperity(expectations, previous_expectations)
        expectations.to_csv(destination / "expectations.csv", index=False, encoding="utf-8-sig")
        prosperity.to_csv(destination / "industry_prosperity.csv", index=False, encoding="utf-8-sig")
        coverage = expectations["symbol"].nunique() / len(watchlist)
        sources["earnings_expectations"] = {
            "status": "complete",
            "rows": len(expectations),
            "coverage": coverage,
            "source": parent.EXPECTATION_URL,
            "duplicate_audit": duplicate_audit,
            "prospective_capture_verified": coverage >= settings.minimum_expectation_coverage,
            "model_training_ready": False,
        }
        sources["industry_prosperity"] = {
            "status": "complete",
            "rows": len(prosperity),
            "has_prior_snapshot": previous_expectations is not None,
            "revision_rows": int(prosperity["revision_coverage"].sum()),
            "effective_date_verified": True,
            "model_training_ready": False,
        }
    except BaseException as error:
        sources["earnings_expectations"] = _source_failure(error)
        sources["industry_prosperity"] = {
            **_source_failure(RuntimeError("blocked because expectation capture failed")),
            "dependency_error": sources["earnings_expectations"]["error"],
        }

    try:
        pages = flow_fetcher()
        for index, (raw, _) in enumerate(pages, 1):
            parent._write_new(destination / "raw" / "flows" / f"page_{index:04d}.json", raw)
        flows = parent.normalize_flows(pages, watchlist, now)
        flows.to_csv(destination / "fund_flows.csv", index=False, encoding="utf-8-sig")
        coverage = flows["symbol"].nunique() / len(watchlist)
        source_times = pd.to_datetime(flows["source_timestamp_utc"], utc=True, errors="coerce")
        future = int((source_times > pd.Timestamp(now)).sum())
        sources["fund_flows"] = {
            "status": "complete",
            "rows": len(flows),
            "coverage": coverage,
            "source": parent.FLOW_URL,
            "source_timestamp_min": source_times.min().isoformat() if source_times.notna().any() else None,
            "source_timestamp_max": source_times.max().isoformat() if source_times.notna().any() else None,
            "future_source_timestamps": future,
            "raw_page_sha256": [parent.sha256_bytes(item[0]) for item in pages],
            "prospective_capture_verified": coverage >= settings.minimum_flow_coverage and future == 0,
            "model_training_ready": False,
        }
    except BaseException as error:
        sources["fund_flows"] = _source_failure(error)
    return sources


def observe(target_date: str | None = None, *, now: datetime | None = None, settings=None) -> dict:
    from .freeze import verify_lock

    settings = settings or ObservationSettings()
    lock = verify_lock(settings)
    now = now or datetime.now(timezone.utc)
    shanghai_date = now.astimezone(parent.SHANGHAI).date().isoformat()
    target_date = target_date or shanghai_date
    if target_date != shanghai_date:
        raise ValueError("historical backfill is forbidden; target must equal current Shanghai date")
    if target_date in _observed_dates(settings):
        raise RuntimeError(f"a PIT expectation observation already exists for {target_date}")

    observation_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    destination = settings.data_root / observation_id
    destination.mkdir(parents=True, exist_ok=False)
    snapshot, watchlist = parent.load_watchlist(settings.membership_path, target_date)
    previous = _previous_expectations(settings)
    sources = capture_sources(
        destination,
        target_date=target_date,
        now=now,
        watchlist=watchlist,
        settings=settings,
        previous_expectations=previous,
    )
    completed = [name for name, value in sources.items() if value["status"] == "complete"]
    failed = [name for name, value in sources.items() if value["status"] == "failed"]
    expectation_observations = len(_successful_expectation_manifests(settings)) + int(
        sources["earnings_expectations"]["status"] == "complete"
    )
    manifest = {
        "version": settings.version,
        "observation_id": observation_id,
        "observed_at_utc": now.astimezone(timezone.utc).isoformat(),
        "observed_date_shanghai": target_date,
        "membership_snapshot": snapshot,
        "watchlist_size": len(watchlist),
        "status": "complete" if not failed else ("partial" if completed else "failed"),
        "completed_sources": completed,
        "failed_sources": failed,
        "sources": sources,
        "partial_capture_preserved": bool(completed and failed),
        "prospective_expectation_observations": expectation_observations,
        "minimum_observations_required": settings.minimum_training_observations,
        "labels_matured": False,
        "historical_pit_verified": False,
        "model_training_ready": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
        "frozen_inputs_intact": True,
        "lock_sha256": lock["lock_sha256"],
    }
    parent._write_new(destination / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    parent._write_new(
        settings.artifact_root / "observations" / f"{observation_id}.json",
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    return manifest
