from __future__ import annotations

import pandas as pd

from stockpilot.prospective_r2.sources import load_pit_context
from stockpilot.prospective_r3.settlement import (
    SettlementBundle,
    certify_label_record,
    load_approved_settlement_bundle,
    load_predictions,
    settle_certified_labels,
)

from .benchmark import verify_benchmark_evidence
from .config import OperationalSettings


def load_operational_settlement_bundle(
    settings: OperationalSettings,
    *,
    lock_verifier=None,
    as_of: str | None = None,
) -> SettlementBundle:
    if lock_verifier is None:
        from .freeze import verify_lock

        lock_verifier = verify_lock
    bundle = load_approved_settlement_bundle(settings, lock_verifier=lock_verifier)
    if bundle.benchmark_status == "APPROVED":
        from stockpilot.prospective_r2.integrity import read_verified_json

        manifest = read_verified_json(settings.settlement_manifest_path)
        verify_benchmark_evidence(manifest["benchmark"], as_of=as_of or "2100-01-01")
    return bundle


def certify_operational_label(record: dict, settings: OperationalSettings) -> dict:
    bundle = load_operational_settlement_bundle(settings, as_of=record.get("maturity_date"))
    return certify_label_record(record, settings, bundle=bundle)


def run_operational_settlement(target_date: str, settings: OperationalSettings) -> dict:
    bundle = load_operational_settlement_bundle(settings, as_of=target_date)
    if bundle.market_status != "APPROVED":
        return {"status": "SETTLEMENT_BLOCKED_MARKET_UNAPPROVED", "mature_records_written": 0}
    if bundle.benchmark_status != "APPROVED":
        return {"status": "SETTLEMENT_BLOCKED_BENCHMARK_UNAPPROVED", "mature_records_written": 0}
    predictions = load_predictions(settings.prediction_root)
    expected: dict[str, set[str]] = {}
    for value in predictions.get("date", pd.Series(dtype=str)).astype(str).unique():
        panel, _ = load_pit_context(value, settings)
        expected[value] = set(panel["symbol"].astype(str).str.zfill(6))
    records = settle_certified_labels(
        predictions,
        bundle=bundle,
        settings=settings,
        as_of=target_date,
        expected_universe_by_date=expected,
    )
    return {
        "status": "SETTLED" if records else "NO_MATURE_LABELS",
        "mature_records_written": len(records),
        "bundle_manifest_sha256": bundle.manifest_sha256,
    }
