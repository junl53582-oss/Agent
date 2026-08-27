from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_v4.lock import sha256_file

from .features import EVENT_LABEL_ENDS, EVENT_TARGETS, load_event_documents, raw_event_years

QUALITY_PATH = "artifacts/research_v15/data_quality.amended.json"
AMENDMENT_PATH = "artifacts/research_v15/amendment_001.json"
ORIGINAL_QUALITY_PATH = "artifacts/research_v15/data_quality.json"


def event_quality_report(events: pd.DataFrame) -> dict:
    eligible = events["eligible"].eq(True)
    dates = pd.to_datetime(events["date"], errors="coerce")
    ends = events[EVENT_LABEL_ENDS].apply(pd.to_datetime, errors="coerce")
    targets = events[EVENT_TARGETS].apply(pd.to_numeric, errors="coerce")
    chronology_bad = pd.Series(False, index=events.index)
    for column in EVENT_LABEL_ENDS:
        chronology_bad |= ends[column].notna() & (ends[column] <= dates)
    chronology_bad |= ends.iloc[:, 0].notna() & ends.iloc[:, 1].notna() & (ends.iloc[:, 0] > ends.iloc[:, 1])
    chronology_bad |= ends.iloc[:, 1].notna() & ends.iloc[:, 2].notna() & (ends.iloc[:, 1] > ends.iloc[:, 2])
    complete = np.isfinite(targets).all(axis=1) & ends.notna().all(axis=1) & ~chronology_bad
    denominator = int(eligible.sum())
    numerator = int((eligible & complete).sum())
    report = {
        "event_documents": len(events),
        "event_symbols": int(events["symbol"].nunique()),
        "event_date_min": str(dates.min().date()) if dates.notna().any() else None,
        "event_date_max": str(dates.max().date()) if dates.notna().any() else None,
        "complete_three_target_ratio": float(complete.mean()) if len(events) else 0.0,
        "eligible_event_documents": denominator,
        "eligible_complete_documents": numerator,
        "eligible_incomplete_documents": denominator - numerator,
        "eligible_complete_three_target_ratio": numerator / denominator if denominator else 0.0,
        "ineligible_documents": int((~eligible).sum()),
        "duplicate_symbol_date_keys": int(events.duplicated(["symbol", "date"]).sum()),
        "invalid_event_dates": int(dates.isna().sum()),
        "invalid_eligibility": int((~events["eligible"].isin([True, False])).sum()),
        "blank_documents": int(events["document"].fillna("").str.strip().eq("").sum()),
        "invalid_event_counts": int((pd.to_numeric(events["event_count"], errors="coerce").fillna(0) <= 0).sum()),
        "label_chronology_violations": int(chronology_bad.sum()),
        "raw_event_years": raw_event_years(events),
        "eligible_completeness_by_year": {},
    }
    for year in sorted(dates[eligible].dt.year.dropna().unique()):
        mask = eligible & dates.dt.year.eq(year)
        report["eligible_completeness_by_year"][str(int(year))] = {
            "eligible": int(mask.sum()), "complete": int((mask & complete).sum()),
            "ratio": float(complete[mask].mean()),
        }
    gates = {
        "minimum_250000_documents": len(events) >= 250000,
        "minimum_750_symbols": report["event_symbols"] >= 750,
        "observed_by_2017_01_31": report["event_date_min"] is not None and report["event_date_min"] <= "2017-01-31",
        "eligible_denominator_nonempty": denominator > 0,
        "eligible_complete_three_target_ratio_at_least_95pct": report["eligible_complete_three_target_ratio"] >= 0.95,
        "unique_symbol_date_keys": report["duplicate_symbol_date_keys"] == 0,
        "dates_valid": report["invalid_event_dates"] == 0,
        "eligibility_valid": report["invalid_eligibility"] == 0,
        "documents_nonempty": report["blank_documents"] == 0,
        "raw_events_positive": report["invalid_event_counts"] == 0,
        "label_chronology_valid": report["label_chronology_violations"] == 0,
    }
    return {**report, "gates": gates, "passed": all(gates.values())}


def audit_event_data(event_path="data/event_documents_pit_v15.csv", root=".") -> dict:
    workspace = Path(root)
    if (workspace / "artifacts/research_v15/plan.lock.json").exists():
        raise RuntimeError("V15已冻结，禁止改写数据验收报告")
    amendment = json.loads((workspace / AMENDMENT_PATH).read_text(encoding="utf-8"))
    original_hash = sha256_file(workspace / ORIGINAL_QUALITY_PATH)
    event_hash = sha256_file(workspace / event_path)
    if original_hash != amendment["original_failed_quality_sha256"] or event_hash != amendment["event_document_sha256"]:
        raise RuntimeError("原始失败记录或事件数据已变化，禁止按既定修订验收")
    report = event_quality_report(load_event_documents(workspace / event_path))
    report.update({
        "amendment_sha256": sha256_file(workspace / AMENDMENT_PATH),
        "original_quality_sha256": original_hash,
        "event_document_sha256": event_hash,
        "performance_seen": False,
    })
    (workspace / QUALITY_PATH).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
