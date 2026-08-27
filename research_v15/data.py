from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research_v10.fundamentals import attach_extended_fundamentals_asof, load_extended_fundamentals
from research_v14.features import load_announcements
from research_v9.data import attach_industry_asof, attach_membership_weight, load_industry_history
from stockpilot.data import load_panel
from stockpilot.membership import attach_point_in_time_membership, load_membership_history

from .features import (
    EVENT_LABEL_ENDS,
    EVENT_TARGETS,
    build_event_documents,
    build_v15_dataset,
    save_event_documents,
)


def load_v15_dataset(
    market_path: str | Path = "data/market_history_v10_hfq.csv",
    membership_path: str | Path = "data/universes/000300/history_v10.csv",
    fundamental_path: str | Path = "data/fundamentals_pit_v10_extended.csv",
    industry_path: str | Path = "data/industry_history_v10.csv",
) -> pd.DataFrame:
    membership = load_membership_history(membership_path)
    panel = attach_point_in_time_membership(load_panel(market_path), membership)
    panel = attach_membership_weight(panel, membership)
    panel = attach_extended_fundamentals_asof(
        panel, load_extended_fundamentals(fundamental_path)
    )
    panel = attach_industry_asof(panel, load_industry_history(industry_path))
    return build_v15_dataset(panel)


def build_v15_event_data(
    announcement_path: str | Path = "data/announcements_pit_v14.csv",
    output_path: str | Path = "data/event_documents_pit_v15.csv",
    quality_path: str | Path = "artifacts/research_v15/data_quality.json",
) -> dict:
    if Path(output_path).exists() or Path(quality_path).exists():
        raise RuntimeError("V15原始事件数据或报告已存在，禁止覆盖；使用audit-data进行独立修订验收")
    dataset = load_v15_dataset()
    events = build_event_documents(dataset, load_announcements(announcement_path))
    save_event_documents(events, output_path)
    complete = events[[*EVENT_TARGETS, *EVENT_LABEL_ENDS]].notna().all(axis=1)
    report = {
        "dataset_rows": int(len(dataset)),
        "dataset_symbols": int(dataset["symbol"].nunique()),
        "dataset_date_min": str(pd.to_datetime(dataset["date"]).min().date()),
        "dataset_date_max": str(pd.to_datetime(dataset["date"]).max().date()),
        "event_documents": int(len(events)),
        "event_symbols": int(events["symbol"].nunique()),
        "event_date_min": str(pd.to_datetime(events["date"]).min().date()),
        "event_date_max": str(pd.to_datetime(events["date"]).max().date()),
        "complete_three_target_ratio": float(complete.mean()),
        "duplicate_symbol_date_keys": int(events.duplicated(["symbol", "date"]).sum()),
        "raw_event_years": sorted(pd.to_datetime(events["date"]).dt.year.unique().astype(int).tolist()),
    }
    gates = {
        "minimum_250000_documents": report["event_documents"] >= 250000,
        "minimum_750_symbols": report["event_symbols"] >= 750,
        "observed_by_2017_01_31": report["event_date_min"] <= "2017-01-31",
        "complete_three_target_ratio_at_least_95pct": report["complete_three_target_ratio"] >= 0.95,
        "unique_symbol_date_keys": report["duplicate_symbol_date_keys"] == 0,
    }
    report["gates"] = gates
    report["passed"] = all(gates.values())
    target = Path(quality_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
