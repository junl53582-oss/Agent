from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data_config import V14DataSettings


def _source_report(path: Path, date_column: str, key_columns: list[str]) -> dict:
    frame = pd.read_csv(path, dtype={"symbol": str})
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame[date_column] = pd.to_datetime(frame[date_column], format="mixed", errors="coerce")
    valid = frame.dropna(subset=[date_column])
    return {
        "rows": len(frame),
        "symbols": int(valid["symbol"].nunique()),
        "date_min": str(valid[date_column].min().date()) if len(valid) else None,
        "date_max": str(valid[date_column].max().date()) if len(valid) else None,
        "invalid_dates": int(frame[date_column].isna().sum()),
        "duplicates": int(frame.duplicated(key_columns).sum()),
        "future_dates": int((frame[date_column].dt.normalize() > pd.Timestamp("2026-08-25")).sum()),
    }


def audit_v14_external(
    settings: V14DataSettings | None = None,
    output: str | Path = "artifacts/research_v14/data_quality.json",
) -> dict:
    settings = settings or V14DataSettings()
    analyst = _source_report(
        settings.analyst_output,
        "report_date",
        ["symbol", "report_date", "title", "institution"],
    )
    northbound = _source_report(
        settings.northbound_output,
        "holding_date",
        ["symbol", "holding_date"],
    )
    announcements = _source_report(
        settings.announcement_output,
        "announcement_date",
        ["symbol", "announcement_id"],
    )
    gates = {
        "analyst": {
            "symbol_coverage": analyst["symbols"] >= 400,
            "starts_by_2018": analyst["date_min"] is not None and analyst["date_min"] <= "2018-01-01",
            "dates_clean": analyst["invalid_dates"] == analyst["future_dates"] == 0,
            "unique": analyst["duplicates"] == 0,
        },
        "northbound": {
            "symbol_coverage": northbound["symbols"] >= 350,
            "starts_by_2018": northbound["date_min"] is not None and northbound["date_min"] <= "2018-01-01",
            "known_stop_not_forward_filled": northbound["date_max"] is not None and northbound["date_max"] <= "2024-08-16",
            "dates_clean": northbound["invalid_dates"] == northbound["future_dates"] == 0,
            "unique": northbound["duplicates"] == 0,
        },
        "announcements": {
            "symbol_coverage": announcements["symbols"] >= 700,
            "query_starts_by_2017": settings.start_date <= "2017-01-01",
            "observed_in_january_2017": announcements["date_min"] is not None and announcements["date_min"] <= "2017-01-31",
            "dates_clean": announcements["invalid_dates"] == announcements["future_dates"] == 0,
            "unique": announcements["duplicates"] == 0,
        },
    }
    accepted = [name for name, values in gates.items() if all(values.values())]
    report = {
        "analyst": analyst,
        "northbound": northbound,
        "announcements": announcements,
        "gates": gates,
        "accepted_sources": accepted,
        "rejected_sources": [name for name in gates if name not in accepted],
        "at_least_one_new_source_passed": bool(accepted),
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
