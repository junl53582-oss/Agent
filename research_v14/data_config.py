from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class V14DataSettings:
    start_date: str = "2017-01-01"
    end_date: str = "2026-08-25"
    workers: int = 8
    membership_path: Path = Path("data/universes/000300/history_v10.csv")
    cache_dir: Path = Path("data/v14_cache")
    analyst_output: Path = Path("data/analyst_reports_pit_v14.csv")
    northbound_output: Path = Path("data/northbound_holdings_pit_v14.csv")
    announcement_output: Path = Path("data/announcements_pit_v14.csv")

