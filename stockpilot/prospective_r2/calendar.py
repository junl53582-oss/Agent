from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .integrity import sha256_file


@dataclass(frozen=True)
class VerifiedTradingCalendar:
    market: str
    start: pd.Timestamp
    end: pd.Timestamp
    holidays: frozenset[pd.Timestamp]
    source: str
    source_url: str
    file_sha256: str

    def is_session(self, value: str | pd.Timestamp) -> bool:
        date = pd.Timestamp(value).normalize()
        return (
            self.start <= date <= self.end
            and date.weekday() < 5
            and date not in self.holidays
        )

    def sessions(self) -> pd.DatetimeIndex:
        weekdays = pd.bdate_range(self.start, self.end)
        return weekdays[~weekdays.isin(list(self.holidays))]


def load_verified_calendar(path: str | Path) -> VerifiedTradingCalendar:
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    required = {"market", "coverage_start", "coverage_end", "closed_weekdays", "source", "source_url"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"trading calendar metadata missing: {sorted(missing)}")
    if payload["market"] != "XSHG" or payload.get("weekends_closed") is not True:
        raise ValueError("calendar is not a verified Shanghai exchange calendar")
    holidays = frozenset(pd.Timestamp(value).normalize() for value in payload["closed_weekdays"])
    if any(value.weekday() >= 5 for value in holidays):
        raise ValueError("closed_weekdays must not contain weekends")
    return VerifiedTradingCalendar(
        market=payload["market"],
        start=pd.Timestamp(payload["coverage_start"]).normalize(),
        end=pd.Timestamp(payload["coverage_end"]).normalize(),
        holidays=holidays,
        source=payload["source"],
        source_url=payload["source_url"],
        file_sha256=sha256_file(target),
    )


def validate_current_session(
    target_date: str,
    actual_shanghai_date: str,
    calendar: VerifiedTradingCalendar,
) -> None:
    target = pd.Timestamp(target_date).normalize()
    actual = pd.Timestamp(actual_shanghai_date).normalize()
    if target < actual:
        raise ValueError("historical prospective backfill is forbidden")
    if target > actual:
        raise ValueError("future prospective observation is forbidden")
    if not calendar.is_session(target):
        raise ValueError("target date is not a verified Shanghai trading date")
