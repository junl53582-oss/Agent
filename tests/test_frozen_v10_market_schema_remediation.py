from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from stockpilot.daily_pit import (
    DailyPitError,
    DailyPitSettings,
    acquire_market,
    materialize_features,
)
from stockpilot.prospective_r2.integrity import sha256_file

TARGET = "2026-09-02"
SHANGHAI = ZoneInfo("Asia/Shanghai")
REQUIRED_MARKET_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]


def _market(symbols: list[str], *, complete: bool) -> pd.DataFrame:
    rows = []
    for position, symbol in enumerate(symbols):
        for offset, date in enumerate(pd.bdate_range("2026-07-01", TARGET)):
            price = 10.0 + position + offset * 0.01
            row = {
                "date": date,
                "symbol": symbol,
                "open": price,
                "close": price * 1.001,
                "volume": 1_000.0,
            }
            if complete:
                row |= {
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "amount": 10_000.0,
                }
            rows.append(row)
    columns = REQUIRED_MARKET_COLUMNS if complete else [
        "date",
        "symbol",
        "open",
        "close",
        "volume",
    ]
    return pd.DataFrame(rows)[columns]


def _settings(tmp_path: Path, symbols: list[str]) -> DailyPitSettings:
    calendar = tmp_path / "calendar.json"
    calendar.write_text(
        json.dumps(
            {
                "market": "XSHG",
                "coverage_start": "2026-08-01",
                "coverage_end": "2026-09-30",
                "closed_weekdays": [],
                "weekends_closed": True,
                "source": "test-fixture",
                "source_url": "https://example.invalid/calendar",
            }
        ),
        encoding="utf-8",
    )
    membership = tmp_path / "membership.csv"
    pd.DataFrame(
        {
            "snapshot_date": ["2026-08-31"] * len(symbols),
            "index_code": ["000300"] * len(symbols),
            "symbol": symbols,
            "weight": [1 / len(symbols)] * len(symbols),
            "source": ["test"] * len(symbols),
        }
    ).to_csv(membership, index=False)
    frozen = tmp_path / "frozen.csv"
    _market(symbols, complete=False).to_csv(frozen, index=False)
    fundamentals = tmp_path / "fundamentals.csv"
    industry = tmp_path / "industry.csv"
    names = tmp_path / "names.csv"
    fundamentals.write_text("fixture", encoding="utf-8")
    industry.write_text("fixture", encoding="utf-8")
    names.write_text("symbol,name\n", encoding="utf-8")
    return DailyPitSettings(
        root=tmp_path / "daily",
        calendar_path=calendar,
        frozen_market_path=frozen,
        membership_path=membership,
        fundamental_path=fundamentals,
        industry_path=industry,
        names_path=names,
        minimum_universe_coverage=1.0,
    )


def test_incomplete_frozen_history_fails_closed_without_synthetic_repair(
    tmp_path: Path,
) -> None:
    symbols = ["000001", "000002", "600000"]
    settings = _settings(tmp_path, symbols)

    def fetcher(*args, **kwargs):
        del args, kwargs
        return _market(symbols, complete=True), []

    now = datetime(2026, 9, 2, 19, 0, tzinfo=SHANGHAI)
    acquire_market(TARGET, symbols, now=now, settings=settings, fetcher=fetcher)
    frozen_hash = sha256_file(settings.frozen_market_path)
    market_path = settings.date_dir(TARGET) / "market.csv"
    market_hash = sha256_file(market_path)

    with pytest.raises(
        DailyPitError,
        match="TARGET_DATE_FEATURE_MATERIALIZATION_FAILED:.*amount, high, low",
    ):
        materialize_features(TARGET, settings=settings)

    assert list(pd.read_csv(settings.frozen_market_path, nrows=0).columns) == [
        "date",
        "symbol",
        "open",
        "close",
        "volume",
    ]
    assert list(pd.read_csv(market_path, nrows=0).columns) == REQUIRED_MARKET_COLUMNS
    assert sha256_file(settings.frozen_market_path) == frozen_hash
    assert sha256_file(market_path) == market_hash
    assert not (settings.date_dir(TARGET) / "panel.parquet").exists()
    assert not (settings.date_dir(TARGET) / "manifest.json").exists()
