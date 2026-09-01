from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from research_v10.features import V10_FEATURES
from stockpilot.daily_pit import (
    DAILY_FEATURE_COLUMNS,
    DailyPitError,
    DailyPitSettings,
    acquire_market,
    materialize_features,
    verify_daily_feature_partition,
)
from stockpilot.prospective_r2.integrity import sha256_file

SHANGHAI = ZoneInfo("Asia/Shanghai")
TARGET = "2026-09-02"


def _write_calendar(path: Path) -> None:
    path.write_text(
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


def _market(symbols: list[str], *, target: str = TARGET) -> pd.DataFrame:
    dates = pd.bdate_range("2026-07-01", target)
    rows = []
    for position, symbol in enumerate(symbols):
        for offset, date in enumerate(dates):
            price = 10.0 + position + offset * 0.01
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price * 1.001,
                    "volume": 1000.0,
                    "amount": 10000.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def configured(tmp_path: Path) -> tuple[DailyPitSettings, list[str]]:
    symbols = ["000001", "000002", "600000"]
    calendar = tmp_path / "calendar.json"
    membership = tmp_path / "membership.csv"
    frozen = tmp_path / "frozen.csv"
    fundamentals = tmp_path / "fundamentals.csv"
    industry = tmp_path / "industry.csv"
    names = tmp_path / "names.csv"
    _write_calendar(calendar)
    pd.DataFrame(
        {
            "snapshot_date": ["2026-08-31"] * len(symbols),
            "index_code": ["000300"] * len(symbols),
            "symbol": symbols,
            "weight": [1 / len(symbols)] * len(symbols),
            "source": ["test"] * len(symbols),
        }
    ).to_csv(membership, index=False)
    _market(symbols, target="2026-08-31").to_csv(frozen, index=False)
    fundamentals.write_text("fixture", encoding="utf-8")
    industry.write_text("fixture", encoding="utf-8")
    names.write_text("symbol,name\n", encoding="utf-8")
    return (
        DailyPitSettings(
            root=tmp_path / "daily",
            calendar_path=calendar,
            frozen_market_path=frozen,
            membership_path=membership,
            fundamental_path=fundamentals,
            industry_path=industry,
            names_path=names,
            minimum_universe_coverage=1.0,
        ),
        symbols,
    )


def _now() -> datetime:
    return datetime(2026, 9, 2, 19, 0, tzinfo=SHANGHAI)


def test_acquisition_is_target_bounded_receipted_and_idempotent(configured) -> None:
    settings, symbols = configured
    calls: list[tuple[str, str]] = []

    def fetcher(requested, start, end, **kwargs):
        del requested, kwargs
        calls.append((start, end))
        frame = _market(symbols)
        future = frame[frame["date"].eq(pd.Timestamp(TARGET))].copy()
        future["date"] = pd.Timestamp("2026-09-03")
        return pd.concat([frame, future], ignore_index=True), []

    first = acquire_market(TARGET, symbols, now=_now(), settings=settings, fetcher=fetcher)
    second = acquire_market(TARGET, symbols, now=_now(), settings=settings, fetcher=fetcher)
    stored = pd.read_csv(settings.date_dir(TARGET) / "market.csv")
    receipt = json.loads((settings.date_dir(TARGET) / "source_receipt.json").read_text())
    assert calls == [(calls[0][0], TARGET)]
    assert pd.to_datetime(stored["date"]).max() == pd.Timestamp(TARGET)
    assert receipt["future_market_used"] is False
    assert receipt["previous_day_substituted"] is False
    assert first["target_date"] == TARGET
    assert second["idempotent"] is True
    assert second["provider_requests_made"] == 0


def test_acquisition_missing_target_fails_without_partition(configured) -> None:
    settings, symbols = configured

    def fetcher(*args, **kwargs):
        del args, kwargs
        return _market(symbols, target="2026-09-01"), []

    with pytest.raises(DailyPitError, match="MARKET_DATA_NOT_READY"):
        acquire_market(TARGET, symbols, now=_now(), settings=settings, fetcher=fetcher)
    assert not settings.date_dir(TARGET).exists()


def test_acquisition_insufficient_coverage_fails_closed(configured) -> None:
    settings, symbols = configured

    def fetcher(*args, **kwargs):
        del args, kwargs
        return _market(symbols[:2]), [{"symbol": symbols[-1], "source": "failed", "error": "x"}]

    with pytest.raises(DailyPitError, match="MARKET_COVERAGE_INSUFFICIENT"):
        acquire_market(TARGET, symbols, now=_now(), settings=settings, fetcher=fetcher)
    assert not settings.date_dir(TARGET).exists()


def test_materialization_exact_schema_pit_and_immutable(monkeypatch, configured) -> None:
    settings, symbols = configured

    def fetcher(*args, **kwargs):
        del args, kwargs
        return _market(symbols), []

    acquire_market(TARGET, symbols, now=_now(), settings=settings, fetcher=fetcher)
    target = pd.Timestamp(TARGET)
    reduced = pd.DataFrame({"date": [target] * len(symbols), "symbol": symbols})
    for index, feature in enumerate(V10_FEATURES):
        reduced[feature] = float(index)
    metadata = pd.DataFrame(
        {
            "date": [target] * len(symbols),
            "symbol": symbols,
            "eligible": [True] * len(symbols),
            "in_universe": [True] * len(symbols),
            "membership_snapshot_date": [pd.Timestamp("2026-08-31")] * len(symbols),
            "available_date": [pd.Timestamp("2026-08-25")] * len(symbols),
            "industry_effective_date": [pd.Timestamp("2026-07-01")] * len(symbols),
            "industry": ["电子"] * len(symbols),
            "broad_sector": ["technology"] * len(symbols),
            "benchmark_weight": [1 / len(symbols)] * len(symbols),
        }
    )
    monkeypatch.setattr(
        "stockpilot.daily_pit.pipeline.stitch_hfq_market",
        lambda frozen, incremental, membership, **kwargs: (
            pd.concat([frozen, incremental], ignore_index=True),
            {"passed": True},
        ),
    )
    monkeypatch.setattr(
        "stockpilot.daily_pit.pipeline.build_latest_pit_feature_panel",
        lambda combined, date, settings: (reduced, {"passed": True}),
    )
    monkeypatch.setattr(
        "stockpilot.daily_pit.pipeline._metadata_for_current",
        lambda combined, date, current_symbols, settings: metadata,
    )
    frozen_hash = sha256_file(settings.frozen_market_path)
    result = materialize_features(TARGET, settings=settings)
    again = materialize_features(TARGET, settings=settings)
    verified = verify_daily_feature_partition(TARGET, settings=settings)
    panel = pd.read_parquet(settings.date_dir(TARGET) / "panel.parquet")
    assert list(panel.columns) == DAILY_FEATURE_COLUMNS
    assert len(panel.columns) == len(set(panel.columns)) == 71
    assert set(V10_FEATURES).issubset(panel.columns)
    assert np.isfinite(panel[V10_FEATURES].to_numpy(dtype=float)).all()
    assert result["prediction_backfill_2026_09_01"] is False
    assert again["idempotent"] is True
    assert verified["verified"] is True
    assert sha256_file(settings.frozen_market_path) == frozen_hash
    assert not (settings.root.parent / "_prediction_attempts").exists()


def test_materialization_conflict_fails_closed(configured) -> None:
    settings, _ = configured
    directory = settings.date_dir(TARGET)
    directory.mkdir(parents=True)
    (directory / "panel.parquet").write_bytes(b"conflict")
    with pytest.raises(DailyPitError, match="DAILY_FEATURE_MANIFEST_INVALID"):
        materialize_features(TARGET, settings=settings)


def test_schema_is_bound_to_frozen_features() -> None:
    assert DAILY_FEATURE_COLUMNS[-len(V10_FEATURES) :] == V10_FEATURES
    assert len(V10_FEATURES) == 61
    assert len(DAILY_FEATURE_COLUMNS) == len(set(DAILY_FEATURE_COLUMNS)) == 71
