from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier, Lock, Thread

import pandas as pd

from stockpilot.prospective_r1.ledger import (
    LedgerSettings,
    SourceCapture,
    observe_sources,
)


DATE = "2026-08-31"
NOW = datetime(2026, 8, 31, 8, tzinfo=timezone.utc)


def _settings(tmp_path: Path) -> LedgerSettings:
    lock = tmp_path / "lock.json"
    lock.write_text("{}", encoding="utf-8")
    return LedgerSettings(data_root=tmp_path / "observations", lock_path=lock)


def _run(tmp_path: Path, fetcher, now=NOW):
    return observe_sources(
        target_date=DATE, observed_at=now, trading_calendar={DATE}, universe={"000001"},
        source_fetchers={"earnings_expectations": ({}, fetcher)},
        membership_snapshot_hash="m", industry_mapping_hash="i", settings=_settings(tmp_path),
    )


def _capture(value=1.0):
    return SourceCapture(
        source="earnings_expectations", request_parameters={}, raw_payloads=(b"raw",),
        normalized=pd.DataFrame({"symbol": ["000001"], "forecast_eps_1": [value]}),
        required_value_columns=("forecast_eps_1",),
    )


def test_atomic_reservation_allows_only_one_concurrent_network_call(tmp_path: Path):
    barrier = Barrier(2)
    counter_lock = Lock()
    calls, outcomes = [], []

    def worker(offset):
        barrier.wait()
        try:
            def fetcher():
                with counter_lock:
                    calls.append(offset)
                return _capture()
            outcomes.append(("ok", _run(tmp_path, fetcher, NOW + timedelta(microseconds=offset))))
        except RuntimeError as error:
            outcomes.append(("rejected", str(error)))

    threads = [Thread(target=worker, args=(index,)) for index in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(calls) == 1
    assert sorted(item[0] for item in outcomes) == ["ok", "rejected"]


def test_failed_attempt_still_blocks_same_day_retry_before_network(tmp_path: Path):
    first = _run(tmp_path, lambda: (_ for _ in ()).throw(ConnectionError("offline")))
    calls = []
    try:
        _run(tmp_path, lambda: calls.append(1))
    except RuntimeError:
        pass
    else:
        raise AssertionError("same-day failed attempt must not be retried")
    assert first["status"] == "FAILED" and calls == []


def test_nonempty_all_nan_required_values_cannot_be_success(tmp_path: Path):
    record = _run(tmp_path, lambda: _capture(float("nan")))
    source = record["sources"]["earnings_expectations"]
    assert source["source_status"] == "REQUEST_FAILED"
    assert "no available required values" in source["failure_reason"]


def test_real_zero_required_value_is_success(tmp_path: Path):
    record = _run(tmp_path, lambda: _capture(0.0))
    assert record["sources"]["earnings_expectations"]["source_status"] == "SUCCESS"

