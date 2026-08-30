from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stockpilot.prospective.alpha import (
    factor_decay_metrics,
    factor_validation_metrics,
    grouped_stability,
    turnover_by_date,
)
from stockpilot.prospective.feature_store import (
    build_feature_panel,
    verify_feature_panel,
    write_feature_panel,
)
from stockpilot.prospective.immutable import sha256_bytes, write_new_bytes, verify_sidecar
from stockpilot.prospective.labels import settle_mature_labels
from stockpilot.prospective.ledger import (
    LedgerSettings,
    SourceCapture,
    load_observations,
    observe_sources,
    validate_observation_request,
)
from stockpilot.prospective.readiness import derive_readiness
from stockpilot.prospective.revision import build_industry_revision, build_revision_panel


NOW = datetime(2026, 8, 31, 8, tzinfo=timezone.utc)
DATE = "2026-08-31"


def _settings(tmp_path: Path) -> LedgerSettings:
    lock = tmp_path / "lock.json"
    lock.write_text("{}", encoding="utf-8")
    return LedgerSettings(data_root=tmp_path / "observations", lock_path=lock)


def _capture(rows: list[dict] | None = None, source: str = "earnings_expectations") -> SourceCapture:
    rows = rows or [{"symbol": "000001", "value": 1.0}]
    return SourceCapture(
        source=source,
        request_parameters={"page": 1},
        raw_payloads=(b'{"page":1}',),
        normalized=pd.DataFrame(rows),
    )


def _observe(tmp_path: Path, fetcher=None, source: str = "earnings_expectations") -> dict:
    settings = _settings(tmp_path)
    return observe_sources(
        target_date=DATE,
        observed_at=NOW,
        trading_calendar={DATE},
        universe={"000001", "000002"},
        source_fetchers={source: ({"page": 1}, fetcher or (lambda: _capture(source=source)))},
        membership_snapshot_hash="m" * 64,
        industry_mapping_hash="i" * 64,
        settings=settings,
    )


def _expectations(value: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["000001", "000002"],
        "forecast_year_1": [2027, 2027],
        "forecast_eps_1": [value, value * 2],
        "industry": ["tech", "tech"],
    })


def _market(days: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2026-08-03", periods=days)
    market = pd.DataFrame({
        "date": dates,
        "symbol": "000001",
        "open": np.arange(days, dtype=float) + 10,
        "is_suspended": False,
        "is_delisted": False,
    })
    benchmark = pd.DataFrame({"date": dates, "open": np.arange(days, dtype=float) + 100})
    return market, benchmark


def _price_files(tmp_path: Path, market: pd.DataFrame, benchmark: pd.DataFrame) -> tuple[Path, Path]:
    market_path, benchmark_path = tmp_path / "market.csv", tmp_path / "benchmark.csv"
    market.to_csv(market_path, index=False)
    benchmark.to_csv(benchmark_path, index=False)
    return market_path, benchmark_path


def test_same_day_second_observation_is_rejected(tmp_path: Path):
    _observe(tmp_path)
    with pytest.raises(RuntimeError, match="already exists"):
        _observe(tmp_path)


def test_same_day_rejection_happens_before_network(tmp_path: Path):
    _observe(tmp_path)
    calls = []
    with pytest.raises(RuntimeError):
        _observe(tmp_path, fetcher=lambda: calls.append(1))
    assert calls == []


def test_historical_backfill_rejected_before_network(tmp_path: Path):
    calls = []
    with pytest.raises(ValueError, match="backfill"):
        observe_sources(
            target_date="2026-08-28", observed_at=NOW, trading_calendar={"2026-08-28"},
            universe={"000001"}, source_fetchers={"x": ({}, lambda: calls.append(1))},
            membership_snapshot_hash="m", industry_mapping_hash="i", settings=_settings(tmp_path),
        )
    assert calls == []


def test_non_trading_date_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="trading date"):
        validate_observation_request(DATE, NOW, set(), _settings(tmp_path))


def test_source_failure_preserves_successful_source(tmp_path: Path):
    settings = _settings(tmp_path)
    record = observe_sources(
        target_date=DATE, observed_at=NOW, trading_calendar={DATE}, universe={"000001"},
        source_fetchers={
            "earnings_expectations": ({}, lambda: _capture()),
            "fund_flows": ({}, lambda: (_ for _ in ()).throw(ConnectionError("offline"))),
        },
        membership_snapshot_hash="m", industry_mapping_hash="i", settings=settings,
    )
    assert record["status"] == "PARTIAL"
    assert record["sources"]["earnings_expectations"]["source_status"] == "SUCCESS"
    assert record["sources"]["fund_flows"]["source_status"] == "REQUEST_FAILED"
    assert record["sources"]["fund_flows"]["silent_fallback_used"] is False


def test_empty_success_distinguished_from_request_failure(tmp_path: Path):
    empty = SourceCapture("announcements", {}, (b"[]",), pd.DataFrame(columns=["symbol"]))
    record = _observe(tmp_path, fetcher=lambda: empty, source="announcements")
    assert record["sources"]["announcements"]["source_status"] == "EMPTY_SUCCESS"


def test_raw_and_normalized_hashes_are_recorded(tmp_path: Path):
    record = _observe(tmp_path)
    source = record["sources"]["earnings_expectations"]
    assert source["raw_response_sha256"] == [sha256_bytes(b'{"page":1}')]
    assert len(source["normalized_data_sha256"]) == 64


def test_exact_duplicate_is_safely_deduplicated(tmp_path: Path):
    duplicate = SourceCapture(
        "earnings_expectations", {}, (b"x",),
        pd.DataFrame([{"symbol": "000001", "value": 1}, {"symbol": "000001", "value": 1}]),
        duplicate_count=1,
    )
    record = _observe(tmp_path, fetcher=lambda: duplicate)
    assert record["sources"]["earnings_expectations"]["row_count"] == 1


def test_conflicting_duplicate_hard_fails_source(tmp_path: Path):
    conflict = SourceCapture(
        "earnings_expectations", {}, (b"x",),
        pd.DataFrame([{"symbol": "000001", "value": 1}, {"symbol": "000001", "value": 2}]),
        duplicate_count=1, conflicting_duplicate_count=1,
    )
    record = _observe(tmp_path, fetcher=lambda: conflict)
    assert record["sources"]["earnings_expectations"]["source_status"] == "REQUEST_FAILED"


def test_source_identity_change_blocks_fallback(tmp_path: Path):
    record = _observe(tmp_path, fetcher=lambda: _capture(source="replacement_provider"))
    source = record["sources"]["earnings_expectations"]
    assert source["source_status"] == "REQUEST_FAILED"
    assert "fallback is forbidden" in source["failure_reason"]


def test_observation_ledger_is_append_only(tmp_path: Path):
    record = _observe(tmp_path)
    path = _settings(tmp_path).data_root / record["observation_id"] / "observation.json"
    with pytest.raises(FileExistsError):
        write_new_bytes(path, b"changed")


def test_first_snapshot_has_no_revision():
    result = build_revision_panel(None, _expectations(), previous_observed_at=None, current_observed_at="2026-08-31T08:00:00+00:00")
    assert not result["revision_available"].any()
    assert result["expectation_revision_abs"].isna().all()


def test_second_snapshot_produces_revision():
    result = build_revision_panel(
        _expectations(1.0), _expectations(1.1),
        previous_observed_at="2026-08-31T08:00:00+00:00",
        current_observed_at="2026-09-01T08:00:00+00:00",
    )
    assert result["revision_available"].all()
    assert result["expectation_revision_abs"].iloc[0] == pytest.approx(0.1)


def test_revision_requires_strictly_later_observation():
    with pytest.raises(ValueError, match="t1.observed_at"):
        build_revision_panel(
            _expectations(), _expectations(),
            previous_observed_at="2026-08-31T08:00:00+00:00",
            current_observed_at="2026-08-31T08:00:00+00:00",
        )


def test_revision_requires_same_forecast_year():
    current = _expectations(1.1)
    current.loc[0, "forecast_year_1"] = 2028
    result = build_revision_panel(
        _expectations(), current,
        previous_observed_at="2026-08-31T08:00:00+00:00",
        current_observed_at="2026-09-01T08:00:00+00:00",
    )
    assert pd.isna(result.loc[result["symbol"] == "000001", "expectation_revision_abs"]).all()


def test_industry_revision_uses_pit_mapping():
    panel = build_revision_panel(
        _expectations(), _expectations(1.1),
        previous_observed_at="2026-08-31T08:00:00+00:00",
        current_observed_at="2026-09-01T08:00:00+00:00",
    )
    industry = build_industry_revision(panel)
    assert industry.loc[0, "industry"] == "tech"
    assert industry.loc[0, "industry_revision_breadth"] == 2


def test_missing_is_distinct_from_real_zero():
    universe = pd.DataFrame({"symbol": ["000001", "000002"], "industry": ["x", "x"], "universe_member": True})
    expectation = pd.DataFrame({"symbol": ["000001"], "forecast_eps_1": [0.0]})
    panel = build_feature_panel(
        universe, expectation, date=DATE, observation_id="x", observation_hash="h",
        qualifying_trading_observation=True,
    ).set_index("symbol")
    assert panel.loc["000001", "expectation_level"] == 0
    assert pd.isna(panel.loc["000002", "expectation_level"])
    assert panel.loc["000001", "expectation_available"]
    assert not panel.loc["000002", "expectation_available"]


def test_feature_store_is_append_only_and_hashed(tmp_path: Path):
    universe = pd.DataFrame({"symbol": ["000001"], "industry": ["x"], "universe_member": True})
    panel = build_feature_panel(
        universe, None, date=DATE, observation_id="x", observation_hash="h",
        qualifying_trading_observation=True,
    )
    manifest = write_feature_panel(panel, tmp_path, source_provenance={"observation": "h"})
    assert verify_feature_panel(tmp_path / "manifests" / f"{DATE}.json")["intact"]
    with pytest.raises(FileExistsError):
        write_feature_panel(panel, tmp_path, source_provenance={"observation": "h"})
    assert manifest["feature_availability"]["expectation_available"] == 0


def test_immutable_hash_mutation_fails(tmp_path: Path):
    path = tmp_path / "value.bin"
    write_new_bytes(path, b"original")
    path.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_sidecar(path)


def test_unmatured_labels_are_not_written(tmp_path: Path):
    market, benchmark = _market(5)
    market_path, benchmark_path = _price_files(tmp_path, market, benchmark)
    prediction = pd.DataFrame({"date": [market["date"].iloc[-1]], "symbol": ["000001"]})
    records = settle_mature_labels(
        prediction, market, benchmark, price_source_path=market_path,
        benchmark_source_path=benchmark_path, ledger_root=tmp_path / "labels",
        as_of=market["date"].max(), corporate_action_handling="HFQ_PIT_GOVERNED",
    )
    assert records == []
    assert not (tmp_path / "labels").exists()


def test_1d_label_writes_only_after_maturity(tmp_path: Path):
    market, benchmark = _market(4)
    market_path, benchmark_path = _price_files(tmp_path, market, benchmark)
    prediction = pd.DataFrame({"date": [market["date"].iloc[0]], "symbol": ["000001"]})
    records = settle_mature_labels(
        prediction, market, benchmark, price_source_path=market_path,
        benchmark_source_path=benchmark_path, ledger_root=tmp_path / "labels",
        as_of=market["date"].max(), corporate_action_handling="HFQ_PIT_GOVERNED",
    )
    assert [item["horizon"] for item in records] == [1]
    assert records[0]["maturity_date"] == str(market["date"].iloc[2].date())


def test_5d_and_20d_labels_obey_calendar_maturity(tmp_path: Path):
    market, benchmark = _market(25)
    market_path, benchmark_path = _price_files(tmp_path, market, benchmark)
    prediction = pd.DataFrame({"date": [market["date"].iloc[0]], "symbol": ["000001"]})
    records = settle_mature_labels(
        prediction, market, benchmark, price_source_path=market_path,
        benchmark_source_path=benchmark_path, ledger_root=tmp_path / "labels",
        as_of=market["date"].max(), corporate_action_handling="HFQ_PIT_GOVERNED",
    )
    assert {item["horizon"] for item in records} == {1, 5, 20}
    assert next(item for item in records if item["horizon"] == 20)["maturity_date"] == str(market["date"].iloc[21].date())


def test_missing_exit_is_recorded_not_dropped(tmp_path: Path):
    market, benchmark = _market(4)
    market.loc[market.index[2], "open"] = np.nan
    market_path, benchmark_path = _price_files(tmp_path, market, benchmark)
    prediction = pd.DataFrame({"date": [market["date"].iloc[0]], "symbol": ["000001"]})
    records = settle_mature_labels(
        prediction, market, benchmark, price_source_path=market_path,
        benchmark_source_path=benchmark_path, ledger_root=tmp_path / "labels",
        as_of=market["date"].max(), corporate_action_handling="HFQ_PIT_GOVERNED",
    )
    assert records[0]["status"] == "MISSING_EXIT_PRICE"


def test_mature_label_is_immutable(tmp_path: Path):
    market, benchmark = _market(4)
    market_path, benchmark_path = _price_files(tmp_path, market, benchmark)
    prediction = pd.DataFrame({"date": [market["date"].iloc[0]], "symbol": ["000001"]})
    kwargs = dict(
        price_source_path=market_path, benchmark_source_path=benchmark_path,
        ledger_root=tmp_path / "labels", as_of=market["date"].max(),
        corporate_action_handling="HFQ_PIT_GOVERNED",
    )
    first = settle_mature_labels(prediction, market, benchmark, **kwargs)
    second = settle_mature_labels(prediction, market, benchmark, **kwargs)
    assert first[0]["forward_return"] == second[0]["forward_return"]
    changed = market.copy()
    changed.loc[changed.index[2], "open"] += 1
    with pytest.raises(RuntimeError, match="immutable"):
        settle_mature_labels(prediction, changed, benchmark, **kwargs)


def test_model_readiness_cannot_be_early():
    observations = [{
        "observation_id": "one", "target_date": DATE, "qualifying_trading_observation": True,
        "sources": {"earnings_expectations": {"source_status": "SUCCESS"}},
    }]
    status = derive_readiness(observations, [])
    assert status.pit_observation_count == 1
    assert not status.model_training_ready
    assert not status.factor_validation_ready


def test_weekend_baseline_not_counted_as_trading_observation():
    observations = [{
        "observation_id": "baseline", "target_date": "2026-08-30",
        "qualifying_trading_observation": False,
        "sources": {"earnings_expectations": {"source_status": "SUCCESS"}},
    }]
    status = derive_readiness(observations, [])
    assert status.source_observation_count == 1 and status.pit_observation_count == 0


def test_production_and_execution_cannot_be_promoted_by_counts():
    observations = [{
        "observation_id": str(index), "target_date": f"d{index}", "qualifying_trading_observation": True,
        "sources": {"earnings_expectations": {"source_status": "SUCCESS"}},
    } for index in range(20)]
    labels = [
        {"prediction_date": f"d{index}", "horizon": horizon, "status": "SETTLED"}
        for index in range(20) for horizon in (1, 5, 20)
    ]
    status = derive_readiness(observations, labels)
    assert status.factor_validation_ready and status.model_training_ready
    assert not status.replacement_evaluation_ready
    assert not status.production_prediction_ready
    assert not status.execution_authorized


def test_factor_metrics_and_decay_are_algorithm_only():
    dates = pd.bdate_range("2026-01-01", periods=4)
    rows = []
    for date in dates:
        for symbol in range(12):
            rows.append({
                "date": date, "symbol": f"{symbol:06d}", "factor": symbol,
                "forward_return_1d": symbol / 100, "forward_return_5d": symbol / 90,
                "forward_return_20d": symbol / 80, "industry": "a" if symbol < 6 else "b",
            })
    frame = pd.DataFrame(rows)
    metrics = factor_validation_metrics(frame, "factor", "forward_return_5d")
    decay = factor_decay_metrics(frame, "factor")
    assert metrics["spearman_rank_ic"] == pytest.approx(1.0)
    assert set(decay["status"]) == {"AVAILABLE"}
    assert turnover_by_date(frame, "factor") == pytest.approx(0.0)


def test_grouped_stability_requires_real_group_column():
    frame = pd.DataFrame({"date": [], "symbol": [], "factor": [], "target": []})
    with pytest.raises(ValueError, match="group missing"):
        grouped_stability(frame, "factor", "target", "regime")


def test_load_observations_only_reads_completed_receipts(tmp_path: Path):
    _observe(tmp_path)
    assert len(load_observations(_settings(tmp_path))) == 1

