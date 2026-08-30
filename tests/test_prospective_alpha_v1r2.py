from __future__ import annotations

import json
import multiprocessing as mp
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pit_data_v2 import core as legacy_core
from stockpilot.prediction.storage import write_immutable_prediction_snapshot
from stockpilot.prospective_r2.alpha import (
    PurgedWalkForwardSplit,
    benjamini_hochberg,
    preregistration_manifest,
)
from stockpilot.prospective_r2.config import OperationalSettings, ReadinessThresholds
from stockpilot.prospective_r2.feature_store import (
    build_feature_panel,
    verify_feature_panel,
    write_feature_panel,
)
from stockpilot.prospective_r2.freeze import verify_parent_locks
from stockpilot.prospective_r2.integrity import (
    IncompleteArtifactError,
    read_verified_json,
    sha256_file,
    verify_immutable,
    write_immutable_bytes,
)
from stockpilot.prospective_r2.labels import (
    settle_mature_labels,
    verify_corporate_action_provenance,
)
from stockpilot.prospective_r2.observation import (
    SourceCapture,
    capture_sources_once,
    reserve_daily_attempt,
)
from stockpilot.prospective_r2.orchestrator import DailyDependencies, run_daily
from stockpilot.prospective_r2.readiness import derive_readiness, observation_qualifies
from stockpilot.prospective_r2.revision import (
    SnapshotProof,
    build_industry_revision,
    build_revision_panel,
)


DATE = "2026-08-31"
NOW = datetime(2026, 8, 31, 8, tzinfo=timezone.utc)


def _settings(tmp_path: Path, *, thresholds: ReadinessThresholds | None = None) -> OperationalSettings:
    artifacts = tmp_path / "artifacts"
    calendar = artifacts / "calendar.json"
    calendar.parent.mkdir(parents=True, exist_ok=True)
    calendar.write_text(json.dumps({
        "market": "XSHG", "coverage_start": "2026-01-01", "coverage_end": "2026-12-31",
        "weekends_closed": True, "closed_weekdays": ["2026-10-01"],
        "source": "test verified calendar", "source_url": "https://example.invalid/calendar",
    }), encoding="utf-8")
    return OperationalSettings(
        data_root=tmp_path / "runtime",
        artifact_root=artifacts,
        calendar_path=calendar,
        membership_path=tmp_path / "membership.csv",
        industry_path=tmp_path / "industry.csv",
        corporate_action_path=tmp_path / "actions.json",
        corporate_action_manifest_path=tmp_path / "actions-manifest.json",
        plan_lock_path=artifacts / "plan.lock.json",
        legacy_barrier_path=tmp_path / "barrier" / "manifest.json",
        thresholds=thresholds or ReadinessThresholds(
            minimum_observation_dates=20,
            minimum_expectation_coverage=.8,
            minimum_label_dates=20,
            minimum_label_coverage=.8,
            minimum_label_symbols=2,
        ),
    )


def _context(target_date: str, settings: OperationalSettings) -> tuple[pd.DataFrame, dict]:
    del target_date, settings
    return pd.DataFrame({
        "symbol": ["000001", "000002"], "industry": ["bank", "tech"],
        "universe_member": [True, True],
    }), {"membership_snapshot_sha256": "m" * 64, "industry_mapping_sha256": "i" * 64}


def _capture(source: str, rows: list[dict], *, confirmed=(), required=()) -> SourceCapture:
    return SourceCapture(
        source=source, request_parameters={"source": source}, raw_payloads=(b"raw",),
        normalized=pd.DataFrame(rows), confirmed_symbols=tuple(confirmed),
        required_value_columns=tuple(required), network_request_count=1,
    )


def _successful_factory(universe, target_date, observed_at, settings):
    del target_date, observed_at, settings
    symbols = sorted(universe)
    return {
        "earnings_expectations": ({}, lambda: _capture(
            "earnings_expectations",
            [{"symbol": symbol, "forecast_year_1": 2027, "forecast_eps_1": index + 1,
              "industry": "bank" if index == 0 else "tech"}
             for index, symbol in enumerate(symbols)],
            required=("forecast_eps_1",),
        )),
        "announcements": ({}, lambda: _capture(
            "announcements",
            [{"symbol": symbol, "announcement_event_count": 0, "announcement_available": True}
             for symbol in symbols], confirmed=symbols,
            required=("announcement_event_count",),
        )),
        "fund_flows": ({}, lambda: _capture(
            "fund_flows", [{"symbol": symbol, "main_net_inflow": 0,
                             "main_net_inflow_ratio": 0} for symbol in symbols],
            required=("main_net_inflow",),
        )),
    }


def _lock_verifier(settings):
    del settings
    return {"v1r2_lock_sha256": "l" * 64, "all_parent_locks_intact": True}


def _prediction(target_date, settings):
    del target_date, settings
    return {"status": "RECORDED", "execution_authorized": False}


def _settlement(target_date, settings):
    del target_date, settings
    return {"status": "NO_MATURE_LABELS", "mature_records_written": 0}


def _deps(factory=_successful_factory, settlement=_settlement):
    return DailyDependencies(
        lock_verifier=_lock_verifier, context_loader=_context,
        source_fetcher_factory=factory, prediction_runner=_prediction,
        settlement_runner=settlement,
    )


def _process_factory(universe, target_date, observed_at, settings):
    del target_date, observed_at
    counter = settings.data_root.parent / "provider_calls.bin"
    with counter.open("ab") as stream:
        stream.write(b"1")
        stream.flush()
        os.fsync(stream.fileno())
    symbol = sorted(universe)[0]
    return {"earnings_expectations": ({}, lambda: _capture(
        "earnings_expectations", [{"symbol": symbol, "forecast_year_1": 2027,
                                    "forecast_eps_1": 1.0, "industry": "bank"}],
        required=("forecast_eps_1",),
    ))}


def _process_worker(settings: OperationalSettings, queue) -> None:
    try:
        run_daily(now=NOW, settings=settings, dependencies=_deps(_process_factory))
        queue.put("ok")
    except RuntimeError:
        queue.put("reserved")


def _direct_observation(tmp_path: Path, covered: int = 2, universe_size: int = 2) -> dict:
    settings = _settings(tmp_path)
    universe = {f"{index:06d}" for index in range(1, universe_size + 1)}
    attempt = reserve_daily_attempt(DATE, NOW, parent_lock_sha256="l" * 64, settings=settings)
    rows = [{"symbol": symbol, "forecast_eps_1": 1.0} for symbol in sorted(universe)[:covered]]
    return capture_sources_once(
        attempt=attempt, target_date=DATE, observed_at=NOW, universe=universe,
        source_fetchers={"earnings_expectations": ({}, lambda: _capture(
            "earnings_expectations", rows, required=("forecast_eps_1",)
        ))}, membership_snapshot_hash="m" * 64, industry_mapping_hash="i" * 64,
        trading_calendar_hash="c" * 64, settings=settings,
    )


def _proof(name: str, day: int) -> SnapshotProof:
    return SnapshotProof(name, f"2026-08-{day:02d}T08:00:00+00:00", "s" * 64, "r" * 64)


def _expectations(value: float) -> pd.DataFrame:
    return pd.DataFrame({"symbol": ["000001", "000002"], "forecast_year_1": [2027, 2027],
                         "forecast_eps_1": [value, value * 2], "industry": ["bank", "tech"]})


def _price_sources(tmp_path: Path):
    dates = pd.bdate_range("2026-08-03", periods=25)
    market = pd.DataFrame({"date": dates, "symbol": "000001",
                           "open": np.arange(25, dtype=float) + 10,
                           "is_suspended": False, "is_delisted": False})
    benchmark = pd.DataFrame({"date": dates, "open": np.arange(25, dtype=float) + 100})
    market_path, benchmark_path = tmp_path / "market.csv", tmp_path / "benchmark.csv"
    market.to_csv(market_path, index=False)
    benchmark.to_csv(benchmark_path, index=False)
    actions = tmp_path / "actions.json"
    actions.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "actions-manifest.json"
    manifest.write_text(json.dumps({"sha256": {actions.as_posix(): sha256_file(actions)}}), encoding="utf-8")
    return market, benchmark, market_path, benchmark_path, actions, manifest


def test_legacy_v2_entry_cannot_bypass_global_reservation(monkeypatch):
    calls = []
    monkeypatch.setattr("requests.sessions.Session.get", lambda *a, **k: calls.append(1))
    with pytest.raises((KeyError, RuntimeError)):
        legacy_core.observe(target_date=DATE, now=NOW)
    assert calls == []


def test_orchestrator_reserves_before_provider(tmp_path):
    settings = _settings(tmp_path)
    seen = []
    def factory(universe, target_date, observed_at, configured):
        assert (configured.attempts_root / f"{DATE}.json").exists()
        seen.append(True)
        return _successful_factory(universe, target_date, observed_at, configured)
    run_daily(now=NOW, settings=settings, dependencies=_deps(factory))
    assert seen == [True]


def test_two_processes_make_at_most_one_provider_call(tmp_path):
    settings = _settings(tmp_path)
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=_process_worker, args=(settings, queue)) for _ in range(2)]
    for process in processes: process.start()
    for process in processes: process.join(20)
    assert all(process.exitcode == 0 for process in processes)
    assert sorted(queue.get(timeout=2) for _ in processes) == ["ok", "reserved"]
    assert (tmp_path / "provider_calls.bin").read_bytes() == b"1"


def test_failed_attempt_blocks_same_day_retry(tmp_path):
    settings = _settings(tmp_path)
    def failed(*args):
        universe = args[0]
        return {"earnings_expectations": ({}, lambda: (_ for _ in ()).throw(ConnectionError("offline")))}
    first = run_daily(now=NOW, settings=settings, dependencies=_deps(failed))
    assert first["status"] == "FAILED"
    with pytest.raises(RuntimeError, match="already reserved"):
        run_daily(now=NOW, settings=settings, dependencies=_deps())


@pytest.mark.parametrize("date, now, message", [
    ("2026-08-30", datetime(2026, 8, 30, 8, tzinfo=timezone.utc), "trading"),
    ("2026-10-01", datetime(2026, 10, 1, 8, tzinfo=timezone.utc), "trading"),
    ("2026-08-28", NOW, "backfill"),
    ("2026-09-01", NOW, "future"),
])
def test_invalid_dates_rejected_before_network(tmp_path, date, now, message):
    calls = []
    def factory(*args): calls.append(1); return {}
    with pytest.raises(ValueError, match=message):
        run_daily(target_date=date, now=now, settings=_settings(tmp_path), dependencies=_deps(factory))
    assert calls == []


def test_low_coverage_observation_does_not_qualify(tmp_path):
    record = _direct_observation(tmp_path, covered=1, universe_size=2)
    assert not observation_qualifies(record, ReadinessThresholds(minimum_expectation_coverage=.8))


def test_success_without_verified_hashes_does_not_qualify(tmp_path):
    record = _direct_observation(tmp_path)
    record["sources"]["earnings_expectations"]["hashes_verified"] = False
    assert not observation_qualifies(record, ReadinessThresholds())


def test_twenty_low_coverage_observations_do_not_unlock(tmp_path):
    record = _direct_observation(tmp_path, covered=1, universe_size=2)
    observations = [record | {"observation_id": str(i), "target_date": f"d{i}"} for i in range(20)]
    status = derive_readiness(observations, [], thresholds=ReadinessThresholds(
        minimum_observation_dates=20, minimum_expectation_coverage=.8,
        minimum_label_dates=20, minimum_label_coverage=.8, minimum_label_symbols=2))
    assert not status.observation_quality_ready and not status.factor_validation_ready


def _labels(symbols_per_day: int) -> list[dict]:
    return [{"prediction_date": f"d{day}", "symbol": f"{symbol:06d}", "horizon": horizon,
             "status": "SETTLED", "expected_universe_size": 300,
             "label_fully_verified": True, "price_provenance_verified": True,
             "benchmark_provenance_verified": True, "corporate_action_verified": True}
            for day in range(20) for horizon in (1, 5, 20)
            for symbol in range(symbols_per_day)]


def test_one_symbol_per_day_does_not_qualify_mature_labels():
    status = derive_readiness([], _labels(1))
    assert (status.mature_1d_count, status.mature_5d_count, status.mature_20d_count) == (0, 0, 0)


def test_qualified_mature_dates_require_count_and_coverage():
    thresholds = ReadinessThresholds(minimum_label_symbols=240, minimum_label_coverage=.8)
    assert derive_readiness([], _labels(239), thresholds=thresholds).mature_1d_count == 0
    assert derive_readiness([], _labels(240), thresholds=thresholds).mature_1d_count == 20


def test_announcement_partial_coverage_preserves_nan():
    universe = pd.DataFrame({"symbol": ["000001", "000002"], "industry": ["a", "b"], "universe_member": True})
    announcements = pd.DataFrame({"symbol": ["000001"], "announcement_event_count": [1],
                                  "announcement_available": [True]})
    panel = build_feature_panel(universe, date=DATE, observation_id="o", observation_hash="h",
                                announcements=announcements).set_index("symbol")
    assert pd.isna(panel.loc["000002", "announcement_event_count"])
    assert not panel.loc["000002", "announcement_available"]


def test_confirmed_zero_announcement_is_real_zero():
    universe = pd.DataFrame({"symbol": ["000001"], "industry": ["a"], "universe_member": True})
    announcements = pd.DataFrame({"symbol": ["000001"], "announcement_event_count": [0],
                                  "announcement_available": [True]})
    panel = build_feature_panel(universe, date=DATE, observation_id="o", observation_hash="h",
                                announcements=announcements)
    assert panel.loc[0, "announcement_event_count"] == 0 and panel.loc[0, "announcement_available"]


def test_unavailable_fund_flow_remains_nan():
    universe = pd.DataFrame({"symbol": ["000001"], "industry": ["a"], "universe_member": True})
    panel = build_feature_panel(universe, date=DATE, observation_id="o", observation_hash="h")
    assert pd.isna(panel.loc[0, "main_net_inflow"]) and not panel.loc[0, "fund_flow_available"]


def test_second_order_revision_rejects_future_earlier_revision():
    with pytest.raises(ValueError, match="T-2"):
        build_revision_panel(_expectations(1), _expectations(2), previous_proof=_proof("p", 29),
                             current_proof=_proof("c", 31), earlier_revision=pd.DataFrame({
                                 "symbol": ["000001", "000002"], "revision_direction": [1, 1]}),
                             earlier_proof=_proof("e", 30))


def test_industry_acceleration_rejects_future_previous_industry():
    panel = build_revision_panel(_expectations(1), _expectations(2),
                                 previous_proof=_proof("p", 29), current_proof=_proof("c", 31))
    previous = pd.DataFrame({"industry": ["bank", "tech"], "industry_revision_mean": [0.1, .2]})
    with pytest.raises(ValueError, match="strict"):
        build_industry_revision(panel, current_proof=_proof("c", 31),
                                previous_industry=previous, previous_proof=_proof("future", 31))


@pytest.mark.parametrize("kind", ["market", "benchmark"])
def test_dataframe_provenance_mismatch_hard_fails(tmp_path, kind):
    market, benchmark, market_path, benchmark_path, actions, manifest = _price_sources(tmp_path)
    supplied_market, supplied_benchmark = market.copy(), benchmark.copy()
    if kind == "market": supplied_market.loc[0, "open"] += 1
    else: supplied_benchmark.loc[0, "open"] += 1
    with pytest.raises(RuntimeError, match=f"{kind} DataFrame provenance mismatch"):
        settle_mature_labels(pd.DataFrame({"date": [market.date.iloc[0]], "symbol": ["000001"]}),
            market_source_path=market_path, benchmark_source_path=benchmark_path,
            corporate_action_dataset_path=actions, corporate_action_manifest_path=manifest,
            ledger_root=tmp_path / "labels", as_of=market.date.max(),
            expected_universe_by_date={str(market.date.iloc[0].date()): {"000001"}},
            market=supplied_market, benchmark=supplied_benchmark)


def test_missing_corporate_action_provenance_fails(tmp_path):
    actions = tmp_path / "actions.json"; actions.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.json"; manifest.write_text('{"sha256":{}}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="not bound"):
        verify_corporate_action_provenance(actions, manifest)


def test_manifest_mutation_is_detected(tmp_path):
    universe = pd.DataFrame({"symbol": ["000001"], "industry": ["a"], "universe_member": True})
    panel = build_feature_panel(universe, date=DATE, observation_id="o", observation_hash="h")
    result = write_feature_panel(panel, tmp_path, source_provenance={})
    Path(result["manifest_path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_feature_panel(result["manifest_path"])


def test_payload_without_sidecar_is_not_intact(tmp_path):
    path = tmp_path / "payload.bin"; path.write_bytes(b"payload")
    with pytest.raises(IncompleteArtifactError): verify_immutable(path)


def test_keyboard_interrupt_is_not_request_failed(tmp_path):
    settings = _settings(tmp_path)
    def factory(*args):
        return {"earnings_expectations": ({}, lambda: (_ for _ in ()).throw(KeyboardInterrupt()))}
    with pytest.raises(KeyboardInterrupt):
        run_daily(now=NOW, settings=settings, dependencies=_deps(factory))
    receipt = read_verified_json(settings.daily_receipts_root / f"{DATE}.json")
    assert receipt["status"] == "INTERRUPTED" and "REQUEST_FAILED" not in json.dumps(receipt)


def test_v30_same_date_append_cannot_overwrite(tmp_path):
    path = tmp_path / "prediction.csv"
    frame = pd.DataFrame({"date": [DATE], "symbol": ["000001"], "rank_5d": [1], "p_up_5d": [.6]})
    assert write_immutable_prediction_snapshot(frame, path)[0]
    assert not write_immutable_prediction_snapshot(frame, path)[0]
    changed = frame.copy(); changed.loc[0, "p_up_5d"] = .7
    with pytest.raises(RuntimeError, match="hash mismatch"):
        write_immutable_prediction_snapshot(changed, path)


def test_mature_labels_settle_even_if_today_source_failed(tmp_path):
    called = []
    def factory(*args):
        return {"earnings_expectations": ({}, lambda: (_ for _ in ()).throw(ConnectionError("offline")))}
    def settlement(*args): called.append(True); return {"status": "SETTLED", "mature_records_written": 1}
    result = run_daily(now=NOW, settings=_settings(tmp_path), dependencies=_deps(factory, settlement))
    assert called == [True] and result["label_settlement"]["mature_records_written"] == 1


def test_readiness_never_promotes_production():
    status = derive_readiness([], [])
    assert not status.model_training_ready
    assert not status.replacement_evaluation_ready
    assert not status.production_prediction_ready


def test_readiness_never_authorizes_execution():
    status = derive_readiness([], [])
    assert not status.execution_authorized


def test_no_v31_training_entrypoint_exists():
    root = Path(__file__).resolve().parents[1]
    assert not list((root / "stockpilot").glob("**/*v31*"))
    cli = (root / "stockpilot/prospective_r2/cli.py").read_text(encoding="utf-8")
    assert 'add_parser("train")' not in cli


def test_purged_walk_forward_and_fdr_are_algorithm_only():
    dates = pd.Series(pd.bdate_range("2026-01-01", periods=20))
    label_end = dates + pd.offsets.BDay(2)
    folds = list(PurgedWalkForwardSplit(min_train_dates=5, validation_dates=3, gap_dates=3).split(dates, label_end))
    assert folds and max(folds[0].train_indices) < min(folds[0].validation_indices)
    fdr = benjamini_hochberg([.001, .2, .9])
    assert (fdr["family_size"] == 3).all()
    manifest = preregistration_manifest([{"name": "x", "formula": "f", "family": "expectation",
                                          "selection_criterion": "train-only FDR"}])
    assert manifest["real_prospective_results_read"] is False


def test_parent_lock_artifacts_remain_intact():
    result = verify_parent_locks()
    assert result["all_parent_locks_intact"]
    assert result["v18_lock_sidecar_verified"]
