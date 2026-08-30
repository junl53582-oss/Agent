from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stockpilot.prospective_r2.integrity import (
    canonical_frame_bytes,
    read_verified_json,
    sha256_bytes,
    sha256_file,
    write_immutable_json,
)
from stockpilot.prospective_r2.observation import (
    SourceCapture,
    capture_sources_once,
    reserve_daily_attempt,
    universe_hash,
)
from stockpilot.prospective_r3.certification import certify_observation
from stockpilot.prospective_r3.config import OperationalSettings, ReadinessThresholds
from stockpilot.prospective_r3.orchestrator import DailyDependencies, run_daily
from stockpilot.prospective_r3.settlement import (
    certify_label_record,
    load_approved_settlement_bundle,
    settle_certified_labels,
    verify_corporate_action_trust_root,
    verify_mapping_lock,
)
from stockpilot.prospective_r3.status import aggregate_daily_status, build_runtime_status


DATE = "2026-08-31"
NOW = datetime(2026, 8, 31, 8, tzinfo=timezone.utc)


def _settings(tmp_path: Path, *, threshold_symbols: int = 2) -> OperationalSettings:
    artifacts = tmp_path / "artifacts"
    calendar = artifacts / "calendar.json"
    calendar.parent.mkdir(parents=True, exist_ok=True)
    calendar.write_text(json.dumps({
        "market": "XSHG", "coverage_start": "2026-01-01", "coverage_end": "2026-12-31",
        "weekends_closed": True, "closed_weekdays": ["2026-10-01"],
        "source": "fixture", "source_url": "https://example.invalid/xshg",
    }), encoding="utf-8")
    plan = artifacts / "plan.lock.json"
    plan.write_text('{"fixture":true}', encoding="utf-8")
    return OperationalSettings(
        data_root=tmp_path / "runtime",
        artifact_root=artifacts,
        calendar_path=calendar,
        membership_path=tmp_path / "membership.csv",
        industry_path=tmp_path / "industry.csv",
        corporate_action_path=tmp_path / "actions.json",
        corporate_action_lock_path=tmp_path / "v20r2" / "plan.lock.json",
        corporate_action_data_audit_path=tmp_path / "data_audit.json",
        settlement_manifest_path=artifacts / "settlement.json",
        plan_lock_path=plan,
        r2_barrier_path=tmp_path / "r2-barrier" / "observation.json",
        prediction_root=tmp_path / "predictions",
        thresholds=ReadinessThresholds(
            minimum_observation_dates=1,
            minimum_expectation_coverage=.8,
            minimum_label_dates=1,
            minimum_label_coverage=.8,
            minimum_label_symbols=threshold_symbols,
        ),
    )


def _context_files(settings: OperationalSettings, symbols=("000001", "000002")) -> tuple[pd.DataFrame, dict]:
    settings.membership_path.parent.mkdir(parents=True, exist_ok=True)
    membership = pd.DataFrame({"snapshot_date": [DATE] * len(symbols), "symbol": symbols})
    membership.to_csv(settings.membership_path, index=False)
    industry = pd.DataFrame({
        "symbol": symbols, "industry": ["bank", "tech"][:len(symbols)],
        "industry_effective_date": ["2026-01-01"] * len(symbols),
    })
    industry.to_csv(settings.industry_path, index=False)
    panel = pd.DataFrame({
        "symbol": symbols, "industry": industry["industry"], "universe_member": True,
    })
    membership_hash = sha256_bytes(canonical_frame_bytes(
        membership, ["snapshot_date", "symbol"]
    ))
    canonical_industry = panel[["symbol", "industry"]].copy()
    canonical_industry["industry_effective_date"] = pd.Timestamp("2026-01-01")
    industry_hash = sha256_bytes(canonical_frame_bytes(
        canonical_industry[["symbol", "industry", "industry_effective_date"]], ["symbol"]
    ))
    return panel, {
        "membership_snapshot_sha256": membership_hash,
        "membership_source_path": settings.membership_path.as_posix(),
        "membership_source_sha256": sha256_file(settings.membership_path),
        "industry_mapping_sha256": industry_hash,
        "industry_source_path": settings.industry_path.as_posix(),
        "industry_source_sha256": sha256_file(settings.industry_path),
        "universe_size": len(panel),
    }


def _context(target_date: str, settings: OperationalSettings):
    assert target_date == DATE
    return _context_files(settings)


def _lock(settings: OperationalSettings) -> dict:
    return {
        "v1r3_lock_sha256": sha256_file(settings.plan_lock_path),
        "frozen_inputs_intact": True,
    }


def _capture(source: str, rows: list[dict], *, required=(), confirmed=()) -> SourceCapture:
    return SourceCapture(
        source=source,
        request_parameters={"source": source},
        raw_payloads=(b"provider-raw",),
        normalized=pd.DataFrame(rows),
        normalized_keys=("symbol",),
        required_value_columns=tuple(required),
        confirmed_symbols=tuple(confirmed),
        network_request_count=1,
    )


def _factory(universe, target_date, observed_at, settings):
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
            [{"symbol": symbol, "announcement_event_count": 0,
              "announcement_available": True} for symbol in symbols],
            required=("announcement_event_count",), confirmed=symbols,
        )),
        "fund_flows": ({}, lambda: _capture(
            "fund_flows",
            [{"symbol": symbol, "main_net_inflow": 0.0,
              "main_net_inflow_ratio": 0.0} for symbol in symbols],
            required=("main_net_inflow",),
        )),
    }


def _observation(tmp_path: Path, *, covered_symbols=("000001", "000002")):
    settings = _settings(tmp_path)
    panel, proof = _context_files(settings)
    universe = set(panel["symbol"])
    attempt = reserve_daily_attempt(
        DATE, NOW, parent_lock_sha256=sha256_file(settings.plan_lock_path), settings=settings
    )
    rows = [
        {"symbol": symbol, "forecast_year_1": 2027, "forecast_eps_1": 1.0,
         "industry": "bank"}
        for symbol in covered_symbols
    ]
    record = capture_sources_once(
        attempt=attempt,
        target_date=DATE,
        observed_at=NOW,
        universe=universe,
        source_fetchers={"earnings_expectations": ({}, lambda: _capture(
            "earnings_expectations", rows, required=("forecast_eps_1",)
        ))},
        membership_snapshot_hash=proof["membership_snapshot_sha256"],
        industry_mapping_hash=proof["industry_mapping_sha256"],
        trading_calendar_hash=sha256_file(settings.calendar_path),
        settings=settings,
    )
    return settings, record


def _cert(record: dict, settings: OperationalSettings):
    return certify_observation(
        record, settings, lock_verifier=_lock, context_loader=_context, certified_at=NOW
    ).to_dict()


def test_fake_calendar_verified_boolean_cannot_qualify(tmp_path):
    settings, record = _observation(tmp_path)
    record["verified_shanghai_trading_date"] = True
    calendar = json.loads(settings.calendar_path.read_text())
    calendar["closed_weekdays"].append(DATE)
    settings.calendar_path.write_text(json.dumps(calendar), encoding="utf-8")
    result = _cert(record, settings)
    assert not result["calendar_verified"] and not result["qualifying_observation"]


def test_fake_64_character_calendar_hash_cannot_qualify(tmp_path):
    settings, record = _observation(tmp_path)
    record["trading_calendar_hash"] = "a" * 64
    assert not _cert(record, settings)["qualifying_observation"]


@pytest.mark.parametrize("field", ["pit_membership_snapshot_hash", "pit_industry_mapping_hash"])
def test_fake_context_hash_cannot_qualify(tmp_path, field):
    settings, record = _observation(tmp_path)
    record[field] = "f" * 64
    assert not _cert(record, settings)["qualifying_observation"]


def test_missing_reservation_cannot_qualify(tmp_path):
    settings, record = _observation(tmp_path)
    attempt = Path(record["attempt_path"])
    attempt.chmod(0o666)
    attempt.unlink()
    assert not _cert(record, settings)["reservation_verified"]


@pytest.mark.parametrize("field,value", [
    ("observation_attempt_id", "wrong"),
    ("parent_lock_sha256", "0" * 64),
])
def test_wrong_reservation_linkage_cannot_qualify(tmp_path, field, value):
    settings, record = _observation(tmp_path)
    path = Path(record["attempt_path"])
    attempt = json.loads(path.read_text())
    attempt[field] = value
    path.chmod(0o666)
    path.write_text(json.dumps(attempt), encoding="utf-8")
    record["attempt_sha256"] = sha256_file(path)
    assert not _cert(record, settings)["reservation_verified"]


def test_mutated_normalized_file_defeats_self_declared_hash_flag(tmp_path):
    settings, record = _observation(tmp_path)
    source = record["sources"]["earnings_expectations"]
    Path(source["normalized_path"]).write_text("symbol,forecast_eps_1\n000001,999\n")
    assert source["hashes_verified"] is True
    result = _cert(record, settings)
    assert not result["normalized_evidence_verified"] and not result["qualifying_observation"]


def test_recomputed_low_coverage_overrides_receipt_claim(tmp_path):
    settings, record = _observation(tmp_path, covered_symbols=("000001",))
    assert record["sources"]["earnings_expectations"]["hashes_verified"] is True
    result = _cert(record, settings)
    assert result["actual_universe_coverage"] == .5
    assert not result["coverage_verified"] and not result["qualifying_observation"]


def _fake_cert(observation, settings):
    del settings
    return {"target_date": observation["target_date"], "qualifying_observation": observation.get("qualifies", True)}


def _fake_label(record, settings):
    del settings
    return {"label_evidence_verified": record.get("verified", False)}


def _label_rows(dates: int, symbols: int, verified: bool) -> list[dict]:
    return [
        {"prediction_date": f"2026-07-{day + 1:02d}", "symbol": f"{symbol:06d}",
         "horizon": horizon, "expected_universe_size": 300, "verified": verified}
        for day in range(dates) for symbol in range(symbols) for horizon in (1, 5, 20)
    ]


def test_twenty_unverified_mature_dates_do_not_unlock(tmp_path):
    settings = replace(_settings(tmp_path), thresholds=ReadinessThresholds())
    result = build_runtime_status(
        settings, [], _label_rows(20, 240, False),
        observation_certifier=_fake_cert, label_certifier=_fake_label,
    )
    assert not result.label_quality_ready


def test_240_of_300_verified_labels_qualifies_one_date(tmp_path):
    settings = _settings(tmp_path, threshold_symbols=240)
    result = build_runtime_status(
        settings, [], _label_rows(1, 240, True),
        observation_certifier=_fake_cert, label_certifier=_fake_label,
    )
    assert (result.mature_1d_count, result.mature_5d_count, result.mature_20d_count) == (1, 1, 1)


def test_observation_count_semantics(tmp_path):
    settings = _settings(tmp_path)
    baseline = build_runtime_status(settings, [], [], observation_certifier=_fake_cert, label_certifier=_fake_label)
    assert (baseline.inherited_source_baseline_count, baseline.runtime_source_observation_count,
            baseline.qualified_pit_observation_count) == (1, 0, 0)
    unqualified = build_runtime_status(settings, [{"observation_id": "a", "target_date": DATE,
                                                    "qualifies": False}], [],
                                       observation_certifier=_fake_cert, label_certifier=_fake_label)
    assert (unqualified.inherited_source_baseline_count, unqualified.runtime_source_observation_count,
            unqualified.qualified_pit_observation_count) == (1, 1, 0)
    qualified = build_runtime_status(settings, [{"observation_id": "a", "target_date": DATE,
                                                  "qualifies": True}], [],
                                     observation_certifier=_fake_cert, label_certifier=_fake_label)
    assert (qualified.inherited_source_baseline_count, qualified.runtime_source_observation_count,
            qualified.qualified_pit_observation_count) == (1, 1, 1)


def _trust_fixture(tmp_path: Path, *, benchmark_approved=True):
    settings = _settings(tmp_path, threshold_symbols=1)
    dates = pd.bdate_range("2026-08-03", periods=25)
    market = pd.DataFrame({"date": dates, "symbol": "000001", "open": np.arange(25.) + 10,
                           "is_suspended": False, "is_delisted": False})
    benchmark = pd.DataFrame({"date": dates, "open": np.arange(25.) + 100})
    market_path, benchmark_path = tmp_path / "market.csv", tmp_path / "benchmark.csv"
    market.to_csv(market_path, index=False); benchmark.to_csv(benchmark_path, index=False)
    actions = settings.corporate_action_path
    actions.write_text("{}", encoding="utf-8")
    audit = settings.corporate_action_data_audit_path
    audit.write_text(json.dumps({"input_sha256": {
        market_path.as_posix(): sha256_file(market_path),
        actions.as_posix(): sha256_file(actions),
    }}), encoding="utf-8")
    lock = settings.corporate_action_lock_path
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"sha256": {
        actions.as_posix(): sha256_file(actions),
        audit.as_posix(): sha256_file(audit),
    }}), encoding="utf-8")
    lock.with_suffix(lock.suffix + ".sha256").write_text(sha256_file(lock) + "\n")
    benchmark_evidence = tmp_path / "benchmark-manifest.json"
    write_immutable_json(benchmark_evidence, {"sha256": {benchmark_path.as_posix(): sha256_file(benchmark_path)}})
    manifest = {
        "market": {"status": "APPROVED", "path": market_path.as_posix(),
                   "data_audit_path": audit.as_posix()},
        "benchmark": ({"status": "APPROVED", "path": benchmark_path.as_posix(),
                       "evidence_manifest_path": benchmark_evidence.as_posix()}
                      if benchmark_approved else {"status": "UNAPPROVED", "path": None}),
        "corporate_actions": {"dataset_path": actions.as_posix(),
                              "trusted_lock_path": lock.as_posix(),
                              "trusted_lock_sha256": sha256_file(lock)},
        "trading_calendar": {"path": settings.calendar_path.as_posix(),
                             "sha256": sha256_file(settings.calendar_path)},
    }
    write_immutable_json(settings.settlement_manifest_path, manifest)
    loader = lambda configured: load_approved_settlement_bundle(
        configured, lock_verifier=lambda _: {"frozen_inputs_intact": True}
    )
    return settings, market, benchmark, actions, lock, loader


def test_fake_action_manifest_without_trusted_lock_fails(tmp_path):
    settings, _, _, actions, lock, _ = _trust_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="lock hash mismatch"):
        verify_corporate_action_trust_root(actions, lock, "0" * 64, "m" * 64)


def test_trusted_action_chain_verifies(tmp_path):
    settings, _, _, actions, lock, _ = _trust_fixture(tmp_path)
    result = verify_corporate_action_trust_root(
        actions, lock, sha256_file(lock), "m" * 64
    )
    assert result["corporate_action_verified"]


@pytest.mark.parametrize("kind", ["dataset", "manifest", "lock"])
def test_mutated_action_trust_chain_fails(tmp_path, kind):
    settings, _, _, actions, lock, loader = _trust_fixture(tmp_path)
    if kind == "dataset": actions.write_text('{"changed":true}')
    elif kind == "manifest": settings.settlement_manifest_path.write_text("{}")
    else: lock.write_text("{}")
    with pytest.raises((RuntimeError, KeyError)):
        loader(settings)


@pytest.mark.parametrize("kind", ["market", "benchmark"])
def test_settlement_dataframe_provenance_mismatch(tmp_path, kind):
    settings, market, benchmark, _, _, loader = _trust_fixture(tmp_path)
    bundle = loader(settings)
    supplied_market, supplied_benchmark = market.copy(), benchmark.copy()
    if kind == "market": supplied_market.loc[0, "open"] += 1
    else: supplied_benchmark.loc[0, "open"] += 1
    with pytest.raises(RuntimeError, match=f"{kind} DataFrame provenance mismatch"):
        settle_certified_labels(
            pd.DataFrame({"date": [market.date.iloc[0]], "symbol": ["000001"]}),
            bundle=bundle, settings=settings, as_of=market.date.max(),
            expected_universe_by_date={str(market.date.iloc[0].date()): {"000001"}},
            market=supplied_market, benchmark=supplied_benchmark,
        )


@pytest.mark.parametrize("observation,prediction,settlement,expected", [
    ("SUCCESS", "RECORDED", "NO_MATURE_LABELS", "COMPLETE"),
    ("SUCCESS", "INPUT_NOT_AVAILABLE", "NO_MATURE_LABELS", "DERIVATIVES_PENDING"),
    ("SUCCESS", "RECORDED", "SETTLEMENT_BLOCKED_BENCHMARK_UNAPPROVED", "DERIVATIVES_PENDING"),
    ("PARTIAL", "RECORDED", "NO_MATURE_LABELS", "PARTIAL"),
    ("INTERRUPTED", "RECORDED", "NO_MATURE_LABELS", "INTERRUPTED"),
])
def test_daily_status_aggregation(observation, prediction, settlement, expected):
    assert aggregate_daily_status(observation, prediction, settlement) == expected


def _daily_dependencies(settlement, prediction_status="RECORDED"):
    certification = lambda item, settings: {
        "target_date": item["target_date"], "qualifying_observation": True,
        "certification_sha256": "c" * 64,
    }
    return DailyDependencies(
        lock_verifier=_lock,
        context_loader=_context,
        source_fetcher_factory=_factory,
        certification_runner=certification,
        observation_certifier=_fake_cert,
        label_certifier=_fake_label,
        prediction_runner=lambda *_: {"status": prediction_status},
        settlement_runner=settlement,
    )


def _write_label(settings: OperationalSettings, date: str, symbol: str, horizon: int):
    target = settings.labels_root / date / f"{symbol}_{horizon}d.json"
    write_immutable_json(target, {
        "prediction_date": date, "symbol": symbol, "horizon": horizon,
        "expected_universe_size": 2, "verified": True,
    })


def test_daily_readiness_before_reads_existing_labels(tmp_path):
    settings = _settings(tmp_path)
    _context_files(settings)
    for horizon in (1, 5, 20):
        for symbol in ("000001", "000002"):
            _write_label(settings, "2026-08-01", symbol, horizon)
    result = run_daily(now=NOW, settings=settings,
                       dependencies=_daily_dependencies(lambda *_: {"status": "NO_MATURE_LABELS", "mature_records_written": 0}))
    assert result["readiness_before"]["mature_1d_count"] == 1


def test_daily_readiness_after_includes_newly_settled_labels_and_matches_canonical(tmp_path):
    settings = _settings(tmp_path)
    _context_files(settings)
    def settlement(_, configured):
        for horizon in (1, 5, 20):
            for symbol in ("000001", "000002"):
                _write_label(configured, DATE, symbol, horizon)
        return {"status": "SETTLED", "mature_records_written": 6}
    deps = _daily_dependencies(settlement)
    result = run_daily(now=NOW, settings=settings, dependencies=deps)
    assert result["readiness_before"]["mature_1d_count"] == 0
    assert result["readiness_after"]["mature_1d_count"] == 1
    from stockpilot.prospective_r2.observation import load_verified_observations
    from stockpilot.prospective_r3.settlement import load_verified_label_records
    canonical = build_runtime_status(
        settings, load_verified_observations(settings), load_verified_label_records(settings.labels_root),
        observation_certifier=_fake_cert, label_certifier=_fake_label,
    ).to_dict()
    assert result["readiness_after"] == canonical
    assert read_verified_json(result["daily_receipt_path"])["readiness_after"] == canonical


def test_cli_status_uses_canonical_builder(monkeypatch, tmp_path):
    from stockpilot.prospective_r3 import cli
    settings = _settings(tmp_path)
    expected = build_runtime_status(settings, [], []).to_dict()
    monkeypatch.setattr(cli, "load_verified_observations", lambda _: [])
    monkeypatch.setattr(cli, "load_verified_label_records", lambda _: [])
    assert cli.status(settings) == expected


def test_settlement_runs_when_today_observation_fails(tmp_path):
    settings = _settings(tmp_path)
    _context_files(settings)
    called = []
    def failed_factory(*_):
        return {"earnings_expectations": ({}, lambda: (_ for _ in ()).throw(ConnectionError("offline")))}
    deps = replace(_daily_dependencies(lambda *_: called.append(True) or {
        "status": "NO_MATURE_LABELS", "mature_records_written": 0}), source_fetcher_factory=failed_factory)
    run_daily(now=NOW, settings=settings, dependencies=deps)
    assert called == [True]


def test_keyboard_interrupt_is_daily_interrupted(tmp_path):
    settings = _settings(tmp_path)
    _context_files(settings)
    deps = replace(_daily_dependencies(lambda *_: {"status": "NO_MATURE_LABELS"}),
                   source_fetcher_factory=lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt): run_daily(now=NOW, settings=settings, dependencies=deps)
    assert read_verified_json(settings.daily_receipts_root / f"{DATE}.json")["daily_status"] == "INTERRUPTED"


def test_v1r2_runtime_is_fail_closed_before_provider(monkeypatch):
    from stockpilot.prospective_r2.orchestrator import (
        DailyDependencies as R2Dependencies,
        run_daily as run_v1r2,
    )
    calls = []
    def provider(*args):
        calls.append(args)
        return {}
    with pytest.raises(Exception):
        run_v1r2(now=NOW, dependencies=R2Dependencies(source_fetcher_factory=provider))
    assert calls == []


def test_all_readiness_safety_states_remain_false(tmp_path):
    status = build_runtime_status(_settings(tmp_path), [], [])
    assert not status.model_training_ready
    assert not status.replacement_evaluation_ready
    assert not status.production_prediction_ready
    assert not status.execution_authorized
    assert not status.v31_trained


def test_no_v31_training_entrypoint_exists():
    root = Path(__file__).resolve().parents[1]
    cli = (root / "stockpilot/prospective_r3/cli.py").read_text(encoding="utf-8")
    assert 'add_parser("train")' not in cli
    assert not list((root / "stockpilot/prospective_r3").glob("*v31*"))
