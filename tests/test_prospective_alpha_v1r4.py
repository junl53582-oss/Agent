from __future__ import annotations

import json
import multiprocessing
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from stockpilot.prospective_r2.integrity import sha256_file, write_immutable_json
from stockpilot.prospective_r2.observation import SourceCapture, reserve_daily_attempt
from stockpilot.prospective_r3.orchestrator import DailyDependencies
from stockpilot.prospective_r3.settlement import SettlementBundle, settle_certified_labels
from stockpilot.prospective_r3.status import build_runtime_status
from stockpilot.prospective_r4.benchmark import APPROVED_IDENTITY, verify_benchmark_evidence
from stockpilot.prospective_r4.config import OperationalSettings
from stockpilot.prospective_r4.orchestrator import run_daily
from stockpilot.prospective_r4.preflight import (
    DailyPreflightBlocked,
    run_preflight,
    seal_prediction_inputs,
)
from stockpilot.prospective_r4.settlement import run_operational_settlement


DATE = "2026-08-31"
AFTER = datetime(2026, 8, 31, 11, tzinfo=timezone.utc)  # 19:00 Shanghai
BEFORE = datetime(2026, 8, 31, 6, tzinfo=timezone.utc)  # 14:00 Shanghai
SYMBOLS = tuple(f"{index:06d}" for index in range(1, 301))


def _settings(tmp_path: Path) -> OperationalSettings:
    artifact = tmp_path / "artifacts"
    calendar = artifact / "calendar.json"
    calendar.parent.mkdir(parents=True, exist_ok=True)
    calendar.write_text(json.dumps({
        "market": "XSHG", "coverage_start": "2026-01-01", "coverage_end": "2026-12-31",
        "weekends_closed": True, "closed_weekdays": ["2026-10-01"],
        "source": "fixture", "source_url": "https://example.invalid/calendar",
    }), encoding="utf-8")
    membership = tmp_path / "membership.csv"
    pd.DataFrame({"snapshot_date": [DATE] * 300, "symbol": SYMBOLS}).to_csv(membership, index=False)
    industry = tmp_path / "industry.csv"
    pd.DataFrame({
        "symbol": SYMBOLS,
        "industry": ["industry"] * 300,
        "industry_effective_date": ["2026-01-01"] * 300,
    }).to_csv(industry, index=False)
    plan = artifact / "plan.lock.json"
    plan.write_text('{"fixture":true}', encoding="utf-8")
    return OperationalSettings(
        data_root=tmp_path / "runtime",
        artifact_root=artifact,
        calendar_path=calendar,
        membership_path=membership,
        industry_path=industry,
        plan_lock_path=plan,
        settlement_manifest_path=artifact / "settlement.json",
        prediction_market_template=str(tmp_path / "inputs" / "hfq_union_{date}.csv"),
        prediction_ranking_template=str(tmp_path / "rankings" / "{date}.csv"),
        prediction_root=tmp_path / "predictions",
        corporate_action_path=tmp_path / "actions.json",
        corporate_action_lock_path=tmp_path / "actions.lock.json",
        v1r3_barrier_path=tmp_path / "v1r3-barrier" / "observation.json",
    )


def _lock(settings: OperationalSettings) -> dict:
    return {
        "v1r4_lock_sha256": sha256_file(settings.plan_lock_path),
        "frozen_inputs_intact": True,
    }


def _write_inputs(settings: OperationalSettings, *, ranking_date=DATE, market_date=DATE) -> None:
    market = Path(settings.prediction_market_template.format(date=DATE))
    market.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "date": [market_date] * 300,
        "symbol": SYMBOLS,
        "open": [10.0] * 300,
        "high": [11.0] * 300,
        "low": [9.0] * 300,
        "close": [10.5] * 300,
        "volume": [1000.0] * 300,
        "amount": [10000.0] * 300,
    })
    frame.to_csv(market, index=False)
    market.with_name(market.name.replace(".csv", ".manifest.json")).write_text(json.dumps({
        "output_rows": 300, "output_symbols": 300, "date_max": market_date,
        "price_positive": True,
    }), encoding="utf-8")
    market.with_name(market.name.replace(".csv", ".failures.csv")).write_text("symbol\n", encoding="utf-8")
    ranking = Path(settings.prediction_ranking_template.format(date=DATE))
    ranking.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "date": [ranking_date] * 240,
        "symbol": SYMBOLS[:240],
        "score": [float(value) for value in range(240, 0, -1)],
        "pred_rank": list(range(1, 241)),
        "generated_at_utc": ["2026-08-31T10:00:00+00:00"] * 240,
        "model": ["research_v6_sector_balanced_ensemble"] * 240,
        "protocol_status": ["retrospective_research"] * 240,
        "execution_authorized": [False] * 240,
        "plan_lock_sha256": [
            "94edfc9e05bd30a58a14e7e11a988a1b7fb0d5358e462df1b20cb23dca4c0f4d"
        ] * 240,
        "training_cutoff": ["2025-12-31"] * 240,
    }).to_csv(ranking, index=False)


def _seal(settings: OperationalSettings) -> dict:
    _write_inputs(settings)
    return seal_prediction_inputs(DATE, now=AFTER, settings=settings, lock_verifier=_lock)


def _reservation(settings: OperationalSettings) -> Path:
    return settings.attempts_root / f"{DATE}.json"


def test_morning_run_rejected_without_reservation_or_provider(tmp_path):
    settings = _settings(tmp_path)
    result = run_preflight(target_date=DATE, now=BEFORE, settings=settings, lock_verifier=_lock)
    assert result["status"] == "TRADING_SESSION_NOT_CLOSED"
    assert result["provider_requests_made"] == 0
    assert not _reservation(settings).exists()


@pytest.mark.parametrize(
    ("date", "now", "status"),
    [
        ("2026-08-30", datetime(2026, 8, 30, 11, tzinfo=timezone.utc), "NOT_VERIFIED_SHANGHAI_TRADING_SESSION"),
        ("2026-08-28", AFTER, "HISTORICAL_DATE_FORBIDDEN"),
        ("2026-09-01", AFTER, "FUTURE_DATE_FORBIDDEN"),
    ],
)
def test_invalid_dates_rejected_before_reservation(tmp_path, date, now, status):
    settings = _settings(tmp_path)
    result = run_preflight(target_date=date, now=now, settings=settings, lock_verifier=_lock)
    assert result["status"] == status
    assert not (settings.attempts_root / f"{date}.json").exists()


def test_missing_market_blocks_without_reservation(tmp_path):
    settings = _settings(tmp_path)
    result = run_preflight(target_date=DATE, now=AFTER, settings=settings, lock_verifier=_lock)
    assert result["status"] == "DAILY_PREFLIGHT_BLOCKED_PREDICTION_INPUT"
    assert not _reservation(settings).exists()


def test_missing_ranking_blocks_without_reservation(tmp_path):
    settings = _settings(tmp_path)
    _write_inputs(settings)
    Path(settings.prediction_ranking_template.format(date=DATE)).unlink()
    with pytest.raises(FileNotFoundError, match="V6_RANKING_INPUT_NOT_READY"):
        seal_prediction_inputs(DATE, now=AFTER, settings=settings, lock_verifier=_lock)
    assert not _reservation(settings).exists()


def test_preflight_is_read_only_and_sealed_inputs_pass(tmp_path):
    settings = _settings(tmp_path)
    sealed = _seal(settings)
    result = run_preflight(target_date=DATE, now=AFTER, settings=settings, lock_verifier=_lock)
    assert sealed["provider_requests_made"] == result["provider_requests_made"] == 0
    assert result["daily_run_allowed"]
    assert result["forward_market_ready"] and result["v6_ranking_ready"]
    assert not _reservation(settings).exists()


@pytest.mark.parametrize("kind", ["market", "ranking"])
def test_mutated_input_hash_blocks_before_reservation(tmp_path, kind):
    settings = _settings(tmp_path)
    _seal(settings)
    path = Path(
        settings.prediction_market_template.format(date=DATE)
        if kind == "market" else settings.prediction_ranking_template.format(date=DATE)
    )
    path.write_bytes(path.read_bytes() + b"\n")
    result = run_preflight(target_date=DATE, now=AFTER, settings=settings, lock_verifier=_lock)
    assert not result["daily_run_allowed"]
    assert not _reservation(settings).exists()


def test_ranking_date_mismatch_cannot_be_sealed(tmp_path):
    settings = _settings(tmp_path)
    _write_inputs(settings, ranking_date="2026-08-28")
    with pytest.raises(ValueError, match="V6_RANKING_DATE_MISMATCH"):
        seal_prediction_inputs(DATE, now=AFTER, settings=settings, lock_verifier=_lock)


def test_market_with_future_date_cannot_be_sealed(tmp_path):
    settings = _settings(tmp_path)
    _write_inputs(settings, market_date="2026-09-01")
    with pytest.raises(ValueError, match="DATE_COVERAGE"):
        seal_prediction_inputs(DATE, now=AFTER, settings=settings, lock_verifier=_lock)


def test_insufficient_market_universe_cannot_be_sealed(tmp_path):
    settings = _settings(tmp_path)
    _write_inputs(settings)
    market = Path(settings.prediction_market_template.format(date=DATE))
    frame = pd.read_csv(market).iloc[:299]
    frame.to_csv(market, index=False)
    manifest = market.with_name(market.name.replace(".csv", ".manifest.json"))
    manifest.write_text(json.dumps({
        "output_rows": 299, "output_symbols": 299, "date_max": DATE, "price_positive": True,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="PIT_UNIVERSE_INCOMPLETE"):
        seal_prediction_inputs(DATE, now=AFTER, settings=settings, lock_verifier=_lock)


def test_insufficient_v6_coverage_cannot_be_sealed(tmp_path):
    settings = _settings(tmp_path)
    _write_inputs(settings)
    ranking = Path(settings.prediction_ranking_template.format(date=DATE))
    pd.read_csv(ranking).iloc[:239].to_csv(ranking, index=False)
    with pytest.raises(ValueError, match="COVERAGE_INCOMPLETE"):
        seal_prediction_inputs(DATE, now=AFTER, settings=settings, lock_verifier=_lock)


def test_v6_execution_flag_cannot_be_true(tmp_path):
    settings = _settings(tmp_path)
    _write_inputs(settings)
    ranking = Path(settings.prediction_ranking_template.format(date=DATE))
    frame = pd.read_csv(ranking); frame["execution_authorized"] = True; frame.to_csv(ranking, index=False)
    with pytest.raises(ValueError, match="EXECUTION_FLAG"):
        seal_prediction_inputs(DATE, now=AFTER, settings=settings, lock_verifier=_lock)


def test_mutated_input_evidence_manifest_blocks(tmp_path):
    settings = _settings(tmp_path)
    sealed = _seal(settings)
    path = Path(sealed["evidence_path"])
    path.write_bytes(path.read_bytes() + b" ")
    result = run_preflight(target_date=DATE, now=AFTER, settings=settings, lock_verifier=_lock)
    assert not result["input_evidence_verified"]
    assert not result["daily_run_allowed"]
    assert not _reservation(settings).exists()


def _capture(source: str, rows: list[dict], *, required=(), confirmed=()) -> SourceCapture:
    return SourceCapture(
        source=source,
        request_parameters={"source": source},
        raw_payloads=(b"raw",),
        normalized=pd.DataFrame(rows),
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
            [{"symbol": symbol, "forecast_year_1": 2027, "forecast_eps_1": 1.0, "industry": "industry"} for symbol in symbols],
            required=("forecast_eps_1",),
        )),
        "announcements": ({}, lambda: _capture(
            "announcements",
            [{"symbol": symbol, "announcement_event_count": 0, "announcement_available": True} for symbol in symbols],
            required=("announcement_event_count",), confirmed=symbols,
        )),
        "fund_flows": ({}, lambda: _capture(
            "fund_flows",
            [{"symbol": symbol, "main_net_inflow": 0.0, "main_net_inflow_ratio": 0.0} for symbol in symbols],
            required=("main_net_inflow",),
        )),
    }


def test_daily_reserves_only_after_full_preflight(tmp_path):
    settings = _settings(tmp_path)
    _seal(settings)
    dependencies = DailyDependencies(
        source_fetcher_factory=_factory,
        prediction_runner=lambda date, configured: {"status": "RECORDED", "date": date},
        settlement_runner=lambda date, configured: {"status": "NO_MATURE_LABELS", "mature_records_written": 0},
    )
    result = run_daily(
        target_date=DATE, now=AFTER, settings=settings, dependencies=dependencies,
        operational_lock_verifier=_lock,
    )
    assert _reservation(settings).exists()
    assert result["preflight"]["daily_run_allowed"]
    assert result["model_retrain_runs"] == result["factor_research_runs"] == 0
    with pytest.raises(DailyPreflightBlocked, match="DAILY_ATTEMPT_ALREADY_RESERVED"):
        run_daily(
            target_date=DATE, now=AFTER, settings=settings, dependencies=dependencies,
            operational_lock_verifier=_lock,
        )


def _reserve_worker(root: str, queue) -> None:
    settings = OperationalSettings(data_root=Path(root), plan_lock_path=Path(root) / "lock.json")
    try:
        reserve_daily_attempt(DATE, AFTER, parent_lock_sha256="a" * 64, settings=settings)
        queue.put("reserved")
    except Exception:
        queue.put("blocked")


def test_concurrent_processes_only_one_reserves(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    processes = [ctx.Process(target=_reserve_worker, args=(str(tmp_path / "runtime"), queue)) for _ in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(20)
        assert process.exitcode == 0
    outcomes = sorted(queue.get(timeout=5) for _ in processes)
    assert outcomes == ["blocked", "reserved"]


def _benchmark(tmp_path: Path, rows: list[dict] | None = None) -> tuple[dict, Path, Path]:
    path = tmp_path / "benchmark.csv"
    pd.DataFrame(rows or [
        {"date": "2026-08-28", "open": 4000.0},
        {"date": "2026-08-31", "open": 4010.0},
    ]).to_csv(path, index=False, lineterminator="\n")
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    evidence_path = tmp_path / "benchmark_evidence.json"
    write_immutable_json(evidence_path, {
        "identity": APPROVED_IDENTITY,
        "source_kind": "OFFICIAL_INDEX_OPEN_SERIES",
        "source_path": path.as_posix(),
        "source_sha256": digest,
        "files": {path.as_posix(): digest},
        "date_min": str(pd.read_csv(path)["date"].min()),
        "date_max": str(pd.read_csv(path)["date"].max()),
        "rows": len(pd.read_csv(path)),
        "fallback_allowed": False,
    })
    return {
        "status": "APPROVED", "identity": APPROVED_IDENTITY, "path": path.as_posix(),
        "evidence_manifest_path": evidence_path.as_posix(),
    }, path, evidence_path


def test_unapproved_benchmark_blocks_settlement(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    bundle = SettlementBundle(
        manifest_path="manifest", manifest_sha256="a", market_status="APPROVED",
        market_path="market", market_sha256="b", benchmark_status="UNAPPROVED",
        benchmark_path=None, benchmark_sha256=None, price_adjustment_mode="HFQ_PIT_GOVERNED",
        corporate_action_dataset_path="actions", corporate_action_dataset_hash="c",
        corporate_action_manifest_hash="d", corporate_action_lock_hash="e",
        corporate_action_lock_verified=True, corporate_action_dataset_verified=True,
        corporate_action_verified=True, trading_calendar_path="calendar", trading_calendar_hash="f",
    )
    monkeypatch.setattr(
        "stockpilot.prospective_r4.settlement.load_operational_settlement_bundle",
        lambda configured, as_of=None: bundle,
    )
    assert run_operational_settlement(DATE, settings)["status"] == "SETTLEMENT_BLOCKED_BENCHMARK_UNAPPROVED"


def test_benchmark_without_evidence_binding_is_blocked(tmp_path):
    benchmark, _, evidence_path = _benchmark(tmp_path)
    payload = json.loads(evidence_path.read_text())
    payload["source_sha256"] = "0" * 64
    evidence_path.unlink(); evidence_path.with_suffix(".json.sha256").unlink()
    write_immutable_json(evidence_path, payload)
    with pytest.raises(RuntimeError, match="BINDING"):
        verify_benchmark_evidence(benchmark, as_of=DATE)


def test_benchmark_source_mutation_is_blocked(tmp_path):
    benchmark, path, _ = _benchmark(tmp_path)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="SIDECAR"):
        verify_benchmark_evidence(benchmark, as_of=DATE)


def test_benchmark_missing_sidecar_is_blocked(tmp_path):
    benchmark, path, _ = _benchmark(tmp_path)
    path.with_suffix(".csv.sha256").unlink()
    with pytest.raises(RuntimeError, match="SIDECAR"):
        verify_benchmark_evidence(benchmark, as_of=DATE)


def test_benchmark_future_date_is_blocked(tmp_path):
    benchmark, _, _ = _benchmark(tmp_path, [{"date": "2026-09-01", "open": 4000.0}])
    with pytest.raises(ValueError, match="AFTER_AS_OF"):
        verify_benchmark_evidence(benchmark, as_of=DATE)


def test_benchmark_duplicate_dates_are_blocked(tmp_path):
    benchmark, _, _ = _benchmark(tmp_path, [
        {"date": DATE, "open": 4000.0}, {"date": DATE, "open": 4001.0},
    ])
    with pytest.raises(ValueError, match="DUPLICATE"):
        verify_benchmark_evidence(benchmark, as_of=DATE)


def test_benchmark_missing_open_is_blocked(tmp_path):
    benchmark, path, evidence = _benchmark(tmp_path)
    pd.DataFrame({"date": [DATE], "close": [4000]}).to_csv(path, index=False)
    digest = sha256_file(path)
    path.with_suffix(".csv.sha256").write_text(digest + "\n", encoding="ascii")
    payload = json.loads(evidence.read_text()); payload["source_sha256"] = digest; payload["files"] = {path.as_posix(): digest}
    evidence.unlink(); evidence.with_suffix(".json.sha256").unlink(); write_immutable_json(evidence, payload)
    with pytest.raises(ValueError, match="SCHEMA"):
        verify_benchmark_evidence(benchmark, as_of=DATE)


def test_portfolio_ledger_cannot_impersonate_benchmark(tmp_path):
    benchmark, _, _ = _benchmark(tmp_path)
    benchmark["identity"] = {**APPROVED_IDENTITY, "instrument_type": "PORTFOLIO_LEDGER"}
    with pytest.raises(RuntimeError, match="IDENTITY"):
        verify_benchmark_evidence(benchmark, as_of=DATE)


def test_valid_official_benchmark_contract_is_recomputable(tmp_path):
    benchmark, _, _ = _benchmark(tmp_path)
    result = verify_benchmark_evidence(benchmark, as_of=DATE)
    assert result["status"] == "APPROVED"
    assert result["identity"] == APPROVED_IDENTITY


def test_open_to_open_horizons_have_no_off_by_one(tmp_path):
    settings = _settings(tmp_path)
    market_path = tmp_path / "market.csv"; benchmark_path = tmp_path / "benchmark.csv"
    dates = pd.bdate_range("2026-08-31", periods=22)
    pd.DataFrame({"date": dates, "symbol": ["000001"] * 22, "open": range(100, 122)}).to_csv(market_path, index=False)
    pd.DataFrame({"date": dates, "open": range(1000, 1022)}).to_csv(benchmark_path, index=False)
    bundle = SettlementBundle(
        manifest_path="manifest", manifest_sha256="a", market_status="APPROVED",
        market_path=str(market_path), market_sha256=sha256_file(market_path),
        benchmark_status="APPROVED", benchmark_path=str(benchmark_path), benchmark_sha256=sha256_file(benchmark_path),
        price_adjustment_mode="HFQ_PIT_GOVERNED", corporate_action_dataset_path="actions",
        corporate_action_dataset_hash="b", corporate_action_manifest_hash="c", corporate_action_lock_hash="d",
        corporate_action_lock_verified=True, corporate_action_dataset_verified=True, corporate_action_verified=True,
        trading_calendar_path="calendar", trading_calendar_hash="e",
    )
    predictions = pd.DataFrame({"date": [str(dates[0].date())], "symbol": ["000001"]})
    # The PIT helper uses the fixture's 2026-08-31 snapshot.
    records = settle_certified_labels(
        predictions, bundle=bundle, settings=settings, as_of=str(dates[-1].date()),
        expected_universe_by_date={str(dates[0].date()): {"000001"}},
    )
    values = {record["horizon"]: record for record in records}
    assert values[1]["entry_date"] == str(dates[1].date()) and values[1]["maturity_date"] == str(dates[2].date())
    assert values[5]["entry_date"] == str(dates[1].date()) and values[5]["maturity_date"] == str(dates[6].date())
    assert values[20]["entry_date"] == str(dates[1].date()) and values[20]["maturity_date"] == str(dates[21].date())


@pytest.mark.parametrize(("symbol_count", "expected", "qualified"), [(239, 300, False), (240, 301, False), (240, 300, True)])
def test_mature_date_requires_symbol_and_coverage_thresholds(tmp_path, symbol_count, expected, qualified):
    settings = _settings(tmp_path)
    labels = [
        {"prediction_date": DATE, "symbol": f"{index:06d}", "horizon": 1, "expected_universe_size": expected}
        for index in range(symbol_count)
    ]
    status = build_runtime_status(
        settings, [], labels,
        observation_certifier=lambda value, configured: {"qualifying_observation": False},
        label_certifier=lambda value, configured: {"label_evidence_verified": True},
    )
    assert (status.mature_1d_count == 1) is qualified


def test_frozen_parent_hashes_unchanged():
    assert sha256_file("artifacts/prospective_alpha_v1r3/plan.lock.json") == "a987f74304718a1aea39d881cad05ec56f252f0bf2fcf37dc2810a518bbd86bb"
    assert sha256_file("artifacts/research_v6/plan.lock.json") == "94edfc9e05bd30a58a14e7e11a988a1b7fb0d5358e462df1b20cb23dca4c0f4d"
    assert sha256_file("artifacts/prediction_forward/v30r1_r2/plan.lock.json") == "110fa074f16235ab413e13b788767913c93e04e051c2105c39b462202d384b17"


def test_frozen_v1r3_entry_is_fail_closed_after_v1r4_activation():
    from stockpilot.prospective_r2.observation import load_verified_observations
    from stockpilot.prospective_r3.config import OperationalSettings as V1R3Settings

    with pytest.raises(Exception):
        load_verified_observations(V1R3Settings())


def test_v1r4_never_promotes_or_trains(tmp_path):
    status = build_runtime_status(_settings(tmp_path), [], []).to_dict()
    assert not status["model_training_ready"]
    assert not status["replacement_evaluation_ready"]
    assert not status["production_prediction_ready"]
    assert not status["execution_authorized"]
    assert not status["v31_trained"]
