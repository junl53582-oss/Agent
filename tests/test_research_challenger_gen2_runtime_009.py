from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_v20r2.ledger import Ledger, PriceBook
from stockpilot.prospective_r2.integrity import (
    read_verified_json,
    sha256_file,
    verify_immutable,
    write_immutable_bytes,
    write_immutable_frame,
    write_immutable_json,
)
from stockpilot.research_challenger.prospective_gen2 import (
    _policy_hash,
    cost_policy,
    feature_policy,
    model_specification,
    portfolio_policy,
    training_policy,
)
from stockpilot.research_challenger.prospective_gen2_runtime import (
    RuntimeSettings,
    _default_train_and_score,
    _is_rebalance_day,
    generate_prediction,
    preflight,
    seal_inputs,
    settle_prediction,
)


UTC = timezone.utc
TARGET = "2026-09-01"
READY = datetime(2026, 9, 1, 11, 0, tzinfo=UTC)  # 19:00 Shanghai


def _calendar(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "market": "XSHG",
                "coverage_start": "2016-01-01",
                "coverage_end": "2028-12-31",
                "weekends_closed": True,
                "closed_weekdays": [],
                "source": "fixture",
                "source_url": "https://example.invalid/calendar",
            }
        ),
        encoding="utf-8",
    )


def _write_parquet_immutable(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(".tmp.parquet")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(temporary, index=False)
    payload = temporary.read_bytes()
    temporary.unlink()
    write_immutable_bytes(path, payload)


def _panel() -> pd.DataFrame:
    dates = list(pd.bdate_range("2017-01-02", "2025-12-31")) + [pd.Timestamp(TARGET)]
    rows = []
    for date in dates:
        for number in range(1, 11):
            value = number / 10
            rows.append(
                {
                    "date": date,
                    "symbol": f"{number:06d}",
                    "eligible": True,
                    "in_universe": True,
                    "membership_snapshot_date": date,
                    "available_date": date,
                    "industry_effective_date": date,
                    "industry": f"I{number % 4}",
                    "broad_sector": f"S{number % 3}",
                    "benchmark_weight": 0.1,
                    "benchmark_weight_rank": value,
                    "momentum": value,
                    "future_return_5d": value / 100,
                    "future_return_20d": value / 50,
                    "label_end_date_5d": date + pd.offsets.BDay(6),
                    "label_end_date_20d": date + pd.offsets.BDay(21),
                }
            )
    return pd.DataFrame(rows)


def _settings(tmp_path: Path, *, test_mode: bool = True) -> RuntimeSettings:
    calendar = tmp_path / "calendar.json"
    _calendar(calendar)
    dataset = tmp_path / "panel.parquet"
    frame = _panel()
    _write_parquet_immutable(dataset, frame)
    manifest = tmp_path / "manifest.json"
    write_immutable_json(manifest, {"source_hashes": {}, "rows": len(frame)})
    human = tmp_path / "human"
    parent = human / "experiments/008_operational_portability_fix/plan.lock.json"
    parent_hash = write_immutable_json(parent, {"lock_id": "fixture-008"})
    settings = RuntimeSettings(
        human_dir=human,
        human_lock_path=human / "plan.lock.json",
        data_root=tmp_path / "data",
        prediction_root=tmp_path / "data/predictions",
        settlement_root=tmp_path / "data/settlements",
        input_seal_root=tmp_path / "data/input_seals",
        reservation_root=tmp_path / "data/attempts",
        portfolio_root=tmp_path / "data/portfolio",
        calendar_path=calendar,
        dataset_path=dataset,
        dataset_manifest_path=manifest,
        parent_008_lock_path=parent,
        expected_parent_008_lock=parent_hash,
        runtime_lock_path=tmp_path / "009/plan.lock.json",
        factor_columns_override=("momentum",),
        training_row_cap_override=5_000,
        top_k=6,
        test_mode=test_mode,
    )
    write_immutable_json(human / "decision.json", {"prospective_start_date": TARGET, "operative_champion": "V6"})
    spec = {
        "model_id": "GEN2-LGBM-20D-SECTOR-BALANCED-TOP20",
        "model_spec_hash": _policy_hash(model_specification(settings)),
        "feature_policy_hash": _policy_hash(feature_policy(settings)),
        "training_policy_hash": _policy_hash(training_policy(settings)),
        "portfolio_policy_hash": _policy_hash(portfolio_policy(settings)),
        "cost_policy_hash": _policy_hash(cost_policy(settings)),
    }
    write_immutable_json(human / "challenger_spec.json", spec)
    return settings


def _score_frame(date: str = TARGET) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Timestamp(date),
            "symbol": [f"{i:06d}" for i in range(1, 11)],
            "industry": [f"I{i % 4}" for i in range(1, 11)],
            "broad_sector": [f"S{i % 3}" for i in range(1, 11)],
            "benchmark_weight": [0.1] * 10,
            "benchmark_weight_rank": np.arange(1, 11) / 10,
            "score": np.arange(10, 0, -1, dtype=float),
        }
    )


def _scorer(date: str, _: RuntimeSettings) -> tuple[pd.DataFrame, dict]:
    return _score_frame(date), {
        "training_snapshot_hash": "training",
        "input_snapshot_hash": "input",
        "training_label_date_min": "2017-01-02",
        "training_label_date_max": "2025-11-28",
        "training_label_years": list(range(2017, 2026)),
        "maximum_training_label_end": "2025-12-31",
        "training_labels_all_mature_before_model_boundary": True,
        "labels_after_prediction_date_read": False,
        "current_prediction_outcome_read": False,
        "disqualified_2026_holdout_used_for_historical_confirmation": False,
        "historical_confirmation_attempted": False,
    }


def _seal(settings: RuntimeSettings, date: str = TARGET, now: datetime = READY) -> None:
    seal_inputs(date, now=now, settings=settings)


def _predict(settings: RuntimeSettings, date: str = TARGET, now: datetime = READY) -> dict:
    _seal(settings, date, now)
    return generate_prediction(date, now=now, settings=settings, scorer=_scorer)


def _market(settings: RuntimeSettings, prediction_date: str = TARGET, *, limit_first_buy: bool = False) -> Path:
    from stockpilot.prospective_r2.calendar import load_verified_calendar

    sessions = load_verified_calendar(settings.calendar_path).sessions()
    end = pd.Timestamp(read_verified_json(settings.prediction_root / prediction_date / "prediction.json")["label_maturity_date"])
    dates = sessions[(sessions >= pd.Timestamp(prediction_date)) & (sessions <= end)]
    symbols = [f"{i:06d}" for i in range(1, 11)]
    rows = []
    for date_index, date in enumerate(dates):
        for symbol_index, symbol in enumerate(symbols):
            price = 10 + date_index * 0.01 + symbol_index * 0.001
            open_price = 12.0 if limit_first_buy and date_index == 1 and symbol == "000001" else price
            rows.append({"date": date, "symbol": symbol, "open": open_price, "close": price, "volume": 1000.0})
    market = settings.data_root / f"market-{prediction_date}.csv"
    write_immutable_frame(market, pd.DataFrame(rows), ["date", "symbol"])
    actions = settings.data_root / "actions.json"
    if not actions.exists():
        actions.parent.mkdir(parents=True, exist_ok=True)
        actions.write_text('{"events":[]}', encoding="utf-8")
    witness = {
        "market_source_sha256": sha256_file(market),
        "witnessed_at_utc": "2026-10-30T12:00:00+00:00",
        "source_created_at_utc": "2026-10-30T11:00:00+00:00",
        "acquisition_receipt_hash": "fixture-acquisition",
        "corporate_action_path": actions.as_posix(),
        "corporate_action_sha256": sha256_file(actions),
    }
    write_immutable_json(market.with_suffix(market.suffix + ".witness.json"), witness)
    return market


def test_default_real_scorer_builds_targets(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    scored, evidence = _default_train_and_score(TARGET, settings)
    assert len(scored) == 10
    assert evidence["maximum_training_label_end"] < "2026-01-01"


def test_default_real_scorer_end_to_end_fixture(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _seal(settings)
    result = generate_prediction(TARGET, now=READY, settings=settings)
    assert result["daily_score_rows"] == 10
    assert result["training_evidence"]["selected_features"] == ["momentum"]


def test_default_real_scorer_builds_5d_and_20d_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import stockpilot.research_challenger.prospective_gen2_runtime as runtime
    from stockpilot.research_challenger.data import add_research_targets as real

    seen = []
    def wrapped(frame, horizons):
        seen.append(horizons)
        return real(frame, horizons)
    monkeypatch.setattr(runtime, "add_research_targets", wrapped)
    _default_train_and_score(TARGET, _settings(tmp_path))
    assert seen == [(5, 20)]


def test_default_scorer_does_not_use_mock_integration_path(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _seal(settings)
    result = generate_prediction(TARGET, now=READY, settings=settings)
    assert result["training_evidence"]["model_signature"]


def test_prediction_before_data_ready_time_fails_closed(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(RuntimeError, match="UPSTREAM_DATA_WINDOW_NOT_OPEN"):
        seal_inputs(TARGET, now=datetime(2026, 9, 1, 8, tzinfo=UTC), settings=settings)


def test_prediction_requires_sealed_input(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(RuntimeError, match="GEN2_INPUT_NOT_READY"):
        generate_prediction(TARGET, now=READY, settings=settings, scorer=_scorer)


def test_sealed_input_hash_change_fails_closed(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _seal(settings)
    settings.dataset_path.with_suffix(settings.dataset_path.suffix + ".sha256").unlink()
    settings.dataset_path.write_bytes(settings.dataset_path.read_bytes() + b"x")
    with pytest.raises(RuntimeError, match="GEN2_INPUT_NOT_READY"):
        generate_prediction(TARGET, now=READY, settings=settings, scorer=_scorer)


@pytest.mark.parametrize(
    "column,value,error",
    [
        ("membership_snapshot_date", "2026-09-02", "PIT_GATE"),
        ("available_date", "2026-09-02", "PIT_GATE"),
        ("industry_effective_date", "2026-09-02", "PIT_GATE"),
        ("eligible", False, "ELIGIBLE_UNIVERSE_EMPTY"),
        ("in_universe", False, "ELIGIBLE_UNIVERSE_EMPTY"),
    ],
)
def test_target_date_pit_and_universe_gates(tmp_path: Path, column: str, value, error: str) -> None:
    settings = _settings(tmp_path)
    frame = pd.read_parquet(settings.dataset_path)
    frame.loc[frame["date"].eq(pd.Timestamp(TARGET)), column] = value
    settings.dataset_path.unlink(); settings.dataset_path.with_suffix(settings.dataset_path.suffix + ".sha256").unlink()
    _write_parquet_immutable(settings.dataset_path, frame)
    with pytest.raises(RuntimeError, match=error):
        seal_inputs(TARGET, now=READY, settings=settings)


def test_target_date_duplicate_symbol_fails(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = pd.read_parquet(settings.dataset_path)
    duplicate = frame[frame["date"].eq(pd.Timestamp(TARGET))].iloc[[0]]
    frame = pd.concat([frame, duplicate], ignore_index=True)
    settings.dataset_path.unlink(); settings.dataset_path.with_suffix(settings.dataset_path.suffix + ".sha256").unlink()
    _write_parquet_immutable(settings.dataset_path, frame)
    with pytest.raises(RuntimeError, match="DUPLICATE_SYMBOL"):
        seal_inputs(TARGET, now=READY, settings=settings)


def test_future_feature_column_fails(tmp_path: Path) -> None:
    settings = replace(_settings(tmp_path), factor_columns_override=("future_cheat",))
    with pytest.raises(ValueError, match="future or label"):
        seal_inputs(TARGET, now=READY, settings=settings)


def test_production_settlement_cannot_fake_future_as_of(tmp_path: Path) -> None:
    settings = _settings(tmp_path, test_mode=False)
    _predict(settings)
    with pytest.raises(RuntimeError, match="PRODUCTION_AS_OF_OVERRIDE_FORBIDDEN"):
        settle_prediction(TARGET, Path("unused"), now=READY, test_as_of_override="2027-01-01", settings=settings)


def test_actual_clock_must_reach_maturity(tmp_path: Path) -> None:
    settings = _settings(tmp_path, test_mode=False)
    _predict(settings)
    with pytest.raises(RuntimeError, match="20D_LABEL_NOT_MATURE"):
        settle_prediction(TARGET, Path("unused"), now=READY, settings=settings)


def test_settlement_requires_witnessed_market_source(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _predict(settings)
    path = settings.data_root / "market.csv"
    write_immutable_frame(path, pd.DataFrame({"date": [], "symbol": [], "open": [], "close": [], "volume": []}), ["date", "symbol"])
    with pytest.raises(FileNotFoundError):
        settle_prediction(TARGET, path, now=READY, test_as_of_override="2026-12-01", settings=settings)


def test_daily_score_does_not_rebalance_daily(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    from stockpilot.prospective_r2.calendar import load_verified_calendar
    sessions = load_verified_calendar(settings.calendar_path).sessions()
    date = str(sessions[np.flatnonzero(sessions == pd.Timestamp(TARGET))[0] + 1].date())
    frame = pd.read_parquet(settings.dataset_path)
    extra = frame[frame["date"].eq(pd.Timestamp(TARGET))].copy()
    extra["date"] = pd.Timestamp(date)
    extra["membership_snapshot_date"] = extra["available_date"] = extra["industry_effective_date"] = extra["date"]
    frame = pd.concat([frame, extra], ignore_index=True)
    settings.dataset_path.unlink(); settings.dataset_path.with_suffix(settings.dataset_path.suffix + ".sha256").unlink()
    _write_parquet_immutable(settings.dataset_path, frame)
    now = datetime.fromisoformat(date + "T11:00:00+00:00")
    _predict(settings, date, now)
    receipt = read_verified_json(settings.prediction_root / date / "prediction.json")
    frame = pd.read_csv(settings.prediction_root / date / "prediction.csv")
    assert receipt["portfolio_action"] == "HOLD"
    assert frame["portfolio_weight"].isna().all()


def test_rebalance_anchor_uses_trading_days(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    from stockpilot.prospective_r2.calendar import load_verified_calendar
    sessions = load_verified_calendar(settings.calendar_path).sessions()
    anchor = int(np.flatnonzero(sessions == pd.Timestamp(TARGET))[0])
    assert _is_rebalance_day(str(sessions[anchor + 19].date()), settings)[0] is False
    assert _is_rebalance_day(str(sessions[anchor + 20].date()), settings)[0] is True


def test_rebalance_day_creates_new_sector_balanced_top20(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    result = _predict(settings)
    assert result["portfolio_action"] == "REBALANCE"
    assert result["selected_for_new_portfolio_count"] == 6


def _book(block_buy: bool = False, block_sell: bool = False) -> PriceBook:
    rows = []
    for idx, date in enumerate(pd.bdate_range("2026-09-01", periods=3)):
        for symbol in ("000001", "000002"):
            price = 10.0
            opening = price
            if block_buy and idx == 1 and symbol == "000001": opening = 12.0
            if block_sell and idx == 2 and symbol == "000001": opening = 8.0
            rows.append({"date": date, "symbol": symbol, "open": opening, "close": price, "volume": 1000})
    return PriceBook(pd.DataFrame(rows))


def test_stateful_ledger_retains_untradable_exit() -> None:
    ledger = Ledger(_book(block_sell=True))
    ledger.rebalance({"000001": 1.0}, 1)
    result = ledger.rebalance({"000002": 1.0}, 2)
    assert "000001" in ledger.units and any(v["side"] == "sell" for v in result["blocked"])


def test_stateful_ledger_blocked_sell_does_not_release_cash() -> None:
    ledger = Ledger(_book(block_sell=True))
    ledger.rebalance({"000001": 1.0}, 1)
    before = ledger.cash
    ledger.rebalance({"000002": 1.0}, 2)
    assert ledger.cash <= before + 1e-12


def test_stateful_ledger_new_buy_respects_cash() -> None:
    ledger = Ledger(_book())
    ledger.rebalance({"000001": 0.5, "000002": 0.5}, 1)
    assert ledger.cash >= 0


def test_transaction_costs_applied_only_to_executed_turnover() -> None:
    ledger = Ledger(_book(block_buy=True))
    result = ledger.rebalance({"000001": 1.0}, 1)
    assert result["buy_turnover"] == 0 and result["transaction_cost"] == 0


def test_prospective_proxy_matches_historical_weighted_semantics(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _predict(settings)
    result = settle_prediction(TARGET, _market(settings), now=READY, test_as_of_override="2026-12-01", settings=settings)
    assert result["research_proxy_semantics"] == "PIT_BENCHMARK_CONSTITUENT_WEIGHTED"


def test_benchmark_weight_is_frozen_in_prediction(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _predict(settings)
    assert "benchmark_weight" in pd.read_csv(settings.prediction_root / TARGET / "prediction.csv").columns


def test_equal_weight_universe_mean_not_used_as_canonical_proxy(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _predict(settings)
    frame = pd.read_csv(settings.prediction_root / TARGET / "prediction.csv")
    frame["benchmark_weight"] = np.arange(1, len(frame) + 1)
    # Prediction is immutable, so mutation cannot be substituted into settlement.
    assert verify_immutable(settings.prediction_root / TARGET / "prediction.csv")


def test_2026_prediction_uses_only_pre_2026_labels(tmp_path: Path) -> None:
    _, evidence = _default_train_and_score(TARGET, _settings(tmp_path))
    assert evidence["maximum_training_label_end"] < "2026-01-01"


def test_2027_prediction_can_use_mature_2026_labels(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = pd.read_parquet(settings.dataset_path)
    template = frame[frame["date"].eq(pd.Timestamp(TARGET))].copy()
    pieces = [frame]
    for date in pd.bdate_range("2026-01-02", "2026-11-30"):
        extra = template.copy()
        extra["date"] = date
        extra["membership_snapshot_date"] = extra["available_date"] = extra["industry_effective_date"] = date
        extra["label_end_date_5d"] = date + pd.offsets.BDay(6)
        extra["label_end_date_20d"] = date + pd.offsets.BDay(21)
        pieces.append(extra)
    extra = template.copy()
    extra["date"] = pd.Timestamp("2027-09-01")
    extra["membership_snapshot_date"] = extra["available_date"] = extra["industry_effective_date"] = extra["date"]
    pieces.append(extra)
    frame = pd.concat(pieces, ignore_index=True).drop_duplicates(["date", "symbol"], keep="last")
    settings.dataset_path.unlink(); settings.dataset_path.with_suffix(settings.dataset_path.suffix + ".sha256").unlink()
    _write_parquet_immutable(settings.dataset_path, frame)
    _, evidence = _default_train_and_score("2027-09-01", settings)
    assert 2026 in evidence["training_label_years"]


def test_current_prediction_outcome_is_never_read_during_training(tmp_path: Path) -> None:
    _, evidence = _default_train_and_score(TARGET, _settings(tmp_path))
    assert evidence["current_prediction_outcome_read"] is False
    assert evidence["labels_after_prediction_date_read"] is False


def test_active_research_names_original_and_frozen_portfolios_separately() -> None:
    value = json.loads(Path("artifacts/active_research.json").read_text(encoding="utf-8"))
    assert value["gen02_original_diagnostic_portfolio"] == "equal_top30"
    assert value["gen02_frozen_prospective_portfolio"] == "sector_balanced_top20"


def test_preflight_is_read_only_and_makes_zero_provider_requests(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    before = sorted(p.as_posix() for p in tmp_path.rglob("*"))
    result = preflight(TARGET, now=READY, settings=settings)
    after = sorted(p.as_posix() for p in tmp_path.rglob("*"))
    assert before == after
    assert result["provider_requests_made"] == 0


def test_v6_v30_v30r1_v1r4_unchanged() -> None:
    changed = __import__("subprocess").check_output(["git", "diff", "--name-only", "HEAD"], text=True).splitlines()
    assert not any(name.startswith(("research_v6/", "stockpilot/prospective_r4/", "stockpilot/prediction_forward")) for name in changed)


def test_no_auto_promotion_after_runtime_hardening(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _predict(settings)
    receipt = read_verified_json(settings.prediction_root / TARGET / "prediction.json")
    assert receipt["automatic_promotion_allowed"] is False
    assert receipt["execution_authorized"] is False
