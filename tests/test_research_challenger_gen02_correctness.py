from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_v20r2.ledger import Ledger, PriceBook
from stockpilot.research_challenger.gen02 import Gen02Settings, open_holdout_once
from stockpilot.research_challenger.gen02_correctness import (
    choose_configuration_corrected,
    factor_decay,
    load_maturity_safe_development_dataset,
    sector_concentration_metrics,
    summarize_stateful_portfolio,
)


def _factor_frame() -> pd.DataFrame:
    rows = []
    for year in range(2020, 2026):
        for day in pd.bdate_range(f"{year}-01-02", periods=2):
            for symbol in range(20):
                rows.append(
                    {
                        "date": day,
                        "symbol": f"{symbol:06d}",
                        "factor": symbol,
                        "future_return_5d": -symbol if year == 2025 else symbol,
                        "future_return_20d": symbol,
                    }
                )
    return pd.DataFrame(rows)


def test_factor_decay_uses_selected_horizon() -> None:
    selected = {year: ("factor",) for year in range(2020, 2026)}
    frame = _factor_frame()
    decay_5 = factor_decay(frame, selected, 5).iloc[0]
    decay_20 = factor_decay(frame, selected, 20).iloc[0]
    assert decay_5["horizon"] == 5
    assert decay_20["horizon"] == 20
    assert bool(decay_5["sign_flip"]) is True
    assert bool(decay_20["sign_flip"]) is False


def test_20d_failure_analysis_does_not_use_5d_factor_decay() -> None:
    selected = {year: ("factor",) for year in range(2020, 2026)}
    decay = pd.concat([factor_decay(_factor_frame(), selected, h) for h in (5, 20)])
    chosen = decay[decay["horizon"].eq(20)]
    assert not chosen.empty
    assert chosen["horizon"].eq(20).all()
    assert not chosen["sign_flip"].any()


def test_sector_gate_distinguishes_mean_and_worst_case() -> None:
    periods = pd.DataFrame({"maximum_sector_weight": [0.70, 0.30, 0.30, 0.30]})
    metrics = sector_concentration_metrics(periods)
    assert metrics["mean_maximum_sector_weight"] < 0.45
    assert metrics["worst_maximum_sector_weight"] > 0.45


def _ledger_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=5)
    rows = []
    for date in dates:
        for symbol in ("600001", "600002"):
            price = 100.0
            if symbol == "600001" and date == dates[2]:
                price = 90.0
            if symbol == "600001" and date == dates[3]:
                price = 81.0
            rows.append({"date": date, "symbol": symbol, "open": price, "close": price, "volume": 1.0})
    return pd.DataFrame(rows)


def test_exit_untradable_position_is_retained() -> None:
    book = PriceBook(_ledger_panel())
    ledger = Ledger(book, charge_costs=False)
    ledger.rebalance({"600001": 1.0}, 1)
    result = ledger.rebalance({"600002": 1.0}, 2)
    assert "600001" in ledger.units
    assert result["sell_turnover"] == 0


def test_exit_untradable_position_consumes_capital() -> None:
    book = PriceBook(_ledger_panel())
    ledger = Ledger(book, charge_costs=False)
    ledger.rebalance({"600001": 1.0}, 1)
    ledger.rebalance({"600002": 1.0}, 2)
    assert ledger.units.get("600002", 0.0) == 0.0
    assert ledger.cash == pytest.approx(0.0)


def test_position_exits_only_when_sellable() -> None:
    panel = _ledger_panel()
    panel.loc[(panel.symbol.eq("600001")) & (panel.date.eq(panel.date.unique()[3])), ["open", "close"]] = 95.0
    book = PriceBook(panel)
    ledger = Ledger(book, charge_costs=False)
    ledger.rebalance({"600001": 1.0}, 1)
    ledger.rebalance({"600002": 1.0}, 2)
    ledger.rebalance({"600002": 1.0}, 3)
    assert "600001" not in ledger.units
    assert "600002" in ledger.units


def test_untradable_final_liquidation_not_faked() -> None:
    book = PriceBook(_ledger_panel())
    ledger = Ledger(book, charge_costs=False)
    ledger.rebalance({"600001": 1.0}, 1)
    result = ledger.rebalance({}, 2)
    assert "600001" in ledger.units
    assert result["sell_turnover"] == 0
    assert ledger.nav(3) == pytest.approx(0.81)


def _gate_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = pd.DataFrame([
        {"model": m, "horizon": 20, "mean_rank_ic": 0.03, "rank_ic_ir": 0.2, "positive_rank_ic_ratio": 0.55}
        for m in ("ridge", "lightgbm_regression")
    ])
    yearly = pd.DataFrame([
        {"model": m, "horizon": 20, "test_year": 2025, "mean_rank_ic": -0.01}
        for m in ("ridge", "lightgbm_regression")
    ])
    portfolio = pd.DataFrame([
        {
            "model": m, "horizon": 20, "portfolio_policy": p,
            "net_research_proxy_alpha": alpha, "max_drawdown": -0.2,
            "annualized_turnover": 5.0, "mean_maximum_sector_weight": 0.3,
            "p95_maximum_sector_weight": 0.4, "worst_maximum_sector_weight": 0.5,
        }
        for m, p, alpha in (("ridge", "ridge_policy", 0.01), ("lightgbm_regression", "lgb_policy", 0.20))
    ])
    return model, yearly, portfolio


def test_selection_rule_matches_frozen_protocol() -> None:
    eligible, near_miss, table = choose_configuration_corrected(*_gate_inputs(), Gen02Settings())
    assert eligible is None
    assert near_miss["selection_semantics"] == "DIAGNOSTIC_NEAR_MISS_NOT_ELIGIBLE"
    assert not table["all_gates_passed"].any()


def test_near_miss_cannot_become_shadow_eligible() -> None:
    eligible, near_miss, _ = choose_configuration_corrected(*_gate_inputs(), Gen02Settings())
    assert eligible is None
    assert near_miss["net_research_proxy_alpha"] > 0


def test_eligible_configuration_is_not_reused_as_near_miss() -> None:
    model, yearly, portfolio = _gate_inputs()
    yearly["mean_rank_ic"] = 0.01
    portfolio["worst_maximum_sector_weight"] = 0.40
    eligible, near_miss, table = choose_configuration_corrected(
        model, yearly, portfolio, Gen02Settings()
    )
    assert eligible is not None
    assert near_miss is None
    assert table["all_gates_passed"].all()


def test_transaction_cost_rate_sum_not_equal_compounded_drag_semantics() -> None:
    periods = pd.DataFrame(
        {
            "gross_return": [0.10, 0.10], "net_return": [0.09, 0.09],
            "transaction_cost_rate": [0.01, 0.01], "research_benchmark_proxy_return": [0.0, 0.0],
            "buy_turnover": [1.0, 0.0], "sell_turnover": [0.0, 1.0],
            "maximum_sector_weight": [0.4, 0.4], "mean_size_rank": [0.5, 0.5],
            "mean_liquidity_rank": [0.5, 0.5], "top_minus_bottom_spread": [0.01, 0.01],
            "blocked_sell_orders": [0, 0], "terminal_unliquidated_positions": [0, 0],
        }
    )
    result = summarize_stateful_portfolio(periods, 20)
    assert result["transaction_cost_rate_sum"] == pytest.approx(0.02)
    assert result["compounded_total_return_drag"] == pytest.approx(1.1**2 - 1.09**2)
    assert result["compounded_total_return_drag"] != pytest.approx(result["transaction_cost_rate_sum"])


def test_gen2_does_not_read_2026_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    dates = pd.to_datetime(["2025-11-28"])
    frame = pd.DataFrame(
        {
            "date": dates, "symbol": ["1"], "label_end_date_5d": dates,
            "label_end_date_20d": dates, "membership_snapshot_date": dates,
            "available_date": dates, "industry_effective_date": dates, "industry": ["A"],
            "eligible": [True], "in_universe": [True], "future_return_1d": [0.0],
            "future_return_5d": [0.0], "future_return_20d": [0.0], "feature": [1.0],
        }
    )
    monkeypatch.setattr("stockpilot.research_challenger.gen02_correctness.verify_dataset_manifest", lambda _: {})
    def fake_read(*args, **kwargs):
        captured.update(kwargs)
        return frame.copy()
    monkeypatch.setattr(pd, "read_parquet", fake_read)
    from dataclasses import replace
    from stockpilot.research_challenger.config import ChallengerSettings
    base = replace(ChallengerSettings(), factor_columns=("feature",))
    _, evidence = load_maturity_safe_development_dataset(base)
    assert ("label_end_date_20d", "<", pd.Timestamp("2026-01-01")) in captured["filters"]
    assert evidence["2026_labels_read"] is False


def test_holdout_false_fails_before_2026_label_read(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="UNTOUCHED_2026_HOLDOUT_FALSE"):
        open_holdout_once(tmp_path, False)
    assert not any(tmp_path.iterdir())


def test_v6_v30_v30r1_v1r4_hashes_unchanged() -> None:
    lock = json.loads(Path("artifacts/research_challenger/gen02/experiments/004_recursive_final_manifest_fix/development_plan.lock.json").read_text(encoding="utf-8"))
    for path, expected in lock["files"].items():
        assert Path(path).is_file()
        import hashlib
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
