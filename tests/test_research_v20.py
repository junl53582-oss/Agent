import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from research_v16.config import V16Settings
from research_v16.model import score_v16
from research_v20.backtest import run_backtest
from research_v20.config import V20Settings
from research_v20.freeze import write_new
from research_v20.timing import historical_market_state, weights_for_momentum
from research_status import snapshot_status


def panel():
    dates = pd.bdate_range("2019-11-01", "2020-03-31")
    rows = []
    for symbol, weight in [("000001", 0.6), ("000002", 0.4)]:
        for i, date in enumerate(dates):
            rows.append({
                "date": date, "symbol": symbol, "close": 100 * 1.001 ** i,
                "benchmark_weight": weight, "in_universe": True, "eligible": True,
                "future_return_20": 0.02, "label_5": 0.01 if symbol == "000001" else -0.01,
                "v10_target_20": 0.02 if symbol == "000001" else -0.02,
                "broad_sector": "technology", "entry_open_20": 100.0,
                "execution_exit_open_20": 102.0, "entry_tradable_20": True,
                "execution_return_20": 0.02,
            })
    return pd.DataFrame(rows)


class V20Tests(unittest.TestCase):
    def test_complete_v16_configuration_contract(self):
        inherited = {item.name for item in fields(V16Settings)}
        actual = {item.name for item in fields(V20Settings)}
        self.assertTrue(inherited <= actual)
        self.assertEqual(V20Settings().baseline_share, 0.65)
        self.assertEqual(V20Settings().text_share, 0.35)

    def test_real_score_v16_accepts_v20_settings(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-02"] * 2), "symbol": ["000001", "000002"],
                              "broad_sector": ["technology"] * 2, "global_model_score": [0.1, -0.1]})
        baseline = SimpleNamespace(predict_components=lambda data: (np.array([0.7, 0.3]), np.array([0.05, 0.02])))
        text = SimpleNamespace(recent_scores=lambda data, settings: pd.DataFrame({
            "symbol": data["symbol"], "text_score": [0.2, -0.2], "char_text_score": [0.1, -0.1], "text_events": [2, 1]}))
        models = SimpleNamespace(baseline_model=baseline, text_model=text, v10=None)
        with patch("research_v16.model.score_v10", side_effect=lambda current, *args: current.copy()):
            scored = score_v16(frame, models, None, None, V20Settings())
        np.testing.assert_allclose(scored["v16_score"], 0.65 * scored["v13_comparable_score"] + 0.35 * scored["text_event_score"])

    def test_future_prices_weights_and_labels_cannot_change_past_state(self):
        original = panel()
        cutoff = pd.Timestamp("2020-02-03")
        changed = original.copy()
        mask = changed["date"] > cutoff
        changed.loc[mask, "close"] *= 5
        changed.loc[mask, "benchmark_weight"] = 0.01
        changed["future_return_20"] = -0.99
        pd.testing.assert_frame_equal(historical_market_state(original).loc[:cutoff], historical_market_state(changed).loc[:cutoff])

    def test_market_return_uses_lagged_weights(self):
        data = panel()
        date = pd.Timestamp("2020-02-03")
        baseline = historical_market_state(data)
        data.loc[data["date"].eq(date), "benchmark_weight"] = [0.01, 0.99]
        changed = historical_market_state(data)
        self.assertAlmostEqual(baseline.loc[date, "market_return"], changed.loc[date, "market_return"])

    def test_missing_bars_fail_coverage_not_bridge_gap(self):
        data = panel()
        date = pd.Timestamp("2020-02-03")
        data = data[~(data["date"].eq(date) & data["symbol"].eq("000001"))]
        state = historical_market_state(data)
        self.assertTrue(pd.isna(state.loc[date, "market_momentum"]))
        self.assertTrue(pd.isna(state.loc[pd.Timestamp("2020-02-04"), "market_momentum"]))

    def test_warmup_and_unknown_regime_fail_closed(self):
        state = historical_market_state(panel())
        self.assertTrue(state["market_momentum"].iloc[:20].isna().all())
        self.assertAlmostEqual(state["market_momentum"].iloc[20], 1.001 ** 20 - 1)
        with self.assertRaises(ValueError):
            weights_for_momentum(float("nan"))

    def test_real_backtest_does_not_select_using_forward_returns(self):
        def fit(dataset, corpus, year, settings, cache):
            cache[year] = (None, None, None)
            return None

        def score(current, *args):
            return current.assign(v16_score=[0.1, -0.1], v13_comparable_score=[0.1, -0.1], text_event_score=[0.2, -0.2])

        def optimize(current, *args):
            return {"000001": 0.6, "000002": 0.4}, {"000001", "000002"}, {}

        settings = V20Settings(test_years=(2020,))
        first = panel()
        changed = first.copy()
        changed["future_return_20"] = -0.5
        with patch("research_v20.backtest.fit_v16_models", side_effect=fit), patch("research_v20.backtest.score_v16", side_effect=score), patch("research_v20.backtest.optimize_v16", side_effect=optimize):
            eq1, holdings1 = run_backtest(first, None, settings)
            eq2, holdings2 = run_backtest(changed, None, settings)
        pd.testing.assert_frame_equal(holdings1, holdings2)
        pd.testing.assert_series_equal(eq1["in_market"], eq2["in_market"])
        pd.testing.assert_series_equal(eq1["market_momentum"], eq2["market_momentum"])
        self.assertTrue(eq1["market_data_end"].le(eq1["date"]).all())

    def test_real_backtest_charges_exit_and_reentry(self):
        def fit(dataset, corpus, year, settings, cache):
            cache[year] = (None, None, None)
            return None
        def score(current, *args):
            return current.assign(v16_score=[0.1, -0.1], v13_comparable_score=[0.1, -0.1], text_event_score=[0.1, -0.1])
        data = panel()
        state = historical_market_state(data)
        dates = data.loc[data["date"].dt.year.eq(2020), "date"].drop_duplicates().sort_values().iloc[::20]
        state.loc[dates, "market_momentum"] = [0.03, -0.03, 0.03, 0.03][:len(dates)]
        with patch("research_v20.backtest.fit_v16_models", side_effect=fit), patch("research_v20.backtest.score_v16", side_effect=score), patch("research_v20.backtest.historical_market_state", return_value=state), patch("research_v20.backtest.optimize_v16", return_value=({"000001": 1.0}, {"000001"}, {})):
            equity, _ = run_backtest(data, None, V20Settings(test_years=(2020,)))
        timing = equity[equity["mode"].eq("v20_timing")].reset_index(drop=True)
        self.assertAlmostEqual(timing.loc[1, "sell_turnover"], 1.0)
        self.assertLess(timing.loc[1, "period_return"], 0)
        self.assertAlmostEqual(timing.loc[2, "buy_turnover"], 1.0)

    def test_mismatched_snapshot_never_emits_recommendation(self):
        status = snapshot_status({"latest_prediction_date": "2026-08-21", "market_timing": {"timing_date": "2026-08-28"}}, "2026-08-29")
        self.assertFalse(status["recommendation_enabled"])
        self.assertFalse(status["execution_authorized"])
        self.assertTrue(any("不一致" in reason for reason in status["reasons"]))

    def test_frozen_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report.json"
            write_new(target, {"first": True})
            with self.assertRaises(FileExistsError):
                write_new(target, {"first": False})


if __name__ == "__main__":
    unittest.main()
