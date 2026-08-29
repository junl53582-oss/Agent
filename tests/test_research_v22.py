import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from research_v20r2.ledger import PriceBook
from research_v22.config import V22Settings
from research_v22.replay import attach_volatility, compare_control, load_scores, portfolio_input, run_replay


def panel():
    dates = pd.bdate_range("2019-12-02", "2020-03-31")
    return pd.DataFrame([{"date": d, "symbol": s, "open": 100.0, "close": 100.0, "volume": 1.0}
                         for d in dates for s in ("600001", "600002")])


class V22Tests(unittest.TestCase):
    def test_portfolio_input_excludes_all_targets(self):
        frame = pd.DataFrame({"symbol": ["600001"], "eligible": [True], "broad_sector": ["technology"],
                              "benchmark_weight": [1.0], "volatility_60": [.2], "global_model_score": [.3],
                              "label_5": [99], "v10_target_20": [-99], "future_return_20": [8]})
        result = portfolio_input(frame, "global_model_score")
        self.assertEqual(set(result), {"symbol", "eligible", "broad_sector", "benchmark_weight", "volatility_60", "portfolio_score"})

    def test_target_changes_do_not_change_optimizer_input(self):
        frame = pd.DataFrame({"symbol": ["600001", "600002"], "eligible": [True, True], "broad_sector": ["technology"] * 2,
                              "benchmark_weight": [.5, .5], "volatility_60": [.2, .3], "global_model_score": [.1, .2],
                              "label_5": [1, 2], "v10_target_20": [3, 4]})
        changed = frame.assign(label_5=[-100, 100], v10_target_20=[100, -100])
        pd.testing.assert_frame_equal(portfolio_input(frame, "global_model_score"), portfolio_input(changed, "global_model_score"))

    def test_score_loader_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            data = pd.DataFrame({"date": ["2020-01-02"] * 2, "symbol": ["1"] * 2, "eligible": [True] * 2,
                                 "broad_sector": ["technology"] * 2, "benchmark_weight": [.5] * 2, "label_5": [1] * 2,
                                 "v10_target_20": [1] * 2, "v16_score": [1] * 2, "global_model_score": [1] * 2})
            data.to_csv(path / "scores_2020.csv", index=False)
            with self.assertRaisesRegex(ValueError, "invalid frozen"):
                load_scores(path, (2020,))

    def test_volatility_join_is_one_to_one(self):
        scores = pd.DataFrame({"date": pd.to_datetime(["2020-01-02"]), "symbol": ["600001"]})
        data = pd.DataFrame({"date": pd.to_datetime(["2020-01-02"]), "symbol": ["600001"], "volatility_60": [.2]})
        self.assertEqual(attach_volatility(scores, data).volatility_60.iloc[0], .2)

    def test_control_comparator_detects_numeric_change(self):
        equity = pd.DataFrame({"date": ["2020-01-02"], "mode": ["v16_replay"], "period_return": [0.1], "benchmark_return": [0.0], "nav": [1.1], "buy_turnover": [1.0], "sell_turnover": [0.0], "transaction_cost": [0.0]})
        holdings = pd.DataFrame({"date": ["2020-01-02"], "mode": ["v16_replay"], "symbol": ["600001"], "units": [.01], "weight": [1.0], "target_weight": [1.0]})
        daily = pd.DataFrame({"date": ["2020-01-03"], "mode": ["v16_replay"], "point": ["after_rebalance"], "nav": [1.0]})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            equity.assign(mode="v16_control", period_return=.2).to_csv(root / "equity.csv", index=False)
            holdings.assign(mode="v16_control").to_csv(root / "holdings.csv", index=False)
            daily.assign(mode="v16_control").to_csv(root / "daily_nav.csv", index=False)
            (root / "settlements.json").write_text(json.dumps({"events": []}))
            with self.assertRaisesRegex(AssertionError, "period_return"):
                compare_control(equity, holdings, daily, [], root)

    def test_replay_execution_ignores_target_values(self):
        data = panel()
        book = PriceBook(data)
        signal = pd.Timestamp("2020-01-02")
        scores = pd.DataFrame({"date": [signal] * 2, "symbol": ["600001", "600002"], "eligible": [True] * 2,
                               "broad_sector": ["technology"] * 2, "benchmark_weight": [.5, .5], "volatility_60": [.2, .3],
                               "v16_score": [.1, .2], "global_model_score": [.2, .1], "label_5": [.1, .2], "v10_target_20": [.2, .1]})
        history = pd.DataFrame({"snapshot_date": pd.to_datetime(["2019-12-31"] * 2), "symbol": ["600001", "600002"], "weight": [.5, .5]})
        schedule = [(signal, book.index("2020-01-03"), book.index("2020-01-31"))]
        settings = V22Settings(test_years=(2020,))
        fake = ({"600001": .5, "600002": .5}, set(), {})
        with patch("research_v22.replay.optimize_v16", return_value=fake):
            one = run_replay(scores, book, history, schedule, settings)[1]
            changed = scores.assign(label_5=[100, -100], v10_target_20=[-100, 100])
            two = run_replay(changed, PriceBook(data), history, schedule, settings)[1]
        pd.testing.assert_frame_equal(one, two)


if __name__ == "__main__":
    unittest.main()
