import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research_v9.backtest import _desired_weights, _execute_weights
from research_v9.config import V9Settings
from research_v9.data import attach_membership_weight
from research_v9.features import FUNDAMENTAL_RAW, V9_FEATURES, _fundamental_changes, _residual_target
from research_v9.model import V9Models, mature_training, score_v9


class ColumnModel:
    def __init__(self, column, direction=1.0):
        self.column = column
        self.direction = direction

    def predict(self, frame):
        return frame[self.column].to_numpy(dtype=float) * self.direction


class ResearchV9Tests(unittest.TestCase):
    def test_membership_weights_never_backfill_before_snapshot(self):
        panel = pd.DataFrame(
            {
                "date": pd.to_datetime(["2014-12-31", "2015-01-30", "2015-01-30"]),
                "symbol": ["000001", "000001", "000002"],
            }
        )
        history = pd.DataFrame(
            {
                "snapshot_date": pd.to_datetime(["2015-01-30", "2015-01-30"]),
                "symbol": ["000001", "000002"],
                "weight": [60.0, 40.0],
            }
        )
        result = attach_membership_weight(panel, history)
        self.assertEqual(result.loc[0, "benchmark_weight"], 0.0)
        self.assertAlmostEqual(result.loc[1, "benchmark_weight"], 0.6)
        self.assertAlmostEqual(result.loc[2, "benchmark_weight"], 0.4)

    def test_fundamental_changes_follow_announcement_sequence(self):
        rows = []
        for available, roe in [("2024-04-30", 10.0), ("2024-08-30", 15.0)]:
            row = {"symbol": "000001", "available_date": pd.Timestamp(available)}
            row.update({column: roe for column in FUNDAMENTAL_RAW})
            rows.append(row)
        result = _fundamental_changes(pd.DataFrame(rows))
        self.assertTrue(pd.isna(result.loc[0, "roe_change"]))
        self.assertAlmostEqual(result.loc[1, "roe_change"], 5 / 11)

    def test_residual_target_removes_locked_styles(self):
        size = 40
        style = np.linspace(-0.5, 0.5, size)
        data = pd.DataFrame(
            {
                "date": pd.Timestamp("2024-01-02"),
                "eligible": True,
                "label_5": 0.03 + 0.7 * style,
                "benchmark_weight_rank": style,
                "momentum": np.sin(np.arange(size)),
                "low_volatility": np.cos(np.arange(size)),
                "industry": ["电子"] * 20 + ["银行"] * 20,
            }
        )
        residual = _residual_target(data)
        self.assertLess(float(residual.abs().max()), 1e-10)

    def test_training_excludes_unmatured_labels(self):
        dataset = pd.DataFrame(
            {
                "date": pd.to_datetime(["2019-12-20", "2019-12-30", "2020-01-02"]),
                "label_end_date_5": pd.to_datetime(["2019-12-30", "2020-01-08", "2020-01-09"]),
                "v9_target_5": [0.1, 999.0, 999.0],
                "eligible": True,
                "symbol": ["000001", "000002", "000003"],
            }
        )
        train = mature_training(dataset, 2020, V9Settings())
        self.assertEqual(list(train["symbol"]), ["000001"])

    def test_core_and_active_budgets_sum_to_one(self):
        size = 40
        current = pd.DataFrame(
            {
                "symbol": [f"{index:06d}" for index in range(size)],
                "benchmark_weight": 1 / size,
                "eligible": True,
                "portfolio_score": np.arange(size, dtype=float),
                "broad_sector": ["technology"] * size,
                "volatility_20": 0.02,
            }
        )
        desired, active = _desired_weights(current, set(), V9Settings())
        self.assertEqual(len(active), 30)
        self.assertAlmostEqual(sum(desired.values()), 1.0)
        nonactive = set(current["symbol"]) - active
        self.assertAlmostEqual(sum(desired[symbol] for symbol in nonactive), 0.75 * 10 / 40)

    def test_executed_returns_align_by_symbol(self):
        current = pd.DataFrame(
            {
                "symbol": ["000001", "000002"],
                "entry_tradable": [True, True],
                "execution_return": [0.10, -0.05],
                "entry_open": [10.0, 20.0],
                "execution_exit_open": [11.0, 19.0],
            },
            index=[100, 200],
        )
        executed, realized = _execute_weights(
            current, {"000001": 0.6, "000002": 0.4}, {}
        )
        gross = sum(weight * realized[symbol] for symbol, weight in executed.items())
        self.assertAlmostEqual(gross, 0.04)

    def test_fixed_ridge_lightgbm_and_v6_blend(self):
        current = pd.DataFrame(
            {
                "symbol": ["000001", "000002", "000003", "000004"],
                "broad_sector": ["consumer"] * 4,
                "score": 0.0,
            }
        )
        for feature in V9_FEATURES:
            current[feature] = np.linspace(-1, 1, 4)
        models = V9Models(
            ridge=ColumnModel(V9_FEATURES[0], 1),
            lightgbm=ColumnModel(V9_FEATURES[0], -1),
            technology=ColumnModel(V9_FEATURES[0], 1),
            technology_fallback=True,
            training_rows=10000,
            training_end=pd.Timestamp("2019-12-31"),
        )
        with patch("research_v9.model.score_v6", return_value=current.copy()):
            result = score_v9(current, models, object(), [], V9Settings())
        expected = 0.6 * result["ridge_score"] + 0.4 * result["nonlinear_score"]
        np.testing.assert_allclose(result["enhanced_global"], expected)
        np.testing.assert_allclose(result["nonlinear_model_score"], 0.3 * expected)


if __name__ == "__main__":
    unittest.main()
