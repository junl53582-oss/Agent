import unittest

import numpy as np
import pandas as pd

from research_v11.config import V11Settings
from research_v10.features import V10_FEATURES
from research_v11.model import TailRanker, _validation_top_tail, tail_relevance
from research_v11.portfolio import apply_exposure, optimize_v11
from research_v11.risk import defensive_exposure


class ResearchV11Tests(unittest.TestCase):
    def test_tail_relevance_is_top_heavy_and_daily(self):
        frame = pd.DataFrame(
            {
                "date": ["2020-01-01"] * 10 + ["2020-01-02"] * 10,
                "v10_target_20": list(range(10)) + list(range(10, 0, -1)),
            }
        )
        labels = tail_relevance(frame)
        for indexes in frame.groupby("date").groups.values():
            self.assertEqual(int((labels.loc[indexes] == 4).sum()), 1)
            self.assertEqual(int((labels.loc[indexes] == 3).sum()), 1)
            self.assertEqual(int((labels.loc[indexes] == 0).sum()), 4)

    def test_top_tail_validation_uses_forward_returns_not_residual_as_payoff(self):
        frame = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-01-01")] * 10,
                "symbol": [f"{i:06d}" for i in range(10)],
                "eligible": True,
                "future_return_20": np.arange(10) / 100,
                "v10_target_20": np.arange(10),
                "benchmark_weight": 0.1,
                "broad_sector": "consumer",
            }
        )
        excess, precision = _validation_top_tail(frame, np.arange(10), 2)
        self.assertAlmostEqual(excess, 0.04)
        self.assertEqual(precision, 1.0)

    def test_failed_global_gate_returns_benchmark_without_active_names(self):
        frame = pd.DataFrame(
            {
                "symbol": [f"{i:06d}" for i in range(20)],
                "benchmark_weight": 0.05,
                "eligible": True,
                "portfolio_score": np.arange(20),
                "broad_sector": "consumer",
                "volatility_60": 0.02,
            }
        )
        desired, active, diagnostics = optimize_v11(frame, set(), False, False)
        self.assertAlmostEqual(sum(desired.values()), 1.0)
        self.assertFalse(active)
        self.assertEqual(diagnostics["active_budget"], 0.0)

    def test_defensive_regime_and_cash_overlay(self):
        frame = pd.DataFrame(
            {
                "in_universe": True,
                "benchmark_weight": [0.5, 0.5],
                "momentum_60": [-0.10, -0.05],
                "ret_20": [-0.02, -0.01],
            }
        )
        exposure, diagnostics = defensive_exposure(frame, V11Settings())
        self.assertEqual(exposure, 0.55)
        self.assertEqual(diagnostics["risk_regime"], "risk_off")
        scaled = apply_exposure({"000001": 0.6, "000002": 0.4}, exposure)
        self.assertAlmostEqual(sum(scaled.values()), 0.55)

    def test_defense_uses_only_same_date_observable_columns(self):
        frame = pd.DataFrame(
            {
                "in_universe": True,
                "benchmark_weight": [0.4, 0.6],
                "momentum_60": [0.01, 0.02],
                "ret_20": [0.01, -0.01],
                "future_return_20": [-999, -999],
            }
        )
        first = defensive_exposure(frame)[0]
        frame["future_return_20"] = [999, 999]
        second = defensive_exposure(frame)[0]
        self.assertEqual(first, second)

    def test_lambdarank_grouped_fit_smoke(self):
        generator = np.random.default_rng(42)
        rows = 1200
        frame = pd.DataFrame(
            generator.normal(size=(rows, len(V10_FEATURES))), columns=V10_FEATURES
        )
        frame["date"] = np.repeat(pd.date_range("2018-01-01", periods=40), 30)
        frame["symbol"] = [f"{index % 30:06d}" for index in range(rows)]
        frame["v10_target_20"] = generator.normal(size=rows)
        model = TailRanker().fit(frame)
        prediction = model.predict(frame.iloc[:30])
        self.assertEqual(len(prediction), 30)
        self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
