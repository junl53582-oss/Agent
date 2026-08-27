import unittest

import numpy as np
import pandas as pd

from research_v10.features import V10_FEATURES
from research_v12.config import V12Settings
from research_v12.features import _sector_benchmark_return
from research_v12.model import PortfolioRanker, mature_embargoed_training, portfolio_relevance
from research_v12.portfolio import apply_exposure, optimize_v12
from research_v12.risk import risk_budget_exposure


class ResearchV12Tests(unittest.TestCase):
    def test_sector_benchmark_is_official_weighted_and_date_local(self):
        frame = pd.DataFrame({
            "date": ["2020-01-01"] * 3,
            "broad_sector": ["technology"] * 2 + ["consumer"],
            "eligible": True,
            "future_return_20": [0.10, 0.00, 0.05],
            "benchmark_weight": [0.30, 0.10, 0.60],
        })
        result = _sector_benchmark_return(frame)
        self.assertAlmostEqual(result.iloc[0], 0.075)
        self.assertAlmostEqual(result.iloc[1], 0.075)
        self.assertAlmostEqual(result.iloc[2], 0.05)

    def test_embargo_excludes_near_boundary_mature_labels(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2019-10-01", "2019-11-20", "2019-12-01"]),
            "label_end_date_20": pd.to_datetime(["2019-11-01", "2019-12-10", "2019-12-20"]),
            "v12_net_marginal_target": [0.1, 0.2, 0.3],
            "eligible": True,
            "broad_sector": "consumer",
            "symbol": ["1", "2", "3"],
        })
        result = mature_embargoed_training(frame, 2020, 2012, 28)
        self.assertEqual(list(result["symbol"]), ["1"])

    def test_relevance_is_computed_within_sector(self):
        frame = pd.DataFrame({
            "date": ["2020-01-01"] * 20,
            "broad_sector": ["technology"] * 10 + ["consumer"] * 10,
            "v12_net_marginal_target": list(range(10)) + list(range(10)),
        })
        labels = portfolio_relevance(frame)
        for indexes in frame.groupby("broad_sector").groups.values():
            self.assertEqual(int((labels.loc[indexes] == 4).sum()), 1)
            self.assertEqual(int((labels.loc[indexes] == 0).sum()), 4)

    def test_failed_gate_is_benchmark_only(self):
        frame = pd.DataFrame({
            "symbol": [f"{i:06d}" for i in range(20)],
            "benchmark_weight": 0.05, "eligible": True,
            "portfolio_score": np.arange(20), "broad_sector": "consumer",
            "volatility_60": 0.02,
        })
        desired, active, diagnostics = optimize_v12(frame, set(), False, False)
        self.assertAlmostEqual(sum(desired.values()), 1.0)
        self.assertFalse(active)
        self.assertEqual(diagnostics["active_budget"], 0.0)

    def test_risk_budget_is_continuous_and_never_reads_future(self):
        settings = V12Settings()
        frame = pd.DataFrame({
            "market_volatility_60": [0.24], "v12_market_momentum_60": [-0.01],
            "future_return_20": [-999],
        })
        exposure, diagnostics = risk_budget_exposure(frame, settings)
        self.assertAlmostEqual(exposure, 0.5)
        frame["future_return_20"] = 999
        self.assertEqual(exposure, risk_budget_exposure(frame, settings)[0])
        self.assertEqual(diagnostics["risk_regime"], "risk_budget")
        self.assertAlmostEqual(sum(apply_exposure({"a": 0.6, "b": 0.4}, exposure).values()), 0.5)

    def test_positive_momentum_keeps_full_exposure(self):
        frame = pd.DataFrame({"market_volatility_60": [0.50], "v12_market_momentum_60": [0.01]})
        self.assertEqual(risk_budget_exposure(frame)[0], 1.0)

    def test_grouped_ranker_smoke(self):
        generator = np.random.default_rng(43)
        rows = 1200
        frame = pd.DataFrame(generator.normal(size=(rows, len(V10_FEATURES))), columns=V10_FEATURES)
        frame["date"] = np.repeat(pd.date_range("2018-01-01", periods=20), 60)
        frame["broad_sector"] = np.tile(np.repeat(["technology", "consumer"], 30), 20)
        frame["symbol"] = [f"{i % 60:06d}" for i in range(rows)]
        frame["v12_net_marginal_target"] = generator.normal(size=rows)
        model = PortfolioRanker().fit(frame)
        prediction = model.predict(frame.iloc[:60])
        self.assertEqual(len(prediction), 60)
        self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
