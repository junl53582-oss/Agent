import unittest

import numpy as np
import pandas as pd

from research_v10.features import broad_sector_v10, technology_subsector
from research_v10.model import mature_training
from research_v10.portfolio import optimize_benchmark_relative
from research_v10.research_config import V10Settings


class ResearchV10ModelTests(unittest.TestCase):
    def test_old_and_new_technology_industry_names_are_recognized(self):
        self.assertEqual(broad_sector_v10("电子元器件"), "technology")
        self.assertEqual(broad_sector_v10("信息服务-软件开发"), "technology")
        self.assertEqual(technology_subsector("半导体材料"), "semiconductor_components")
        self.assertEqual(technology_subsector("计算机软件"), "software_computing")

    def test_twenty_day_training_requires_mature_label(self):
        dataset = pd.DataFrame(
            {
                "date": pd.to_datetime(["2019-12-01", "2019-12-20", "2020-01-02"]),
                "label_end_date_20": pd.to_datetime(
                    ["2019-12-30", "2020-01-20", "2020-02-01"]
                ),
                "v10_target_20": [0.1, 999.0, 999.0],
                "eligible": True,
                "symbol": ["000001", "000002", "000003"],
            }
        )
        result = mature_training(
            dataset, 2020, "v10_target_20", "label_end_date_20", 2012
        )
        self.assertEqual(list(result["symbol"]), ["000001"])

    def test_portfolio_is_sector_neutral_and_stock_capped(self):
        size = 60
        sectors = ["technology"] * 20 + ["consumer"] * 20 + ["finance_real_estate"] * 20
        current = pd.DataFrame(
            {
                "symbol": [f"{index:06d}" for index in range(size)],
                "benchmark_weight": 1 / size,
                "eligible": True,
                "portfolio_score": np.linspace(-1, 1, size),
                "broad_sector": sectors,
                "volatility_60": 0.02,
            }
        )
        desired, active, diagnostics = optimize_benchmark_relative(
            current, set(), 1.0, True, V10Settings()
        )
        self.assertTrue(active)
        self.assertAlmostEqual(sum(desired.values()), 1.0)
        self.assertLessEqual(
            diagnostics["maximum_stock_active_weight"],
            V10Settings().maximum_stock_active_weight + 1e-12,
        )
        self.assertLessEqual(diagnostics["maximum_sector_deviation"], 1e-12)
        frame = current.set_index("symbol")
        desired_series = pd.Series(desired)
        for sector, indexes in frame.groupby("broad_sector").groups.items():
            self.assertAlmostEqual(
                float(desired_series.loc[indexes].sum()),
                float(frame.loc[indexes, "benchmark_weight"].sum()),
            )

    def test_negative_technology_gate_removes_active_technology(self):
        current = pd.DataFrame(
            {
                "symbol": [f"{index:06d}" for index in range(40)],
                "benchmark_weight": 1 / 40,
                "eligible": True,
                "portfolio_score": np.arange(40, dtype=float),
                "broad_sector": ["technology"] * 20 + ["consumer"] * 20,
                "volatility_60": 0.02,
            }
        )
        _, active, _ = optimize_benchmark_relative(
            current, set(), 1.0, False, V10Settings()
        )
        sectors = current.set_index("symbol").loc[list(active), "broad_sector"]
        self.assertTrue((sectors != "technology").all())


if __name__ == "__main__":
    unittest.main()

