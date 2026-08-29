import unittest
from dataclasses import asdict

import numpy as np
import pandas as pd

from research_v3.features import FUNDAMENTAL_RAW
from research_v10.fundamentals import EXTRA_COLUMNS
from research_v15.features import build_v15_dataset
from research_v20.config import V20Settings
from research_v20r1.config import V20R1Settings
from stockpilot.data import make_demo_panel


class MemoryRepairTests(unittest.TestCase):
    def test_copy_on_write_feature_and_label_values_are_exactly_equal(self):
        panel = make_demo_panel(symbols=30, periods=320, end="2020-03-31")
        symbols = panel["symbol"].astype(int)
        panel["industry"] = np.where(symbols % 2 == 0, "电子", "银行")
        panel["benchmark_weight"] = 1 / 30
        panel["in_universe"] = True
        panel["available_date"] = panel["date"].dt.to_period("Q").dt.start_time
        panel["fundamental_age_days"] = (panel["date"] - panel["available_date"]).dt.days
        for index, column in enumerate([*FUNDAMENTAL_RAW, *EXTRA_COLUMNS, "book_value_per_share", "earnings_per_share"]):
            panel[column] = 1.0 + (symbols % 17) * 0.01 + index * 0.1 + panel["available_date"].dt.quarter * 0.01
        before = panel.copy()
        with pd.option_context("mode.copy_on_write", False):
            expected = build_v15_dataset(panel)
        with pd.option_context("mode.copy_on_write", True):
            actual = build_v15_dataset(panel)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        pd.testing.assert_frame_equal(panel, before, check_exact=True)
        self.assertEqual(len(actual), len(panel))

    def test_reset_index_avoids_copy_but_mutation_remains_isolated(self):
        with pd.option_context("mode.copy_on_write", True):
            frame = pd.DataFrame({"x": np.arange(1000, dtype=np.float64)})
            result = frame.reset_index(drop=True)
            self.assertTrue(np.shares_memory(frame["x"].to_numpy(), result["x"].to_numpy()))
            result.loc[0, "x"] = 999
            self.assertEqual(frame.loc[0, "x"], 0)

    def test_no_model_parameter_changes(self):
        parent, revised = asdict(V20Settings()), asdict(V20R1Settings())
        parent.pop("artifact_dir")
        revised.pop("artifact_dir")
        self.assertEqual(parent, revised)


if __name__ == "__main__":
    unittest.main()
