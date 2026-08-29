import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd

from research_v16.config import V16Settings
from research_v28.evaluation import active_drawdown, conversion_gate
from research_v28.model import confidence_from_validation, tail_labels
from research_v28.replay import confidence_optimize, portfolio_input


class V28Tests(unittest.TestCase):
    def test_tail_labels_are_date_local_top_twenty_percent(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"] * 10 + ["2020-01-02"] * 10),
                              "target": list(range(10)) + list(range(10, 0, -1))})
        labels = tail_labels(frame, "target", 0.8)
        self.assertEqual(int(labels[:10].sum()), 2)
        self.assertEqual(int(labels[10:].sum()), 2)
        self.assertTrue(labels[8] and labels[9] and labels[10] and labels[11])

    def test_crossfit_confidence_has_fixed_tiers(self):
        positive = {"ic5": .01, "ic20": .02, "spread5": .001, "spread20": .002}
        mixed = {"ic5": .01, "ic20": .02, "spread5": -.0005, "spread20": .002}
        negative = {key: -value for key, value in positive.items()}
        self.assertEqual(confidence_from_validation([positive, positive]), 1.0)
        self.assertEqual(confidence_from_validation([positive, mixed]), 0.5)
        self.assertEqual(confidence_from_validation([mixed, mixed]), 0.25)
        self.assertEqual(confidence_from_validation([negative, negative]), 0.0)

    def test_portfolio_input_excludes_targets_and_control_confidence_is_one(self):
        frame = pd.DataFrame({"symbol": ["1"], "eligible": [True], "broad_sector": ["technology"],
                              "benchmark_weight": [1.0], "volatility_60": [.2], "v16_score": [.1],
                              "v28_score": [.2], "model_confidence": [.25], "label_5": [999]})
        control = portfolio_input(frame, "v16_score")
        candidate = portfolio_input(frame, "v28_score")
        self.assertNotIn("label_5", control)
        self.assertEqual(control.model_confidence.iloc[0], 1.0)
        self.assertEqual(candidate.model_confidence.iloc[0], .25)

    def test_confidence_scales_only_active_budget(self):
        current = pd.DataFrame({"model_confidence": [.25, .25], "portfolio_score": [1.0, 0.0]})
        captured = {}
        def fake(frame, previous, enabled, technology, settings):
            captured["budget"] = settings.maximum_active_budget
            captured["columns"] = list(frame)
            return {}, set(), {}
        with patch("research_v28.replay.base_optimize", side_effect=fake):
            _, _, diagnostics = confidence_optimize(current, set(), True, True, V16Settings())
        self.assertAlmostEqual(captured["budget"], .15 * .25)
        self.assertNotIn("model_confidence", captured["columns"])
        self.assertEqual(diagnostics["model_confidence"], .25)

    def test_active_drawdown_uses_strategy_over_benchmark(self):
        frame = pd.DataFrame({"period_return": [.10, -.10, .05], "benchmark_return": [.10, 0, 0]})
        self.assertAlmostEqual(active_drawdown(frame), -0.10)

    def test_conversion_gate_does_not_use_absolute_strategy_drawdown(self):
        candidate = pd.DataFrame({"period_return": [-.20, .30], "benchmark_return": [-.20, .20],
                                  "test_year": [2020, 2021], "in_market": [True, True],
                                  "rank_ic_5": [.1, .1], "rank_ic_20": [.1, .1], "technology_rank_ic_5": [.1, .1],
                                  "buy_turnover": [.01, .01], "sell_turnover": [.01, .01], "transaction_cost": [.001, .001]})
        control = candidate.assign(buy_turnover=.02, sell_turnover=.02, transaction_cost=.002)
        result = conversion_gate(candidate, control)
        self.assertIn("active_drawdown_not_below_minus_10pct", result["checks"])
        self.assertNotIn("max_drawdown_not_below_minus_18pct", result["checks"])


if __name__ == "__main__":
    unittest.main()
