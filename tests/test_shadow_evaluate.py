import unittest

import pandas as pd

from stockpilot.config import Settings
from stockpilot.shadow import is_signal_due
from stockpilot.shadow_evaluate import _period_accounting


class ShadowEvaluationTests(unittest.TestCase):
    def test_signal_cadence_starts_on_first_observation(self):
        due = [day for day in range(1, 13) if is_signal_due(day, 5)]
        self.assertEqual(due, [1, 6, 11])
        self.assertFalse(is_signal_due(0, 5))

    def test_continuing_holding_does_not_need_reentry(self):
        signals = pd.DataFrame(
            {
                "symbol": ["000001", "000002"],
                "weight": [0.5, 0.5],
            }
        )
        current = pd.DataFrame(
            {
                "symbol": ["000001", "000002", "000003"],
                "entry_date": pd.to_datetime(["2026-08-25"] * 3),
                "execution_exit_date": pd.to_datetime(["2026-09-01"] * 3),
                "entry_tradable": [False, False, True],
                "entry_limit_up": [True, True, False],
                "entry_open": [10.0, 20.0, 30.0],
                "execution_exit_open": [11.0, 22.0, 30.0],
                "execution_return": [float("nan"), float("nan"), 0.0],
                "future_return": [0.1, 0.1, 0.0],
            }
        )
        result, weights, details = _period_accounting(
            signals,
            current,
            {"000001": 0.5},
            Settings(),
        )
        self.assertEqual(weights, {"000001": 0.5})
        self.assertAlmostEqual(result["gross_return"], 0.05)
        self.assertEqual(result["blocked_entries"], 1)
        self.assertTrue(details[0]["executed"])
        self.assertFalse(details[1]["executed"])


if __name__ == "__main__":
    unittest.main()
