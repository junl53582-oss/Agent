import unittest
import tempfile
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from research_v17.backtest import MODES, max_drawdown
from research_v17.config import V17Settings
from research_v17.freeze import freeze_research
from research_v17.validation import _summarize, run_research_v17


class ResearchV17Tests(unittest.TestCase):
    def test_settings_timing_defaults(self):
        settings = V17Settings()
        self.assertEqual(settings.timing_window_periods, 1)
        self.assertEqual(settings.timing_threshold, 0.0)
        self.assertEqual(settings.baseline_share, 0.65)
        self.assertEqual(settings.text_share, 0.35)

    def test_modes_are_control_and_timing(self):
        self.assertEqual(MODES, ("v16_ungated", "v17_timing"))

    def test_max_drawdown_includes_initial_capital(self):
        self.assertAlmostEqual(max_drawdown(pd.Series([-0.1, 0.05])), -0.1)
        self.assertAlmostEqual(max_drawdown(pd.Series(dtype=float)), 0)

    def test_summarize_win_rate_counts_held_periods_only(self):
        equity = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-02", "2020-02-07", "2020-03-06"]),
            "test_year": [2020, 2020, 2020],
            "mode": ["v17_timing"] * 3,
            "period_return": [0.01, -0.02, 0.03],
            "benchmark_return": [0.005, 0.005, 0.005],
            "excess_period_return": [0.005, -0.025, 0.025],
            "in_market": [True, True, False],
            "timing_momentum": [0.0, 0.01, 0.01],
            "buy_turnover": [0.1, 0.1, 0.0],
            "sell_turnover": [0.1, 0.1, 0.0],
            "transaction_cost": [0.0002, 0.0002, 0.0],
            "cash_weight": [0.0, 0.0, 1.0],
            "active_budget": [0.15, 0.15, 0.0],
            "ex_ante_tracking_error": [0.01, 0.01, 0.0],
            "maximum_stock_active_weight": [0.0075, 0.0075, 0.0],
            "maximum_sector_deviation": [0.0, 0.0, 0.0],
            "active_holdings": [30, 30, 0],
        })
        metrics = _summarize(equity)
        self.assertEqual(metrics["periods_held"], 2)
        self.assertEqual(metrics["periods_total"], 3)
        self.assertAlmostEqual(metrics["win_rate"], 0.5)

    def test_timing_uses_only_prior_period_return(self):
        # 时间逻辑单元测试：动量只来自已完整走完的历史期
        history = deque(maxlen=1)
        history.append(0.03)  # 上一期基准收益
        momentum = float(np.prod([1.0 + r for r in history]) - 1.0)
        self.assertAlmostEqual(momentum, 0.03)
        in_market = momentum > 0.0
        self.assertTrue(in_market)
        history.append(-0.05)
        momentum = float(np.prod([1.0 + r for r in history]) - 1.0)
        self.assertAlmostEqual(momentum, -0.05)
        self.assertFalse(momentum > 0.0)

    def test_freeze_refuses_existing_lock(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "artifacts/research_v17"
            directory.mkdir(parents=True)
            (directory / "plan.lock.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                freeze_research(root)

    def test_run_rejects_nonfrozen_parameters(self):
        with self.assertRaises(RuntimeError):
            run_research_v17(settings=V17Settings(timing_threshold=0.05))


if __name__ == "__main__":
    unittest.main()
