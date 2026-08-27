import tempfile
import unittest
from pathlib import Path

from stockpilot.backtest import run_walk_forward
from stockpilot.config import Settings
from stockpilot.data import make_demo_panel


class BacktestTests(unittest.TestCase):
    def test_walk_forward_creates_all_artifacts(self):
        panel = make_demo_panel(symbols=12, periods=460)
        settings = Settings(
            min_train_days=180,
            train_window_days=360,
            top_n=3,
            retrain_every=20,
        )
        result = run_walk_forward(panel, settings)
        self.assertGreater(len(result.equity), 10)
        self.assertEqual(result.latest_signals.shape[0], 3)
        self.assertIn("max_drawdown", result.metrics)
        self.assertIn("signal_execution_rate", result.metrics)
        self.assertEqual(result.metrics["model"], "ridge")
        self.assertIn("executed", result.signals.columns)
        self.assertIn("excess_return", result.yearly.columns)
        self.assertLessEqual(result.metrics["max_drawdown"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            result.save(tmp)
            self.assertTrue((Path(tmp) / "summary.json").exists())
            self.assertTrue((Path(tmp) / "latest_signals.csv").exists())
            self.assertTrue((Path(tmp) / "yearly.csv").exists())


if __name__ == "__main__":
    unittest.main()
