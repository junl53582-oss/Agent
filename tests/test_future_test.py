import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stockpilot.data import make_demo_panel, save_panel
from stockpilot.future_test import freeze_future_test, future_test_status, verify_frozen_inputs


class FutureTestTests(unittest.TestCase):
    def test_freeze_refuses_history_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            market = save_panel(make_demo_panel(symbols=6, periods=320), root / "market.csv")
            membership = root / "membership.csv"
            exposure = root / "exposure.csv"
            selected = root / "selected.json"
            membership.write_text("membership", encoding="utf-8")
            exposure.write_text("exposure", encoding="utf-8")
            selected.write_text('{"model_name":"ridge"}', encoding="utf-8")
            cutoff = make_demo_panel(symbols=6, periods=320)["date"].max()
            manifest = root / "manifest.json"
            with self.assertRaisesRegex(ValueError, "晚于"):
                freeze_future_test(
                    market, membership, exposure, selected, manifest, str(cutoff.date())
                )
            start = str((cutoff + pd.offsets.BDay(1)).date())
            freeze_future_test(market, membership, exposure, selected, manifest, start)
            with self.assertRaises(FileExistsError):
                freeze_future_test(market, membership, exposure, selected, manifest, start)
            status = future_test_status(manifest, market)
            self.assertFalse(status["ready_for_evaluation"])
            self.assertEqual(status["observed_trading_days"], 0)
            self.assertFalse(
                json.loads(manifest.read_text(encoding="utf-8"))["execution_authorized"]
            )
            self.assertTrue(status["frozen_inputs_intact"])

            bars = root / "bars"
            signals = root / "signals"
            bars.mkdir()
            signals.mkdir()
            pd.DataFrame({"date": [start]}).to_csv(bars / f"{start}.csv", index=False)
            pd.DataFrame({"date": [start]}).to_csv(signals / f"{start}.csv", index=False)
            status = future_test_status(manifest, market, bars, signals)
            self.assertEqual(status["observed_trading_days"], 1)
            self.assertEqual(status["signal_snapshots"], 1)

            exposure.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exposure"):
                verify_frozen_inputs(manifest)


if __name__ == "__main__":
    unittest.main()
