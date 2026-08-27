import unittest

import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

from research_v10.audit import _period_core
from research_v10.data import normalize_cached_market


class ResearchV10Tests(unittest.TestCase):
    def test_tencent_amount_is_normalized_to_volume(self):
        with TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            raw.mkdir()
            pd.DataFrame(
                {
                    "date": ["2020-01-02"],
                    "open": [10.0],
                    "close": [12.0],
                    "high": [12.0],
                    "low": [10.0],
                    "amount": [1000.0],
                    "symbol": ["000001"],
                }
            ).to_csv(raw / "000001_2015-01-01_2026-08-21_auto.csv", index=False)
            output = Path(directory) / "market.csv"
            panel, report = normalize_cached_market(raw, output)
            self.assertEqual(panel.loc[0, "volume"], 1000.0)
            self.assertEqual(panel.loc[0, "amount"], 11000.0)
            self.assertEqual(report["tencent_schema_symbols"], 1)

    def test_core_replication_uses_symbol_aligned_returns(self):
        current = pd.DataFrame(
            {
                "symbol": ["000001", "000002"],
                "in_universe": True,
                "benchmark_weight": [0.6, 0.4],
                "future_return": [0.10, -0.05],
                "entry_tradable": True,
                "execution_return": [0.10, -0.05],
                "entry_open": [10.0, 20.0],
                "execution_exit_open": [11.0, 19.0],
            },
            index=[100, 200],
        )
        executed, result = _period_core(current, {}, 0.0, 0.0)
        self.assertAlmostEqual(sum(executed.values()), 1.0)
        self.assertAlmostEqual(result["gross_return"], 0.04)
        self.assertAlmostEqual(result["benchmark_return"], 0.04)
        self.assertAlmostEqual(result["gross_excess"], 0.0)


if __name__ == "__main__":
    unittest.main()
