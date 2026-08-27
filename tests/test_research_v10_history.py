import unittest

import pandas as pd

from research_v10.history_data import _normalize_hfq


class ResearchV10HistoryTests(unittest.TestCase):
    def test_tencent_hfq_amount_maps_to_volume(self):
        raw = pd.DataFrame(
            {
                "date": ["2010-01-04"],
                "open": [10.0],
                "close": [12.0],
                "high": [12.5],
                "low": [9.5],
                "amount": [1000.0],
            }
        )
        result = _normalize_hfq(raw, "000001", "tencent")
        self.assertEqual(result.loc[0, "volume"], 1000.0)
        self.assertEqual(result.loc[0, "amount"], 11000.0)
        self.assertTrue((result[["open", "high", "low", "close"]] > 0).all().all())


if __name__ == "__main__":
    unittest.main()

