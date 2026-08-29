import unittest

import pandas as pd

from research_v20r2.ledger import PriceBook
from research_v22r1.schedule import schedule_from_parent


class V22R1Tests(unittest.TestCase):
    def test_mode_column_collision_is_repaired(self):
        dates = pd.bdate_range("2019-12-02", periods=110)
        panel = pd.DataFrame({"date": dates, "symbol": "600001", "open": 100.0, "close": 100.0, "volume": 1.0})
        book = PriceBook(panel)
        parent = []
        signals = dates[:73]
        for signal in signals:
            i = book.index(signal)
            parent.append({"date": signal, "entry_date": book.dates[i + 1], "end_date": book.dates[i + 21], "mode": "v16_control"})
        result = schedule_from_parent(pd.DataFrame(parent), book)
        self.assertEqual(len(result), 73)
        self.assertEqual(result[0], (signals[0], 1, 21))

    def test_rejects_non_parent_mode(self):
        dates = pd.bdate_range("2019-12-02", periods=110)
        panel = pd.DataFrame({"date": dates, "symbol": "600001", "open": 100.0, "close": 100.0, "volume": 1.0})
        with self.assertRaisesRegex(ValueError, "unexpected parent"):
            schedule_from_parent(pd.DataFrame({"date": dates[:73], "entry_date": dates[1:74], "end_date": dates[21:94], "mode": "other"}), PriceBook(panel))


if __name__ == "__main__":
    unittest.main()
