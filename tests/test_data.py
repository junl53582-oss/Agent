import unittest

from stockpilot.data import make_demo_panel, validate_panel


class DataTests(unittest.TestCase):
    def test_demo_panel_is_valid_and_deterministic(self):
        first = make_demo_panel(symbols=6, periods=320, seed=7)
        second = make_demo_panel(symbols=6, periods=320, seed=7)
        self.assertEqual(len(first), 6 * 320)
        self.assertEqual(first["symbol"].nunique(), 6)
        self.assertAlmostEqual(first["close"].iloc[-1], second["close"].iloc[-1])
        self.assertTrue((first[["open", "high", "low", "close"]] > 0).all().all())
        self.assertEqual(len(validate_panel(first)), len(first))


if __name__ == "__main__":
    unittest.main()
