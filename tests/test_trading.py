import unittest

import pandas as pd

from stockpilot.trading import add_execution_columns, board_name, price_limit_rate


class TradingRuleTests(unittest.TestCase):
    def test_board_and_historical_limit_rules(self):
        self.assertEqual(board_name("688001"), "科创板")
        self.assertEqual(board_name("300001"), "创业板")
        self.assertAlmostEqual(price_limit_rate("600001", "*ST示例"), 0.05)
        self.assertAlmostEqual(price_limit_rate("300001", "示例", "2020-01-01"), 0.10)
        self.assertAlmostEqual(price_limit_rate("300001", "示例", "2021-01-01"), 0.20)
        self.assertAlmostEqual(price_limit_rate("688001", "示例"), 0.20)
        self.assertAlmostEqual(price_limit_rate("830001", "示例"), 0.30)

    def test_limit_up_blocks_entry_and_limit_down_defers_exit(self):
        dates = pd.bdate_range("2024-01-02", periods=7)
        frame = pd.DataFrame(
            {
                "date": dates,
                "symbol": "600001",
                "name": "示例股份",
                "open": [10.0, 11.0, 10.0, 9.0, 9.5, 9.6, 9.7],
                "close": [10.0, 10.5, 10.0, 9.2, 9.5, 9.6, 9.7],
                "volume": 1000,
            }
        )
        grouped = frame.groupby("symbol", group_keys=False)
        frame["entry_open"] = grouped["open"].shift(-1)
        frame["exit_open"] = grouped["open"].shift(-3)
        result = add_execution_columns(frame, horizon=2)
        self.assertTrue(bool(result.loc[0, "entry_limit_up"]))
        self.assertFalse(bool(result.loc[0, "entry_tradable"]))
        self.assertTrue(bool(result.loc[0, "exit_limit_down"]))
        self.assertTrue(bool(result.loc[0, "exit_deferred"]))
        self.assertAlmostEqual(result.loc[0, "execution_exit_open"], 9.5)


if __name__ == "__main__":
    unittest.main()
