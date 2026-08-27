import unittest

import pandas as pd

from stockpilot.portfolio import portfolio_weights, select_with_buffer_and_cap, turnover


class PortfolioTests(unittest.TestCase):
    def test_buffer_keeps_existing_holding_and_group_cap_applies(self):
        frame = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D", "E"],
                "score": [5, 4, 3, 2, 1],
                "pred_rank": [1, 2, 3, 4, 5],
                "board": ["主板", "主板", "主板", "创业板", "科创板"],
                "volatility_20": [0.1, 0.2, 0.3, 0.1, 0.1],
            }
        )
        selected = select_with_buffer_and_cap(
            frame, top_n=3, previous_symbols={"D"}, hold_buffer=1, industry_cap=0.5
        )
        self.assertIn("D", set(selected["symbol"]))
        self.assertLessEqual(int((selected["board"] == "主板").sum()), 1)

    def test_inverse_volatility_and_turnover(self):
        frame = pd.DataFrame({"volatility_20": [0.1, 0.2]}, index=[3, 4])
        weights = portfolio_weights(frame, "inverse_volatility")
        self.assertAlmostEqual(float(weights.sum()), 1)
        self.assertGreater(weights.loc[3], weights.loc[4])
        buys, sells = turnover({"A": 0.5, "B": 0.5}, {"A": 0.5, "C": 0.5})
        self.assertAlmostEqual(buys, 0.5)
        self.assertAlmostEqual(sells, 0.5)


if __name__ == "__main__":
    unittest.main()
