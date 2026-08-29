import unittest

import pandas as pd

from research_v23.diagnostics import cost_diagnostic, equal_date_tail_spreads, selection_diagnostics, tail_members


class V23Tests(unittest.TestCase):
    def test_tails_are_fixed_ten_percent_with_symbol_tiebreak(self):
        frame = pd.DataFrame({"symbol": [f"{i:06d}" for i in range(20)], "global_model_score": [1.0] * 20})
        result = tail_members(frame)
        self.assertEqual(result[result["tail"].eq("top")].symbol.tolist(), ["000000", "000001"])
        self.assertEqual(result[result["tail"].eq("bottom")].symbol.tolist(), ["000018", "000019"])

    def test_tail_spread_uses_only_target_valid_rows(self):
        scores = pd.DataFrame({"date": pd.to_datetime(["2020-01-02"] * 10), "symbol": [f"{i:06d}" for i in range(10)],
                               "eligible": True, "global_model_score": list(range(10)), "label_5": list(range(10)),
                               "v10_target_20": [None] + list(range(1, 10))})
        result = equal_date_tail_spreads(scores)
        self.assertEqual(result.loc[result.target.eq("label_5"), "top_minus_bottom"].iloc[0], 9.0)
        self.assertTrue(pd.isna(result.loc[result.target.eq("v10_target_20"), "top_minus_bottom"].iloc[0]))

    def test_selection_uses_recorded_targets_without_reoptimizing(self):
        scores = pd.DataFrame({"date": pd.to_datetime(["2020-01-02"] * 2), "symbol": ["600001", "600002"], "eligible": True,
                               "broad_sector": "technology", "benchmark_weight": [.5, .5], "label_5": [.2, -.1], "v10_target_20": [.1, -.2]})
        holdings = pd.DataFrame({"date": ["2020-01-02"] * 2, "symbol": ["600001", "600002"], "mode": "global_only", "target_weight": [.6, .4]})
        result, active = selection_diagnostics(scores, holdings)
        self.assertEqual(active.overweight.tolist(), [True, False])
        self.assertAlmostEqual(result.loc[result.target.eq("label_5"), "overweight_minus_other"].iloc[0], .3)

    def test_cost_delta_is_directly_reported(self):
        equity = pd.DataFrame({"mode": ["v16_replay", "global_only"], "buy_turnover": [.2, .3], "sell_turnover": [.2, .3], "transaction_cost": [.01, .02]})
        result = cost_diagnostic(equity)
        self.assertAlmostEqual(result["global_minus_control"]["average_one_way_turnover"], .1)
        self.assertAlmostEqual(result["global_minus_control"]["average_transaction_cost"], .01)


if __name__ == "__main__":
    unittest.main()
