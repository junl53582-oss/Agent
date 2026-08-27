import unittest

import pandas as pd

from research_v3.features import V3_FEATURES, build_v3_dataset
from research_v3.fundamentals import attach_fundamentals_asof, normalize_fundamentals
from research_v3.validation import nested_year_selection
from stockpilot.data import make_demo_panel


class ResearchV3Tests(unittest.TestCase):
    def test_fundamentals_only_become_visible_on_notice_date(self):
        raw = pd.DataFrame(
            {
                "SECURITY_CODE": ["000001", "000001"],
                "REPORT_DATE": ["2024-03-31", "2024-06-30"],
                "NOTICE_DATE": ["2024-04-30", "2024-06-01"],
                "ROEJQ": [8.0, 9.0],
            }
        )
        normalized = normalize_fundamentals(raw)
        self.assertEqual(len(normalized), 1)
        panel = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-04-29", "2024-04-30"]),
                "symbol": ["000001", "000001"],
            }
        )
        attached = attach_fundamentals_asof(panel, normalized)
        self.assertTrue(pd.isna(attached.loc[0, "roe"]))
        self.assertEqual(attached.loc[1, "roe"], 8.0)

    def test_multihorizon_dataset_contains_fundamental_features(self):
        panel = make_demo_panel(symbols=12, periods=340)
        for index, column in enumerate(
            [
                "roe",
                "roic",
                "debt_ratio",
                "revenue_growth",
                "profit_growth",
                "operating_cash_margin",
                "gross_margin",
            ],
            1,
        ):
            panel[column] = panel["symbol"].astype(int) * index
        result = build_v3_dataset(panel)
        self.assertTrue(set(V3_FEATURES).issubset(result.columns))
        self.assertIn("label_20", result)
        self.assertFalse(result[V3_FEATURES].isna().any().any())

    def test_nested_selection_uses_previous_year_only(self):
        rows = []
        for year in [2024, 2025]:
            for candidate, value in [("A", 0.02), ("B", -0.01)]:
                rows.append(
                    {
                        "date": f"{year}-06-30",
                        "candidate": candidate,
                        "period_return": value if year == 2024 else -value,
                        "benchmark_return": 0.0,
                        "rank_ic": value,
                        "buy_turnover": 0.1,
                        "sell_turnover": 0.1,
                    }
                )
        folds, selected = nested_year_selection(pd.DataFrame(rows))
        self.assertEqual(folds.iloc[0]["selected_candidate"], "A")
        self.assertEqual(selected.iloc[0]["candidate"], "A")


if __name__ == "__main__":
    unittest.main()
