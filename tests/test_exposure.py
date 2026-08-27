import unittest

import pandas as pd

from stockpilot.exposure import (
    attach_exposures,
    attach_exposures_asof,
    combine_exposure,
    exposure_coverage,
    normalize_industry_history,
    normalize_market_cap,
)


class ExposureTests(unittest.TestCase):
    def test_market_cap_and_industry_are_point_in_time(self):
        market = normalize_market_cap(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                    "close": [10.0, 11.0, 12.0],
                    "outstanding_share": [100.0, 100.0, 100.0],
                }
            ),
            "000001",
        )
        raw_industry = pd.DataFrame(
            {
                "变更日期": ["2024-01-03", "2024-01-03"],
                "分类标准": ["申银万国行业分类标准(旧)", "申银万国行业分类标准"],
                "行业门类": ["旧行业", "新行业"],
                "行业编码": ["OLD", "NEW"],
            }
        )
        industry = normalize_industry_history(raw_industry, "000001")
        exposure = combine_exposure(market, industry)
        self.assertEqual(exposure.loc[0, "float_market_cap"], 1000)
        self.assertTrue(pd.isna(exposure.loc[0, "industry"]))
        self.assertEqual(exposure.loc[1, "industry"], "新行业")
        self.assertLessEqual(exposure.loc[1, "industry_effective_date"], exposure.loc[1, "date"])

    def test_attach_and_coverage(self):
        panel = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "symbol": ["000001", "000001"],
                "in_universe": [True, True],
            }
        )
        exposure = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "symbol": ["000001", "000001"],
                "float_market_cap": [1000.0, 1100.0],
                "outstanding_share": [100.0, 100.0],
                "industry": ["银行", "银行"],
                "industry_code": ["1", "1"],
                "industry_effective_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
                "exposure_source": ["test", "test"],
            }
        )
        enriched = attach_exposures(panel, exposure)
        coverage = exposure_coverage(enriched)
        self.assertEqual(coverage["float_market_cap_coverage"], 1)
        self.assertEqual(coverage["industry_coverage"], 1)
        self.assertTrue(coverage["industry_point_in_time"])

        future = pd.concat(
            [
                panel,
                pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2024-01-04"]),
                        "symbol": ["000001"],
                        "in_universe": [True],
                    }
                ),
            ],
            ignore_index=True,
        )
        carried = attach_exposures_asof(future, exposure)
        latest = carried.iloc[-1]
        self.assertEqual(latest["industry"], "银行")
        self.assertEqual(latest["exposure_age_days"], 1)


if __name__ == "__main__":
    unittest.main()
