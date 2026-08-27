from __future__ import annotations

import numpy as np
import pandas as pd

from research_v3.features import _cross_section_rank
from research_v5.features import MODEL_FEATURES, build_v5_dataset


FUNDAMENTAL_FEATURES = [
    "roe_rank",
    "roic_rank",
    "debt_ratio_rank",
    "revenue_growth_rank",
    "profit_growth_rank",
    "operating_cash_margin_rank",
    "gross_margin_rank",
    "fundamental_coverage",
]
VALUATION_FEATURES = ["book_to_price_rank", "earnings_yield_rank"]
BEHAVIOR_FEATURES = [*MODEL_FEATURES, "industry_momentum"]
ENHANCED_FEATURES = [*FUNDAMENTAL_FEATURES, *VALUATION_FEATURES, *BEHAVIOR_FEATURES]


def build_v8_dataset(panel: pd.DataFrame) -> pd.DataFrame:
    data = build_v5_dataset(panel)
    close = pd.to_numeric(data["close"], errors="coerce").where(lambda value: value > 0)
    data["book_to_price"] = pd.to_numeric(
        data["book_value_per_share"], errors="coerce"
    ) / close
    data["earnings_yield"] = pd.to_numeric(
        data["earnings_per_share"], errors="coerce"
    ) / close
    data["book_to_price_rank"] = _cross_section_rank(data, "book_to_price")
    data["earnings_yield_rank"] = _cross_section_rank(data, "earnings_yield")
    industry_momentum = data.groupby(["date", "industry"], dropna=False)["ret_20"].rank(
        pct=True, method="average"
    )
    data["industry_momentum"] = industry_momentum.sub(0.5).fillna(0.0)
    data[ENHANCED_FEATURES] = (
        data[ENHANCED_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    return data

