from __future__ import annotations

import numpy as np
import pandas as pd

from stockpilot.features import FEATURE_COLUMNS as TECHNICAL_FEATURES
from stockpilot.features import build_dataset

FUNDAMENTAL_RAW = [
    "roe",
    "roic",
    "debt_ratio",
    "revenue_growth",
    "profit_growth",
    "operating_cash_margin",
    "gross_margin",
]
FUNDAMENTAL_FEATURES = [f"{column}_rank" for column in FUNDAMENTAL_RAW]
V3_FEATURES = [*TECHNICAL_FEATURES, *FUNDAMENTAL_FEATURES, "fundamental_coverage"]


def _cross_section_rank(data: pd.DataFrame, column: str) -> pd.Series:
    numeric = pd.to_numeric(data[column], errors="coerce")

    def rank_group(values: pd.Series) -> pd.Series:
        valid = values.dropna()
        if len(valid) < 5:
            return pd.Series(0.0, index=values.index)
        lower, upper = valid.quantile([0.01, 0.99])
        clipped = values.clip(lower=lower, upper=upper)
        return clipped.rank(pct=True, method="average") - 0.5

    return numeric.groupby(data["date"], group_keys=False).apply(rank_group).fillna(0.0)


def build_v3_dataset(panel: pd.DataFrame, horizons: tuple[int, ...] = (5, 10, 20)) -> pd.DataFrame:
    if 5 not in horizons:
        raise ValueError("V3执行回测要求包含5日周期")
    base = build_dataset(panel, horizon=5, label_mode="neutral")
    base = base.rename(
        columns={
            "label": "label_5",
            "future_return": "future_return_5",
            "label_end_date": "label_end_date_5",
        }
    )
    for horizon in horizons:
        if horizon == 5:
            continue
        extra = build_dataset(panel, horizon=horizon, label_mode="neutral")
        extra = extra[["date", "symbol", "label", "future_return", "label_end_date"]].rename(
            columns={
                "label": f"label_{horizon}",
                "future_return": f"future_return_{horizon}",
                "label_end_date": f"label_end_date_{horizon}",
            }
        )
        base = base.merge(extra, on=["date", "symbol"], how="left", validate="one_to_one")
    available = base[FUNDAMENTAL_RAW].notna()
    base["fundamental_coverage"] = available.mean(axis=1)
    for column in FUNDAMENTAL_RAW:
        base[f"{column}_rank"] = _cross_section_rank(base, column)
    base["debt_ratio_rank"] *= -1
    base["quality_score"] = base[
        [
            "roe_rank",
            "roic_rank",
            "debt_ratio_rank",
            "operating_cash_margin_rank",
            "gross_margin_rank",
        ]
    ].mean(axis=1)
    base["growth_score"] = base[["revenue_growth_rank", "profit_growth_rank"]].mean(axis=1)
    base["stable_factor_score"] = (
        0.35 * base["quality_score"]
        + 0.20 * base["growth_score"]
        - 0.45 * base["volatility_20_rank"]
    )
    base[V3_FEATURES] = base[V3_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return base.sort_values(["date", "symbol"]).reset_index(drop=True)
