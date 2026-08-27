from __future__ import annotations

import pandas as pd

from research_v12.features import build_v12_dataset


def build_v13_dataset(panel: pd.DataFrame) -> pd.DataFrame:
    data = build_v12_dataset(panel)
    market_index = (
        data[["date", "market_return_1"]]
        .drop_duplicates("date")
        .sort_values("date")
        .set_index("date")
    )
    equity = (1 + market_index["market_return_1"].fillna(0)).cumprod()
    rolling_high = equity.rolling(120, min_periods=20).max()
    market_index["v13_market_drawdown_120"] = equity / rolling_high - 1
    return data.join(market_index["v13_market_drawdown_120"], on="date")

