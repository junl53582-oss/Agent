from __future__ import annotations

import pandas as pd

from research_v5.features import build_v5_dataset
from stockpilot.features import build_dataset


def build_v7_dataset(panel: pd.DataFrame, horizons: tuple[int, ...] = (5, 20, 60)) -> pd.DataFrame:
    if 5 not in horizons:
        raise ValueError("V7要求包含5日执行周期")
    data = build_v5_dataset(panel)
    for horizon in horizons:
        if horizon == 5:
            continue
        labels = build_dataset(panel, horizon=horizon, label_mode="neutral")
        labels = labels[["date", "symbol", "label", "future_return", "label_end_date"]].rename(
            columns={
                "label": f"label_{horizon}",
                "future_return": f"future_return_{horizon}",
                "label_end_date": f"label_end_date_{horizon}",
            }
        )
        data = data.merge(labels, on=["date", "symbol"], how="left", validate="one_to_one")
    return data
