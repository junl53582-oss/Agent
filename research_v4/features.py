from __future__ import annotations

import numpy as np
import pandas as pd

from research_v3.features import build_v3_dataset

FACTOR_COLUMNS = ["quality", "growth", "low_volatility", "trend"]


def _rank_factor(data: pd.DataFrame, values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return (
        numeric.groupby(data["date"])
        .rank(pct=True, method="average")
        .sub(0.5)
        .fillna(0.0)
    )


def build_v4_dataset(panel: pd.DataFrame) -> pd.DataFrame:
    data = build_v3_dataset(panel, horizons=(5,))
    data["quality"] = _rank_factor(data, data["quality_score"])
    data["growth"] = _rank_factor(data, data["growth_score"])
    data["low_volatility"] = _rank_factor(data, -data["volatility_20_rank"])
    trend_raw = data[["ret_20_rank", "momentum_60_rank", "ma_gap_20_rank"]].mean(axis=1)
    data["trend"] = _rank_factor(data, trend_raw)
    data[FACTOR_COLUMNS] = data[FACTOR_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
    return data
