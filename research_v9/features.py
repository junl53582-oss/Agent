from __future__ import annotations

import numpy as np
import pandas as pd

from research_v3.features import FUNDAMENTAL_RAW, _cross_section_rank
from research_v8.features import ENHANCED_FEATURES, build_v8_dataset


FUNDAMENTAL_CHANGE_FEATURES = [f"{name}_change_rank" for name in FUNDAMENTAL_RAW]
TECHNICAL_LONG_FEATURES = [
    "momentum_120_rank",
    "volatility_60_rank",
    "downside_volatility_60_rank",
    "drawdown_120_rank",
    "price_position_120_rank",
    "volume_trend_60_rank",
    "intraday_strength_20_rank",
    "overnight_gap_20_rank",
]
PIT_FEATURES = [
    *FUNDAMENTAL_CHANGE_FEATURES,
    "fundamental_freshness_rank",
    "benchmark_weight_rank",
]
TECHNOLOGY_FEATURES = [
    "technology_momentum_rank",
    "technology_growth_rank",
    "technology_quality_rank",
    "technology_valuation_rank",
]
V9_FEATURES = [
    *ENHANCED_FEATURES,
    *PIT_FEATURES,
    *TECHNICAL_LONG_FEATURES,
    *TECHNOLOGY_FEATURES,
]


def _rank(data: pd.DataFrame, values: pd.Series) -> pd.Series:
    return values.groupby(data["date"]).rank(pct=True, method="average").sub(0.5).fillna(0)


def _fundamental_changes(data: pd.DataFrame) -> pd.DataFrame:
    keys = ["symbol", "available_date", *FUNDAMENTAL_RAW]
    filings = (
        data.loc[data["available_date"].notna(), keys]
        .drop_duplicates(["symbol", "available_date"], keep="last")
        .sort_values(["symbol", "available_date"])
    )
    for column in FUNDAMENTAL_RAW:
        current = pd.to_numeric(filings[column], errors="coerce")
        previous = current.groupby(filings["symbol"]).shift(1)
        filings[f"{column}_change"] = (current - previous) / (previous.abs() + 1.0)
    columns = ["symbol", "available_date", *[f"{name}_change" for name in FUNDAMENTAL_RAW]]
    return data.merge(filings[columns], on=["symbol", "available_date"], how="left")


def _residual_target(data: pd.DataFrame) -> pd.Series:
    """Remove only same-date observable style and industry effects from the target."""
    result = pd.Series(np.nan, index=data.index, dtype=float)
    for indexes in data.groupby("date", sort=False).groups.values():
        group = data.loc[indexes]
        valid = group["eligible"].fillna(False) & group["label_5"].notna()
        if valid.sum() < 20:
            continue
        idx = group.index[valid]
        style = group.loc[idx, ["benchmark_weight_rank", "momentum", "low_volatility"]]
        categories = pd.get_dummies(
            group.loc[idx, "industry"].fillna("未知").astype(str), drop_first=True, dtype=float
        )
        design = np.column_stack(
            [np.ones(len(idx)), style.to_numpy(dtype=float), categories.to_numpy(dtype=float)]
        )
        target = group.loc[idx, "label_5"].to_numpy(dtype=float)
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        result.loc[idx] = target - design @ coefficients
    return result


def build_v9_dataset(panel: pd.DataFrame) -> pd.DataFrame:
    data = _fundamental_changes(build_v8_dataset(panel))
    data = data.sort_values(["symbol", "date"]).copy()
    grouped = data.groupby("symbol", group_keys=False)
    close = pd.to_numeric(data["close"], errors="coerce")
    open_price = pd.to_numeric(data["open"], errors="coerce")
    volume = pd.to_numeric(data["volume"], errors="coerce")
    returns = grouped["close"].pct_change()
    data["momentum_120"] = grouped["close"].pct_change(120)
    data["volatility_60"] = returns.groupby(data["symbol"]).transform(
        lambda value: value.rolling(60).std()
    )
    downside = returns.where(returns < 0, 0.0)
    data["downside_volatility_60"] = downside.groupby(data["symbol"]).transform(
        lambda value: value.rolling(60).std()
    )
    high_120 = grouped["high"].transform(lambda value: value.rolling(120).max())
    low_120 = grouped["low"].transform(lambda value: value.rolling(120).min())
    data["drawdown_120"] = close / high_120 - 1
    data["price_position_120"] = (close - low_120) / (high_120 - low_120).replace(0, np.nan)
    volume_20 = grouped["volume"].transform(lambda value: value.rolling(20).mean())
    volume_60 = grouped["volume"].transform(lambda value: value.rolling(60).mean())
    data["volume_trend_60"] = volume_20 / volume_60.replace(0, np.nan) - 1
    intraday = (close - open_price) / open_price.replace(0, np.nan)
    data["intraday_strength_20"] = intraday.groupby(data["symbol"]).transform(
        lambda value: value.rolling(20).mean()
    )
    previous_close = grouped["close"].shift(1)
    gap = open_price / previous_close.replace(0, np.nan) - 1
    data["overnight_gap_20"] = gap.groupby(data["symbol"]).transform(
        lambda value: value.rolling(20).mean()
    )

    raw_long = [name.removesuffix("_rank") for name in TECHNICAL_LONG_FEATURES]
    for raw, ranked in zip(raw_long, TECHNICAL_LONG_FEATURES):
        data[ranked] = _rank(data, data[raw])
    # Risk features have an intuitive positive direction before entering the models.
    data["volatility_60_rank"] *= -1
    data["downside_volatility_60_rank"] *= -1

    for column in FUNDAMENTAL_RAW:
        data[f"{column}_change_rank"] = _cross_section_rank(data, f"{column}_change")
    data["debt_ratio_change_rank"] *= -1
    age = pd.to_numeric(data.get("fundamental_age_days"), errors="coerce")
    data["fundamental_freshness_rank"] = _rank(data, -age)
    data["benchmark_weight_rank"] = _rank(
        data, pd.to_numeric(data["benchmark_weight"], errors="coerce")
    )

    technology = data["broad_sector"].eq("technology")
    tech_group = [data["date"], technology]
    data["technology_momentum_rank"] = data["momentum"].groupby(tech_group).rank(
        pct=True, method="average"
    ).sub(0.5).where(technology, 0).fillna(0)
    data["technology_growth_rank"] = data["growth"].groupby(tech_group).rank(
        pct=True, method="average"
    ).sub(0.5).where(technology, 0).fillna(0)
    data["technology_quality_rank"] = data["quality"].groupby(tech_group).rank(
        pct=True, method="average"
    ).sub(0.5).where(technology, 0).fillna(0)
    valuation = data[["book_to_price_rank", "earnings_yield_rank"]].mean(axis=1)
    data["technology_valuation_rank"] = valuation.groupby(tech_group).rank(
        pct=True, method="average"
    ).sub(0.5).where(technology, 0).fillna(0)

    data[V9_FEATURES] = data[V9_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    data["v9_target_5"] = _residual_target(data)
    return data.sort_values(["date", "symbol"]).reset_index(drop=True)

