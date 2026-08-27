from __future__ import annotations

import numpy as np
import pandas as pd

from .trading import add_execution_columns, board_name

FEATURE_COLUMNS = [
    "ret_1_rank",
    "ret_5_rank",
    "ret_20_rank",
    "momentum_60_rank",
    "ma_gap_20_rank",
    "volatility_20_rank",
    "range_10_rank",
    "volume_ratio_20_rank",
    "amount_rank",
    "rsi_14_rank",
]


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _neutralize_future_return(data: pd.DataFrame) -> pd.Series:
    """Cross-sectionally remove observable size/liquidity and group effects."""
    result = pd.Series(np.nan, index=data.index, dtype=float)
    size_column = next(
        (name for name in ["float_market_cap", "market_cap"] if name in data.columns), None
    )
    if size_column:
        size_proxy = np.log1p(pd.to_numeric(data[size_column], errors="coerce").clip(lower=0))
    else:
        size_proxy = data["amount_log"]
    industry_group = (
        data["industry"].fillna(data["board"]) if "industry" in data.columns else data["board"]
    )
    membership = (
        data["in_universe"].fillna(False).astype(bool)
        if "in_universe" in data.columns
        else pd.Series(True, index=data.index)
    )
    for indexes in data.groupby("date", sort=False).groups.values():
        group = data.loc[indexes]
        valid = (
            group["future_return"].notna()
            & size_proxy.loc[indexes].notna()
            & membership.loc[indexes]
        )
        if valid.sum() < 5:
            continue
        valid_indexes = group.index[valid]
        size_rank = size_proxy.loc[valid_indexes].rank(pct=True).to_numpy(dtype=float) - 0.5
        categories = pd.get_dummies(
            industry_group.loc[valid_indexes].fillna("未知").astype(str),
            drop_first=True,
            dtype=float,
        )
        design = np.column_stack(
            [np.ones(len(valid_indexes)), size_rank, categories.to_numpy(dtype=float)]
        )
        target = group.loc[valid_indexes, "future_return"].to_numpy(dtype=float)
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        result.loc[valid_indexes] = target - design @ coefficients
    return result


def build_dataset(
    panel: pd.DataFrame, horizon: int = 5, label_mode: str = "neutral"
) -> pd.DataFrame:
    """Build close-known features and next-open-to-future-open labels without leakage."""
    data = panel.sort_values(["symbol", "date"]).copy()
    grouped = data.groupby("symbol", group_keys=False)
    data["ret_1"] = grouped["close"].pct_change(1)
    data["ret_5"] = grouped["close"].pct_change(5)
    data["ret_20"] = grouped["close"].pct_change(20)
    data["momentum_60"] = grouped["close"].pct_change(60)
    ma20 = grouped["close"].transform(lambda s: s.rolling(20).mean())
    data["ma_gap_20"] = data["close"] / ma20 - 1
    data["volatility_20"] = grouped["close"].transform(lambda s: s.pct_change().rolling(20).std())
    daily_range = (data["high"] - data["low"]) / data["close"]
    data["range_10"] = daily_range.groupby(data["symbol"]).transform(lambda s: s.rolling(10).mean())
    volume_ma = grouped["volume"].transform(lambda s: s.rolling(20).mean())
    data["volume_ratio_20"] = data["volume"] / volume_ma.replace(0, np.nan)
    data["amount_log"] = np.log1p(data["amount"].clip(lower=0))
    data["rsi_14"] = grouped["close"].transform(_rsi)
    data["board"] = data["symbol"].map(board_name)

    raw_features = [
        "ret_1",
        "ret_5",
        "ret_20",
        "momentum_60",
        "ma_gap_20",
        "volatility_20",
        "range_10",
        "volume_ratio_20",
        "amount_log",
        "rsi_14",
    ]
    for col in raw_features:
        rank_name = "amount_rank" if col == "amount_log" else f"{col}_rank"
        data[rank_name] = data.groupby("date")[col].rank(pct=True) - 0.5

    data["entry_open"] = grouped["open"].shift(-1)
    data["exit_open"] = grouped["open"].shift(-(horizon + 1))
    data["label_end_date"] = grouped["date"].shift(-(horizon + 1))
    data["future_return"] = data["exit_open"] / data["entry_open"] - 1
    label_universe = (
        data["in_universe"].fillna(False).astype(bool)
        if "in_universe" in data.columns
        else pd.Series(True, index=data.index)
    )
    benchmark = data["future_return"].where(label_universe).groupby(data["date"]).transform("mean")
    data["raw_label"] = data["future_return"] - benchmark
    data["neutral_label"] = _neutralize_future_return(data)
    if label_mode == "neutral":
        data["label"] = data["neutral_label"]
    elif label_mode == "market_relative":
        data["label"] = data["raw_label"]
    else:
        raise ValueError("label_mode 必须是 neutral 或 market_relative")

    observations = grouped.cumcount() + 1
    liquid_cutoff = data.groupby("date")["amount"].transform(lambda s: s.quantile(0.2))
    names = data["name"].astype(str) if "name" in data else pd.Series("", index=data.index)
    special_treatment = names.str.upper().str.contains(r"ST|退", regex=True, na=False)
    listed_long_enough = pd.Series(True, index=data.index)
    if "list_date" in data:
        list_date = pd.to_datetime(data["list_date"], errors="coerce")
        listed_long_enough = (data["date"] - list_date).dt.days >= 180
    point_in_time_member = pd.Series(True, index=data.index)
    if "in_universe" in data:
        point_in_time_member = data["in_universe"].fillna(False).astype(bool)
    daily_close_return = grouped["close"].pct_change()
    bad_price_symbol = daily_close_return.abs().groupby(data["symbol"]).transform("max") > 0.35
    data["data_quality_ok"] = ~bad_price_symbol
    data["eligible"] = (
        (observations >= 120)
        & (data["amount"] >= liquid_cutoff)
        & (data["volume"] > 0)
        & ~special_treatment
        & listed_long_enough
        & point_in_time_member
        & data["data_quality_ok"]
        & data[FEATURE_COLUMNS].notna().all(axis=1)
    )
    return add_execution_columns(data, horizon)
