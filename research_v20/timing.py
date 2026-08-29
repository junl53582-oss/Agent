import numpy as np
import pandas as pd

from .config import V20Settings


def historical_market_state(dataset, settings=None):
    """PIT constituent proxy, NOT the official CSI300 index.

    A close-T decision uses closes through T and weights known at T-1.
    Execution must be after T. Never read a forward-return/label column.
    Missing constituent bars reduce coverage rather than bridging missing days.
    """
    settings = settings or V20Settings()
    frame = dataset[["date", "symbol", "close", "benchmark_weight", "in_universe"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("duplicate market date/symbol")
    frame = frame.sort_values(["symbol", "date"])
    dates = pd.DatetimeIndex(frame["date"].unique()).sort_values()
    day_number = pd.Series(range(len(dates)), index=dates)
    frame["day_number"] = frame["date"].map(day_number)
    weights = pd.to_numeric(frame["benchmark_weight"], errors="coerce")
    frame["weight"] = weights.where(frame["in_universe"].eq(True), 0.0).fillna(0).clip(lower=0)
    total_weight = frame.groupby("date")["weight"].transform("sum")
    frame["weight"] = frame["weight"] / total_weight.replace(0, np.nan)
    grouped = frame.groupby("symbol", sort=False)
    previous_close = grouped["close"].shift(1)
    previous_weight = grouped["weight"].shift(1)
    consecutive = frame["day_number"].sub(grouped["day_number"].shift(1)).eq(1)
    returns = pd.to_numeric(frame["close"], errors="coerce") / previous_close - 1
    valid = consecutive & np.isfinite(returns) & previous_close.gt(0)
    frame["covered_weight"] = previous_weight.where(valid, 0.0).fillna(0.0)
    frame["weighted_return"] = returns.where(valid, 0.0) * frame["covered_weight"]
    daily = frame.groupby("date")[["weighted_return", "covered_weight"]].sum().reindex(dates)
    daily["market_return"] = (
        daily["weighted_return"] / daily["covered_weight"].replace(0, np.nan)
    ).where(daily["covered_weight"] >= settings.minimum_market_coverage)
    window = settings.timing_window_days
    daily["market_momentum"] = (1 + daily["market_return"]).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1
    daily["market_data_end"] = daily.index
    daily["market_data_start"] = pd.Series(dates, index=dates).shift(window)
    return daily


def weights_for_momentum(momentum, settings=None):
    settings = settings or V20Settings()
    if not np.isfinite(momentum):
        raise ValueError("market history is incomplete; cannot classify regime")
    if momentum > settings.bull_threshold:
        return settings.weight_bull, "bull"
    if momentum < settings.bear_threshold:
        return settings.weight_bear, "bear"
    return settings.weight_neutral, "neutral"
