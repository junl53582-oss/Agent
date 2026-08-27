from __future__ import annotations

import numpy as np
import pandas as pd


def board_name(symbol: str) -> str:
    symbol = str(symbol).zfill(6)
    if symbol.startswith(("300", "301")):
        return "创业板"
    if symbol.startswith("688"):
        return "科创板"
    if symbol.startswith(("4", "8", "92")):
        return "北交所"
    if symbol.startswith("6"):
        return "沪市主板"
    return "深市主板"


def price_limit_rate(symbol: str, name: str = "", date: pd.Timestamp | str | None = None) -> float:
    """Approximate the normal daily price limit by board and security name."""
    symbol = str(symbol).zfill(6)
    upper_name = str(name).upper()
    if "ST" in upper_name:
        return 0.05
    if symbol.startswith(("300", "301")):
        if date is not None and pd.Timestamp(date) < pd.Timestamp("2020-08-24"):
            return 0.10
        return 0.20
    if symbol.startswith("688"):
        return 0.20
    if symbol.startswith(("4", "8", "92")):
        return 0.30
    return 0.10


def add_execution_columns(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add next-open execution flags and one-day deferred exits for limit-down days."""
    result = data.sort_values(["symbol", "date"]).copy()
    grouped = result.groupby("symbol", group_keys=False)
    names = result["name"] if "name" in result else pd.Series("", index=result.index)
    result["limit_rate"] = [
        price_limit_rate(symbol, name, date)
        for symbol, name, date in zip(result["symbol"], names, result["date"])
    ]
    result["board"] = result["symbol"].map(board_name)
    result["entry_date"] = grouped["date"].shift(-1)
    next_volume = grouped["volume"].shift(-1)
    entry_gap = result["entry_open"] / result["close"] - 1
    result["entry_limit_up"] = entry_gap >= result["limit_rate"] * 0.995
    result["entry_tradable"] = (
        result["entry_open"].notna() & (next_volume > 0) & ~result["entry_limit_up"]
    )

    exit_reference_close = grouped["close"].shift(-horizon)
    exit_volume = grouped["volume"].shift(-(horizon + 1))
    exit_gap = result["exit_open"] / exit_reference_close - 1
    result["exit_limit_down"] = exit_gap <= -result["limit_rate"] * 0.995
    normal_exit = result["exit_open"].notna() & (exit_volume > 0) & ~result["exit_limit_down"]

    deferred_open = grouped["open"].shift(-(horizon + 2))
    deferred_volume = grouped["volume"].shift(-(horizon + 2))
    result["exit_deferred"] = ~normal_exit & deferred_open.notna() & (deferred_volume > 0)
    result["execution_exit_open"] = np.where(normal_exit, result["exit_open"], deferred_open)
    normal_exit_date = grouped["date"].shift(-(horizon + 1))
    deferred_exit_date = grouped["date"].shift(-(horizon + 2))
    result["execution_exit_date"] = normal_exit_date.where(normal_exit, deferred_exit_date)
    result["execution_return"] = result["execution_exit_open"] / result["entry_open"] - 1
    result.loc[~result["entry_tradable"], "execution_return"] = np.nan
    return result.sort_values(["date", "symbol"]).reset_index(drop=True)
