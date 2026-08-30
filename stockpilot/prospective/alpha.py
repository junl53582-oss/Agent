from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _daily_ic(frame: pd.DataFrame, factor: str, target: str, method: str) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for date, group in frame.groupby("date", sort=True):
        pair = group[[factor, target]].dropna()
        if len(pair) >= 10 and pair[factor].nunique() > 1 and pair[target].nunique() > 1:
            values[pd.Timestamp(date)] = float(pair[factor].corr(pair[target], method=method))
    return pd.Series(values, dtype=float)


def factor_validation_metrics(frame: pd.DataFrame, factor: str, target: str) -> dict:
    required = {"date", "symbol", factor, target}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"factor validation input missing {sorted(missing)}")
    pearson = _daily_ic(frame, factor, target, "pearson")
    rank = _daily_ic(frame, factor, target, "spearman")
    std = float(rank.std(ddof=1)) if len(rank) > 1 else math.nan
    mean = float(rank.mean()) if len(rank) else math.nan
    return {
        "pearson_ic": float(pearson.mean()) if len(pearson) else math.nan,
        "spearman_rank_ic": mean,
        "ic_mean": mean,
        "ic_std": std,
        "icir": mean / std if np.isfinite(std) and std > 0 else math.nan,
        "positive_ic_ratio": float(rank.gt(0).mean()) if len(rank) else math.nan,
        "t_stat": mean / (std / math.sqrt(len(rank))) if np.isfinite(std) and std > 0 else math.nan,
        "coverage": float(frame[factor].notna().mean()),
        "cross_sectional_dispersion": float(
            frame.groupby("date")[factor].std(ddof=1).mean()
        ),
        "dates": int(len(rank)),
    }


def factor_decay_metrics(frame: pd.DataFrame, factor: str) -> pd.DataFrame:
    rows = []
    for horizon in (1, 5, 20):
        target = f"forward_return_{horizon}d"
        if target not in frame:
            rows.append({"horizon": horizon, "status": "LABEL_UNAVAILABLE"})
            continue
        rows.append({"horizon": horizon, "status": "AVAILABLE", **factor_validation_metrics(frame, factor, target)})
    return pd.DataFrame(rows)


def grouped_stability(frame: pd.DataFrame, factor: str, target: str, group: str) -> pd.DataFrame:
    if group not in frame:
        raise ValueError(f"stability group missing: {group}")
    rows = []
    for value, subset in frame.groupby(group, dropna=False):
        rows.append({group: value, "sample_size": len(subset), **factor_validation_metrics(subset, factor, target)})
    return pd.DataFrame(rows)


def turnover_by_date(frame: pd.DataFrame, factor: str) -> float:
    ranks = frame.pivot(index="date", columns="symbol", values=factor).rank(axis=1, pct=True)
    return float(ranks.diff().abs().mean(axis=1).mean())
