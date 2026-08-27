from __future__ import annotations

import numpy as np
import pandas as pd

from research_v10.features import build_v10_dataset

from .config import V12Settings


def _sector_benchmark_return(data: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=data.index, dtype=float)
    for indexes in data.groupby(["date", "broad_sector"], sort=False).groups.values():
        group = data.loc[indexes]
        valid = group["eligible"].fillna(False) & group["future_return_20"].notna()
        if valid.sum() < 1:
            continue
        idx = group.index[valid]
        weights = pd.to_numeric(group.loc[idx, "benchmark_weight"], errors="coerce").clip(lower=0)
        target = pd.to_numeric(group.loc[idx, "future_return_20"], errors="coerce")
        benchmark = float(np.average(target, weights=weights)) if weights.sum() > 0 else float(target.mean())
        result.loc[idx] = benchmark
    return result


def build_v12_dataset(
    panel: pd.DataFrame, settings: V12Settings | None = None
) -> pd.DataFrame:
    settings = settings or V12Settings()
    data = build_v10_dataset(panel)
    data["sector_benchmark_return_20"] = _sector_benchmark_return(data)
    liquidity_rank = data.groupby("date")["amount"].rank(pct=True, method="average").fillna(0.5)
    fixed_round_trip = (
        2 * settings.fee_rate + 2 * settings.slippage + settings.stamp_duty
    )
    data["estimated_round_trip_cost"] = (
        fixed_round_trip + settings.liquidity_impact_max * (1 - liquidity_rank)
    )
    data["v12_net_marginal_target"] = (
        data["future_return_20"]
        - data["sector_benchmark_return_20"]
        - data["estimated_round_trip_cost"]
    )

    universe = data["in_universe"].fillna(False)
    weights = pd.to_numeric(data["benchmark_weight"], errors="coerce").clip(lower=0).where(universe, 0)
    ret1 = pd.to_numeric(data["ret_1"], errors="coerce")
    weighted = (ret1 * weights).groupby(data["date"]).sum(min_count=1)
    weight_sum = weights.where(ret1.notna()).groupby(data["date"]).sum(min_count=1)
    market_return = weighted / weight_sum.replace(0, np.nan)
    market_volatility = market_return.rolling(60, min_periods=40).std() * np.sqrt(252)
    market_momentum = (1 + market_return.fillna(0)).rolling(60, min_periods=40).apply(np.prod, raw=True) - 1
    data = data.join(market_return.rename("market_return_1"), on="date")
    data = data.join(market_volatility.rename("market_volatility_60"), on="date")
    # V10 already contains a cross-sectional market_momentum_60 feature.
    # Keep the V12 official-weight benchmark series under its own name.
    data = data.join(market_momentum.rename("v12_market_momentum_60"), on="date")
    return data.sort_values(["date", "symbol"]).reset_index(drop=True)
