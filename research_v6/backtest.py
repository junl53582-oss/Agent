from __future__ import annotations

import numpy as np
import pandas as pd

from research_v4.config import V4Settings
from research_v4.stability import learn_factor_specs
from research_v5.backtest import max_drawdown
from research_v5.models import fit_v5_models
from stockpilot.portfolio import turnover

from .config import V6Settings
from .model import score_v6, select_sector_balanced


def run_v6_backtest(
    dataset: pd.DataFrame, settings: V6Settings | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    settings = settings or V6Settings()
    v5_models = {}
    v4_specs = {}
    spec_rows = []
    for year in settings.test_years:
        v5_models[year] = fit_v5_models(dataset, year)
        specs, _ = learn_factor_specs(dataset, year, V4Settings())
        v4_specs[year] = specs
        for spec in specs:
            spec_rows.append({"test_year": year, **spec.__dict__})
    eligible = dataset[dataset["eligible"] & dataset["future_return_5"].notna()].copy()
    eligible["date"] = pd.to_datetime(eligible["date"])
    test = eligible[eligible["date"].dt.year.isin(settings.test_years)]
    dates = test["date"].drop_duplicates().sort_values().reset_index(drop=True)
    previous_weights: dict[str, float] = {}
    rows, signals, sector_ics = [], [], []
    buy_rate = settings.fee_rate + settings.slippage
    sell_rate = settings.fee_rate + settings.slippage + settings.stamp_duty
    for date in dates.iloc[:: settings.rebalance_every]:
        year = int(date.year)
        current = score_v6(test[test["date"] == date], v5_models[year], v4_specs[year], settings)
        current["pred_rank"] = current["score"].rank(ascending=False, method="first")
        selected = select_sector_balanced(current, settings)
        continuing = selected["symbol"].isin(previous_weights)
        selected["holding_return"] = selected["execution_exit_open"] / selected["entry_open"] - 1
        selected["realized_return"] = selected["execution_return"].where(
            ~continuing, selected["holding_return"]
        )
        selected["executed"] = (
            continuing | selected["entry_tradable"].fillna(False)
        ) & selected["realized_return"].notna()
        selected["weight"] = selected["weight"].where(selected["executed"], 0.0)
        weights = {row.symbol: float(row.weight) for row in selected.itertuples() if row.weight > 0}
        buys, sells = turnover(previous_weights, weights)
        cost = buys * buy_rate + sells * sell_rate
        gross = float((selected["weight"] * selected["realized_return"].fillna(0)).sum())
        rank_ic = current["score"].corr(current["label_5"], method="spearman")
        rows.append(
            {"date": date, "test_year": year, "regime": current["regime"].iloc[0], "period_return": gross - cost, "benchmark_return": float(current["future_return_5"].mean()), "rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan, "holdings": len(weights), "cash_weight": 1 - sum(weights.values()), "buy_turnover": buys, "sell_turnover": sells, "transaction_cost": cost}
        )
        for sector, group in current.groupby("broad_sector"):
            value = group["score"].corr(group["label_5"], method="spearman")
            if len(group) >= 10 and pd.notna(value):
                sector_ics.append({"date": date, "test_year": year, "broad_sector": sector, "rank_ic": value, "stocks": len(group)})
        for rank, row in enumerate(selected.itertuples(), 1):
            signals.append({"date": date, "test_year": year, "rank": rank, "symbol": row.symbol, "broad_sector": row.broad_sector, "score": row.score, "weight": row.weight, "executed": bool(row.executed)})
        previous_weights = weights
    return pd.DataFrame(rows), pd.DataFrame(signals), pd.DataFrame(sector_ics), pd.DataFrame(spec_rows)


__all__ = ["max_drawdown", "run_v6_backtest"]
