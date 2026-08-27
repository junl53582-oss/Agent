from __future__ import annotations

import numpy as np
import pandas as pd

from research_v4.config import V4Settings
from research_v4.stability import learn_factor_specs
from research_v5.models import fit_v5_models
from research_v6.config import V6Settings
from research_v6.model import score_v6, select_sector_balanced
from stockpilot.portfolio import turnover

from .config import V7Settings
from .model import fit_multihorizon_models, score_multihorizon


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min()) if not equity.empty else 0.0


def run_v7_backtest(dataset: pd.DataFrame, settings: V7Settings | None = None):
    settings = settings or V7Settings()
    yearly = {}
    for year in settings.test_years:
        v4_specs, _ = learn_factor_specs(dataset, year, V4Settings())
        yearly[year] = (fit_multihorizon_models(dataset, year, settings), fit_v5_models(dataset, year), v4_specs)
    eligible = dataset[dataset["eligible"] & dataset["future_return_5"].notna()].copy()
    eligible["date"] = pd.to_datetime(eligible["date"])
    test = eligible[eligible["date"].dt.year.isin(settings.test_years)]
    dates = test["date"].drop_duplicates().sort_values().reset_index(drop=True)
    previous: dict[str, float] = {}
    rows, signals, sector_ics = [], [], []
    buy_rate = settings.fee_rate + settings.slippage
    sell_rate = settings.fee_rate + settings.slippage + settings.stamp_duty
    for date in dates.iloc[:: settings.rebalance_every]:
        year = int(date.year)
        mh_models, v5_models, v4_specs = yearly[year]
        current = score_multihorizon(test[test["date"] == date], mh_models, settings)
        v6 = score_v6(current, v5_models, v4_specs, V6Settings())
        current["v6_score"] = v6["score"]
        current["model_score"] = (
            settings.multihorizon_share * current["multihorizon_score"]
            + settings.v6_share * current["v6_score"]
            - settings.uncertainty_penalty * current["horizon_uncertainty"]
        )
        current["score"] = current["model_score"] + current["symbol"].isin(previous) * settings.holding_bonus
        selected = select_sector_balanced(current, V6Settings(top_n=settings.top_n, min_positions=settings.min_positions))
        continuing = selected["symbol"].isin(previous)
        selected["holding_return"] = selected["execution_exit_open"] / selected["entry_open"] - 1
        selected["realized_return"] = selected["execution_return"].where(~continuing, selected["holding_return"])
        selected["executed"] = (continuing | selected["entry_tradable"].fillna(False)) & selected["realized_return"].notna()
        selected["weight"] = selected["weight"].where(selected["executed"], 0.0)
        weights = {row.symbol: float(row.weight) for row in selected.itertuples() if row.weight > 0}
        buys, sells = turnover(previous, weights)
        cost = buys * buy_rate + sells * sell_rate
        gross = float((selected["weight"] * selected["realized_return"].fillna(0)).sum())
        ic = current["model_score"].corr(current["label_5"], method="spearman")
        rows.append({"date": date, "test_year": year, "period_return": gross - cost, "benchmark_return": float(current["future_return_5"].mean()), "rank_ic": ic, "holdings": len(weights), "cash_weight": 1 - sum(weights.values()), "buy_turnover": buys, "sell_turnover": sells, "transaction_cost": cost, "mean_uncertainty": float(current["horizon_uncertainty"].mean()), "mean_agreement": float(current["horizon_agreement"].mean())})
        for sector, group in current.groupby("broad_sector"):
            value = group["model_score"].corr(group["label_5"], method="spearman")
            if len(group) >= 10 and pd.notna(value):
                sector_ics.append({"date": date, "test_year": year, "broad_sector": sector, "rank_ic": value})
        for rank, row in enumerate(selected.sort_values("model_score", ascending=False).itertuples(), 1):
            signals.append({"date": date, "test_year": year, "rank": rank, "symbol": row.symbol, "broad_sector": row.broad_sector, "score": row.model_score, "uncertainty": row.horizon_uncertainty, "agreement": row.horizon_agreement, "weight": row.weight})
        previous = weights
    return pd.DataFrame(rows), pd.DataFrame(signals), pd.DataFrame(sector_ics)
