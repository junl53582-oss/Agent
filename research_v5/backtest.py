from __future__ import annotations

import numpy as np
import pandas as pd

from stockpilot.portfolio import portfolio_weights, select_with_buffer_and_cap, turnover

from .config import V5Settings
from .models import fit_v5_models, model_diagnostics, score_v5


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min()) if not equity.empty else 0.0


def run_v5_backtest(
    dataset: pd.DataFrame, settings: V5Settings | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    settings = settings or V5Settings()
    models_by_year = {}
    diagnostics = []
    for year in settings.test_years:
        models = fit_v5_models(dataset, year, settings)
        models_by_year[year] = models
        diagnostics.extend(model_diagnostics(year, models))
    eligible = dataset[dataset["eligible"] & dataset["future_return_5"].notna()].copy()
    eligible["date"] = pd.to_datetime(eligible["date"])
    test = eligible[eligible["date"].dt.year.isin(settings.test_years)]
    dates = test["date"].drop_duplicates().sort_values().reset_index(drop=True)
    rebalance_dates = dates.iloc[:: settings.rebalance_every]
    previous_weights: dict[str, float] = {}
    rows, signals, sector_ics = [], [], []
    buy_rate = settings.fee_rate + settings.slippage
    sell_rate = settings.fee_rate + settings.slippage + settings.stamp_duty
    for date in rebalance_dates:
        current = score_v5(test[test["date"] == date], models_by_year[int(date.year)])
        current["pred_rank"] = current["score"].rank(ascending=False, method="first")
        selected = select_with_buffer_and_cap(
            current,
            settings.top_n,
            set(previous_weights),
            settings.hold_buffer,
            settings.industry_cap,
        )
        if len(selected) < settings.min_positions:
            selected = selected.iloc[0:0].copy()
        selected["planned_weight"] = portfolio_weights(selected, "inverse_volatility")
        continuing = selected["symbol"].isin(previous_weights)
        selected["holding_return"] = selected["execution_exit_open"] / selected["entry_open"] - 1
        selected["realized_return"] = selected["execution_return"].where(
            ~continuing, selected["holding_return"]
        )
        selected["executed"] = (
            continuing | selected["entry_tradable"].fillna(False)
        ) & selected["realized_return"].notna()
        selected["weight"] = selected["planned_weight"].where(selected["executed"], 0.0)
        weights = {
            row.symbol: float(row.weight) for row in selected.itertuples() if row.weight > 0
        }
        buys, sells = turnover(previous_weights, weights)
        cost = buys * buy_rate + sells * sell_rate
        gross = float((selected["weight"] * selected["realized_return"].fillna(0)).sum())
        rank_ic = current["score"].corr(current["label_5"], method="spearman")
        rows.append(
            {
                "date": date,
                "test_year": int(date.year),
                "regime": current["regime"].iloc[0],
                "period_return": gross - cost,
                "benchmark_return": float(current["future_return_5"].mean()),
                "rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
                "holdings": len(weights),
                "cash_weight": 1 - sum(weights.values()),
                "buy_turnover": buys,
                "sell_turnover": sells,
                "transaction_cost": cost,
            }
        )
        for sector, group in current.groupby("broad_sector"):
            value = group["score"].corr(group["label_5"], method="spearman")
            if len(group) >= 10 and pd.notna(value):
                sector_ics.append(
                    {"date": date, "test_year": int(date.year), "broad_sector": sector, "rank_ic": value, "stocks": len(group)}
                )
        for rank, row in enumerate(selected.sort_values("score", ascending=False).itertuples(), 1):
            signals.append(
                {
                    "date": date,
                    "test_year": int(date.year),
                    "rank": rank,
                    "symbol": row.symbol,
                    "broad_sector": row.broad_sector,
                    "regime": row.regime,
                    "score": row.score,
                    "weight": row.weight,
                    "executed": bool(row.executed),
                }
            )
        previous_weights = weights
    return (
        pd.DataFrame(rows),
        pd.DataFrame(signals),
        pd.DataFrame(sector_ics),
        pd.DataFrame(diagnostics),
    )
