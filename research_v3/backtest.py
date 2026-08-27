from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockpilot.portfolio import portfolio_weights, select_with_buffer_and_cap, turnover

from .config import V3Settings
from .models import fit_v3_models, score_v3_models


@dataclass(frozen=True)
class V3Candidate:
    name: str
    score_column: str
    agreement_threshold: float = 0.0


CANDIDATES = [
    V3Candidate("stable_factors", "stable_factor_score"),
    V3Candidate("ridge_multi_horizon", "ridge_score", 0.45),
    V3Candidate("lightgbm_multi_horizon", "lightgbm_score", 0.45),
    V3Candidate("ensemble_agree_45", "ensemble_score", 0.45),
    V3Candidate("ensemble_agree_67", "ensemble_score", 2 / 3),
    V3Candidate("ensemble_agree_75", "ensemble_score", 0.75),
]


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min()) if not equity.empty else 0.0


def run_v3_backtest(
    dataset: pd.DataFrame, settings: V3Settings | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    settings = settings or V3Settings()
    complete = dataset[dataset["eligible"] & dataset["future_return_5"].notna()]
    dates = complete["date"].drop_duplicates().sort_values().reset_index(drop=True)
    rebalance_dates = dates.iloc[settings.min_train_days :: settings.rebalance_every]
    if rebalance_dates.empty:
        raise ValueError("历史不足，无法运行V3走步回测")
    previous_weights = {candidate.name: {} for candidate in CANDIDATES}
    rows: list[dict] = []
    signals: list[dict] = []
    models = None
    last_train = -10_000
    buy_rate = settings.fee_rate + settings.slippage
    sell_rate = settings.fee_rate + settings.slippage + settings.stamp_duty
    positions = {date: index for index, date in enumerate(dates)}
    for date in rebalance_dates:
        position = positions[date]
        if models is None or position - last_train >= settings.retrain_every:
            models = fit_v3_models(dataset, date, settings.horizons, settings.train_window_days)
            last_train = position
        current = complete[complete["date"] == date].copy()
        scored = score_v3_models(current, models, settings.horizons)
        benchmark_return = float(scored["future_return_5"].mean())
        for candidate in CANDIDATES:
            eligible = scored[scored["agreement"] >= candidate.agreement_threshold].copy()
            eligible["score"] = eligible[candidate.score_column]
            eligible["pred_rank"] = eligible["score"].rank(ascending=False, method="first")
            selected = select_with_buffer_and_cap(
                eligible,
                settings.top_n,
                set(previous_weights[candidate.name]),
                settings.hold_buffer,
                settings.industry_cap,
            )
            if len(selected) < settings.min_positions:
                selected = selected.iloc[0:0].copy()
            selected["planned_weight"] = portfolio_weights(selected, "inverse_volatility")
            continuing = selected["symbol"].isin(previous_weights[candidate.name])
            selected["holding_return"] = (
                selected["execution_exit_open"] / selected["entry_open"] - 1
            )
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
            buys, sells = turnover(previous_weights[candidate.name], weights)
            cost = buys * buy_rate + sells * sell_rate
            gross = float((selected["weight"] * selected["realized_return"].fillna(0)).sum())
            rank_ic = float(
                scored[candidate.score_column].corr(scored["label_5"], method="spearman")
            )
            rows.append(
                {
                    "date": date,
                    "candidate": candidate.name,
                    "period_return": gross - cost,
                    "benchmark_return": benchmark_return,
                    "rank_ic": rank_ic,
                    "holdings": len(weights),
                    "cash_weight": 1 - sum(weights.values()),
                    "buy_turnover": buys,
                    "sell_turnover": sells,
                    "transaction_cost": cost,
                }
            )
            for rank, row in enumerate(
                selected.sort_values("score", ascending=False).itertuples(), 1
            ):
                signals.append(
                    {
                        "date": date,
                        "candidate": candidate.name,
                        "rank": rank,
                        "symbol": row.symbol,
                        "score": row.score,
                        "agreement": row.agreement,
                        "weight": row.weight,
                        "executed": bool(row.executed),
                    }
                )
            previous_weights[candidate.name] = weights
    equity = pd.DataFrame(rows)
    equity["excess_return"] = equity["period_return"] - equity["benchmark_return"]
    metrics = []
    periods_per_year = 252 / settings.rebalance_every
    for name, group in equity.groupby("candidate"):
        total = float((1 + group["period_return"]).prod() - 1)
        benchmark = float((1 + group["benchmark_return"]).prod() - 1)
        std = group["period_return"].std(ddof=1)
        metrics.append(
            {
                "candidate": name,
                "periods": len(group),
                "total_return": total,
                "benchmark_return": benchmark,
                "excess_return": total - benchmark,
                "sharpe": float(group["period_return"].mean() / std * np.sqrt(periods_per_year))
                if std > 0
                else 0.0,
                "max_drawdown": _max_drawdown(group["period_return"]),
                "mean_rank_ic": float(group["rank_ic"].mean()),
                "average_cash_weight": float(group["cash_weight"].mean()),
                "average_one_way_turnover": float(
                    (group["buy_turnover"] + group["sell_turnover"]).mean() / 2
                ),
            }
        )
    return (
        equity,
        pd.DataFrame(metrics).sort_values("excess_return", ascending=False),
        pd.DataFrame(signals),
    )
