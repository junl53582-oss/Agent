from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from stockpilot.portfolio import portfolio_weights, select_with_buffer_and_cap, turnover

from .config import V4Settings
from .stability import FactorSpec, learn_factor_specs, score_with_specs


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min()) if not equity.empty else 0.0


def _serialize_specs(test_year: int, specs: list[FactorSpec]) -> list[dict]:
    rows = []
    for spec in specs:
        row = asdict(spec)
        row["test_year"] = test_year
        row["training_years"] = ",".join(map(str, spec.training_years))
        rows.append(row)
    return rows


def run_v4_backtest(
    dataset: pd.DataFrame, settings: V4Settings | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    settings = settings or V4Settings()
    specs_by_year: dict[int, list[FactorSpec]] = {}
    spec_rows: list[dict] = []
    diagnostic_frames = []
    for year in settings.test_years:
        specs, diagnostics = learn_factor_specs(dataset, year, settings)
        specs_by_year[year] = specs
        spec_rows.extend(_serialize_specs(year, specs))
        diagnostic_frames.append(diagnostics)

    eligible = dataset[dataset["eligible"] & dataset["future_return_5"].notna()].copy()
    eligible["date"] = pd.to_datetime(eligible["date"])
    test = eligible[eligible["date"].dt.year.isin(settings.test_years)]
    dates = test["date"].drop_duplicates().sort_values().reset_index(drop=True)
    rebalance_dates = dates.iloc[:: settings.rebalance_every]
    previous_weights: dict[str, float] = {}
    rows = []
    signals = []
    buy_rate = settings.fee_rate + settings.slippage
    sell_rate = settings.fee_rate + settings.slippage + settings.stamp_duty
    for date in rebalance_dates:
        current = test[test["date"] == date].copy()
        specs = specs_by_year[int(date.year)]
        current["score"] = score_with_specs(current, specs)
        current["pred_rank"] = current["score"].rank(ascending=False, method="first")
        active = any(spec.selected for spec in specs)
        selected = select_with_buffer_and_cap(
            current,
            settings.top_n,
            set(previous_weights),
            settings.hold_buffer,
            settings.industry_cap,
        ) if active else current.iloc[0:0].copy()
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
        rank_ic = current["score"].corr(current["label_5"], method="spearman") if active else np.nan
        rows.append(
            {
                "date": date,
                "test_year": int(date.year),
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
        for rank, row in enumerate(selected.sort_values("score", ascending=False).itertuples(), 1):
            signals.append(
                {
                    "date": date,
                    "test_year": int(date.year),
                    "rank": rank,
                    "symbol": row.symbol,
                    "score": row.score,
                    "weight": row.weight,
                    "executed": bool(row.executed),
                }
            )
        previous_weights = weights
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    return pd.DataFrame(rows), pd.DataFrame(spec_rows), diagnostics, pd.DataFrame(signals)
