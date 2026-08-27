from __future__ import annotations

import numpy as np
import pandas as pd

from research_v4.config import V4Settings
from research_v4.stability import learn_factor_specs
from research_v5.models import fit_v5_models
from research_v6.config import V6Settings
from research_v6.model import score_v6, select_sector_balanced
from stockpilot.portfolio import turnover

from .config import V9Settings
from .model import fit_v9_models, score_v9


MODES = ("v6_portfolio", "v9_alpha", "v9_nonlinear", "v9_full")


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min()) if not equity.empty else 0.0


def _desired_weights(
    current: pd.DataFrame,
    previous_active: set[str],
    settings: V9Settings,
) -> tuple[dict[str, float], set[str]]:
    core = current[current["benchmark_weight"] > 0].copy()
    core_total = float(core["benchmark_weight"].sum())
    core_weights = (
        (core.set_index("symbol")["benchmark_weight"] / core_total * settings.core_share).to_dict()
        if core_total > 0
        else {}
    )
    candidates = current[current["eligible"]].copy()
    candidates["score"] = candidates["portfolio_score"] + (
        candidates["symbol"].isin(previous_active).astype(float) * settings.holding_bonus
    )
    selected = select_sector_balanced(
        candidates,
        V6Settings(top_n=settings.top_n, min_positions=settings.min_positions),
    )
    active = {
        str(row.symbol): float(row.weight) * settings.active_share
        for row in selected.itertuples()
        if row.weight > 0
    }
    desired = dict(core_weights)
    for symbol, weight in active.items():
        desired[symbol] = desired.get(symbol, 0.0) + weight
    return desired, set(active)


def _execute_weights(
    current: pd.DataFrame,
    desired: dict[str, float],
    previous: dict[str, float],
) -> tuple[dict[str, float], pd.Series]:
    by_symbol = current.set_index("symbol", drop=False)
    executed: dict[str, float] = {}
    # Portfolio weights are keyed by symbol, so realized returns must use the
    # same index.  A RangeIndex here silently aligns every return to zero.
    returns = pd.Series(0.0, index=current["symbol"].astype(str))
    for symbol, weight in desired.items():
        if symbol not in by_symbol.index:
            continue
        row = by_symbol.loc[symbol]
        continuing = symbol in previous
        tradable = continuing or bool(row["entry_tradable"])
        realized = (
            float(row["execution_exit_open"] / row["entry_open"] - 1)
            if continuing
            else float(row["execution_return"])
        )
        if tradable and np.isfinite(realized):
            executed[symbol] = weight
            returns.loc[symbol] = realized
    return executed, returns


def run_v9_backtest(
    dataset: pd.DataFrame, settings: V9Settings | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    settings = settings or V9Settings()
    yearly = {}
    for year in settings.test_years:
        v5_models = fit_v5_models(dataset, year)
        v4_specs, _ = learn_factor_specs(dataset, year, V4Settings())
        v9_models = fit_v9_models(dataset, year, settings)
        yearly[year] = (v5_models, v4_specs, v9_models)

    scope = dataset[
        dataset["in_universe"].fillna(False)
        & dataset["future_return_5"].notna()
        & pd.to_datetime(dataset["date"]).dt.year.isin(settings.test_years)
    ].copy()
    scope["date"] = pd.to_datetime(scope["date"])
    dates = scope["date"].drop_duplicates().sort_values().reset_index(drop=True)
    previous = {mode: {} for mode in MODES}
    previous_active = {mode: set() for mode in MODES}
    rows: list[dict] = []
    signals: list[dict] = []
    sector_ics: list[dict] = []
    buy_rate = settings.fee_rate + settings.slippage
    sell_rate = settings.fee_rate + settings.slippage + settings.stamp_duty

    for date in dates.iloc[:: settings.rebalance_every]:
        year = int(date.year)
        v5_models, v4_specs, v9_models = yearly[year]
        raw = scope[scope["date"] == date].copy()
        v9 = score_v9(raw, v9_models, v5_models, v4_specs, settings)
        v6 = score_v6(raw, v5_models, v4_specs, V6Settings())
        score_columns = {
            "v6_portfolio": v6["score"],
            "v9_alpha": v9["alpha_model_score"],
            "v9_nonlinear": v9["nonlinear_model_score"],
            "v9_full": v9["model_score"],
        }
        benchmark_return = float(
            (raw["benchmark_weight"] * raw["future_return_5"]).sum()
            / raw["benchmark_weight"].sum()
        )
        for mode in MODES:
            current = v9.copy()
            current["portfolio_score"] = score_columns[mode]
            desired, active_symbols = _desired_weights(
                current, previous_active[mode], settings
            )
            executed, realized = _execute_weights(current, desired, previous[mode])
            buys, sells = turnover(previous[mode], executed)
            cost = buys * buy_rate + sells * sell_rate
            gross = float(
                sum(weight * float(realized.get(symbol, 0.0)) for symbol, weight in executed.items())
            )
            evaluation = current[current["eligible"] & current["label_5"].notna()]
            rank_ic = evaluation["portfolio_score"].corr(
                evaluation["label_5"], method="spearman"
            )
            rows.append(
                {
                    "date": date,
                    "test_year": year,
                    "mode": mode,
                    "period_return": gross - cost,
                    "benchmark_return": benchmark_return,
                    "excess_period_return": gross - cost - benchmark_return,
                    "rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
                    "holdings": len(executed),
                    "active_holdings": len(active_symbols),
                    "cash_weight": 1 - sum(executed.values()),
                    "buy_turnover": buys,
                    "sell_turnover": sells,
                    "transaction_cost": cost,
                }
            )
            for sector, group in evaluation.groupby("broad_sector"):
                value = group["portfolio_score"].corr(group["label_5"], method="spearman")
                if len(group) >= 10 and pd.notna(value):
                    sector_ics.append(
                        {
                            "date": date,
                            "test_year": year,
                            "mode": mode,
                            "broad_sector": sector,
                            "rank_ic": float(value),
                        }
                    )
            for rank, row in enumerate(
                current[current["symbol"].isin(active_symbols)].sort_values(
                    "portfolio_score", ascending=False
                ).itertuples(),
                1,
            ):
                signals.append(
                    {
                        "date": date,
                        "test_year": year,
                        "mode": mode,
                        "rank": rank,
                        "symbol": row.symbol,
                        "broad_sector": row.broad_sector,
                        "score": row.portfolio_score,
                        "target_weight": desired.get(row.symbol, 0.0),
                    }
                )
            previous[mode] = executed
            previous_active[mode] = active_symbols
    return pd.DataFrame(rows), pd.DataFrame(signals), pd.DataFrame(sector_ics)
