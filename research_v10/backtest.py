from __future__ import annotations

import numpy as np
import pandas as pd

from research_v4.config import V4Settings
from research_v4.stability import learn_factor_specs
from research_v5.models import fit_v5_models
from research_v6.config import V6Settings
from research_v6.model import score_v6
from stockpilot.portfolio import turnover

from .model import fit_v10_models, score_v10
from .portfolio import benchmark_weights, optimize_benchmark_relative
from .research_config import V10Settings


MODES = ("core", "v6_relative", "v10_ridge", "v10_global", "v10_full")


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min()) if not equity.empty else 0.0


def _execute(
    current: pd.DataFrame,
    desired: dict[str, float],
    previous: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    lookup = current.drop_duplicates("symbol").set_index("symbol")
    executed = {}
    realized = {}
    for symbol, weight in desired.items():
        if symbol not in lookup.index:
            continue
        row = lookup.loc[symbol]
        continuing = symbol in previous
        value = (
            float(row["execution_exit_open_20"] / row["entry_open_20"] - 1)
            if continuing
            else float(row["execution_return_20"])
        )
        if (continuing or bool(row["entry_tradable_20"])) and np.isfinite(value):
            executed[symbol] = float(weight)
            realized[symbol] = value
    return executed, realized


def run_v10_backtest(
    dataset: pd.DataFrame, settings: V10Settings | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    settings = settings or V10Settings()
    yearly = {}
    model_diagnostics = []
    for year in settings.test_years:
        v5_models = fit_v5_models(dataset, year)
        v4_specs, _ = learn_factor_specs(dataset, year, V4Settings())
        v10_models = fit_v10_models(dataset, year, settings)
        yearly[year] = (v5_models, v4_specs, v10_models)
        for validation_year, values in v10_models.validation_year_ics.items():
            model_diagnostics.append(
                {
                    "test_year": year,
                    "validation_year": validation_year,
                    **values,
                    "technology_enabled": v10_models.technology_enabled,
                    "confidence": v10_models.confidence,
                }
            )

    scope = dataset[
        dataset["in_universe"].fillna(False)
        & dataset["future_return_20"].notna()
        & pd.to_datetime(dataset["date"]).dt.year.isin(settings.test_years)
    ].copy()
    scope["date"] = pd.to_datetime(scope["date"])
    dates = scope["date"].drop_duplicates().sort_values().reset_index(drop=True)
    previous = {mode: {} for mode in MODES}
    previous_active = {mode: set() for mode in MODES}
    rows, signals, sector_ics = [], [], []
    buy_rate = settings.fee_rate + settings.slippage
    sell_rate = settings.fee_rate + settings.slippage + settings.stamp_duty

    for date in dates.iloc[:: settings.rebalance_every]:
        year = int(date.year)
        v5_models, v4_specs, models = yearly[year]
        raw = scope[scope["date"] == date].copy()
        v10 = score_v10(raw, models, v5_models, v4_specs, settings)
        v6 = score_v6(raw, v5_models, v4_specs, V6Settings())
        mode_scores = {
            "v6_relative": v6["score"],
            "v10_ridge": v10["ridge_model_score"],
            "v10_global": v10["global_model_score"],
            "v10_full": v10["model_score"],
        }
        base = benchmark_weights(v10)
        benchmark_return = float(
            sum(base.get(row.symbol, 0.0) * float(row.future_return_20) for row in v10.itertuples())
        )
        for mode in MODES:
            current = v10.copy()
            if mode == "core":
                desired, active, diagnostics = base, set(), {
                    "active_budget": 0.0,
                    "ex_ante_tracking_error": 0.0,
                    "maximum_stock_active_weight": 0.0,
                    "maximum_sector_deviation": 0.0,
                    "active_holdings": 0,
                }
                current["portfolio_score"] = 0.0
            else:
                current["portfolio_score"] = mode_scores[mode]
                desired, active, diagnostics = optimize_benchmark_relative(
                    current,
                    previous_active[mode],
                    1.0 if mode == "v6_relative" else models.confidence,
                    True if mode == "v6_relative" else models.technology_enabled,
                    settings,
                )
            executed, realized = _execute(current, desired, previous[mode])
            buys, sells = turnover(previous[mode], executed)
            cost = buys * buy_rate + sells * sell_rate
            gross = float(sum(weight * realized[symbol] for symbol, weight in executed.items()))
            evaluation = current[current["eligible"] & current["label_5"].notna()]
            ic_5 = (
                evaluation["portfolio_score"].corr(evaluation["label_5"], method="spearman")
                if mode != "core"
                else np.nan
            )
            ic_20 = (
                evaluation["portfolio_score"].corr(
                    evaluation["v10_target_20"], method="spearman"
                )
                if mode != "core"
                else np.nan
            )
            truth_top = set(
                evaluation.nlargest(settings.active_top_n, "v10_target_20")["symbol"]
            )
            precision = len(active & truth_top) / settings.active_top_n if mode != "core" else np.nan
            selected = current[current["symbol"].isin(active)]
            selected_excess = (
                float(selected["future_return_20"].mean() - benchmark_return)
                if not selected.empty
                else np.nan
            )
            rows.append(
                {
                    "date": date,
                    "test_year": year,
                    "mode": mode,
                    "period_return": gross - cost,
                    "benchmark_return": benchmark_return,
                    "excess_period_return": gross - cost - benchmark_return,
                    "rank_ic_5": float(ic_5) if pd.notna(ic_5) else np.nan,
                    "rank_ic_20": float(ic_20) if pd.notna(ic_20) else np.nan,
                    "top30_precision": precision,
                    "selected_excess_return": selected_excess,
                    "buy_turnover": buys,
                    "sell_turnover": sells,
                    "transaction_cost": cost,
                    "cash_weight": 1 - sum(executed.values()),
                    "confidence": models.confidence if mode != "core" else 0.0,
                    "technology_enabled": models.technology_enabled if mode != "core" else False,
                    **diagnostics,
                }
            )
            if mode != "core":
                for sector, group in evaluation.groupby("broad_sector"):
                    value_5 = group["portfolio_score"].corr(group["label_5"], method="spearman")
                    value_20 = group["portfolio_score"].corr(
                        group["v10_target_20"], method="spearman"
                    )
                    if len(group) >= 10 and (pd.notna(value_5) or pd.notna(value_20)):
                        sector_ics.append(
                            {
                                "date": date,
                                "test_year": year,
                                "mode": mode,
                                "broad_sector": sector,
                                "rank_ic_5": value_5,
                                "rank_ic_20": value_20,
                            }
                        )
                for rank, row in enumerate(
                    selected.sort_values("portfolio_score", ascending=False).itertuples(), 1
                ):
                    signals.append(
                        {
                            "date": date,
                            "test_year": year,
                            "mode": mode,
                            "rank": rank,
                            "symbol": row.symbol,
                            "broad_sector": row.broad_sector,
                            "technology_subsector": row.technology_subsector,
                            "score": row.portfolio_score,
                            "benchmark_weight": base.get(row.symbol, 0.0),
                            "target_weight": desired.get(row.symbol, 0.0),
                        }
                    )
            previous[mode] = executed
            previous_active[mode] = active
    return (
        pd.DataFrame(rows),
        pd.DataFrame(signals),
        pd.DataFrame(sector_ics),
        pd.DataFrame(model_diagnostics),
    )

