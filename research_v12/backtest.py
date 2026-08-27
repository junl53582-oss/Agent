from __future__ import annotations

import numpy as np
import pandas as pd

from research_v10.backtest import _execute, max_drawdown
from research_v10.portfolio import benchmark_weights, optimize_benchmark_relative
from research_v10.research_config import V10Settings
from research_v4.config import V4Settings
from research_v4.stability import learn_factor_specs
from research_v5.models import fit_v5_models
from stockpilot.portfolio import turnover

from .config import V12Settings
from .model import fit_v12_models, score_v12
from .portfolio import apply_exposure, optimize_v12
from .risk import risk_budget_exposure


MODES = (
    "core",
    "v10_global",
    "v12_portfolio_ungated",
    "v12_portfolio_gated",
    "v12_risk_budget",
)


def run_v12_backtest(dataset: pd.DataFrame, settings: V12Settings | None = None):
    settings = settings or V12Settings()
    yearly = {}
    validation_rows = []
    for year in settings.test_years:
        v5_models = fit_v5_models(dataset, year)
        v4_specs, _ = learn_factor_specs(dataset, year, V4Settings())
        models = fit_v12_models(dataset, year, settings)
        yearly[year] = (v5_models, v4_specs, models)
        for validation_year, values in models.validation_diagnostics.items():
            validation_rows.append(
                {
                    "test_year": year,
                    "validation_year": validation_year,
                    **values,
                    "global_gate": models.global_gate,
                    "technology_gate": models.technology_gate,
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
        current = score_v12(
            scope[scope["date"] == date].copy(), models, v5_models, v4_specs, settings
        )
        base = benchmark_weights(current)
        benchmark_return = float(
            sum(base.get(row.symbol, 0.0) * float(row.future_return_20) for row in current.itertuples())
        )
        exposure, risk_diagnostics = risk_budget_exposure(current, settings)
        for mode in MODES:
            frame = current.copy()
            frame["portfolio_score"] = (
                frame["global_model_score"] if mode == "v10_global" else frame["v12_score"]
            )
            if mode == "core":
                desired, active, diagnostics = base, set(), {
                    "active_budget": 0.0,
                    "ex_ante_tracking_error": 0.0,
                    "maximum_stock_active_weight": 0.0,
                    "maximum_sector_deviation": 0.0,
                    "active_holdings": 0,
                }
            elif mode == "v10_global":
                desired, active, diagnostics = optimize_benchmark_relative(
                    frame,
                    previous_active[mode],
                    models.v10.confidence,
                    models.v10.technology_enabled,
                    V10Settings(),
                )
            else:
                global_gate = mode == "v12_portfolio_ungated" or models.global_gate
                technology_gate = mode == "v12_portfolio_ungated" or models.technology_gate
                desired, active, diagnostics = optimize_v12(
                    frame,
                    previous_active[mode],
                    global_gate,
                    technology_gate,
                    settings,
                )
                if mode == "v12_risk_budget":
                    desired = apply_exposure(desired, exposure)
            executed, realized = _execute(frame, desired, previous[mode])
            buys, sells = turnover(previous[mode], executed)
            cost = buys * buy_rate + sells * sell_rate
            gross = float(sum(weight * realized[symbol] for symbol, weight in executed.items()))
            evaluation = frame[frame["eligible"] & frame["label_5"].notna()]
            ic_5 = evaluation["portfolio_score"].corr(evaluation["label_5"], method="spearman") if mode != "core" else np.nan
            ic_20 = evaluation["portfolio_score"].corr(evaluation["v10_target_20"], method="spearman") if mode != "core" else np.nan
            truth_top = set(evaluation.nlargest(settings.active_top_n, "v12_net_marginal_target")["symbol"])
            precision = len(active & truth_top) / settings.active_top_n if mode != "core" and active else np.nan
            selected = frame[frame["symbol"].isin(active)]
            selected_excess = float(selected["future_return_20"].mean() - benchmark_return) if not selected.empty else np.nan
            selected_net_marginal = float(selected["v12_net_marginal_target"].mean()) if not selected.empty else np.nan
            mode_exposure = exposure if mode == "v12_risk_budget" else 1.0
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
                    "selected_net_marginal": selected_net_marginal,
                    "buy_turnover": buys,
                    "sell_turnover": sells,
                    "transaction_cost": cost,
                    "cash_weight": 1 - sum(executed.values()),
                    "equity_exposure": mode_exposure,
                    "global_gate": models.global_gate if mode.startswith("v12") else False,
                    "technology_gate": models.technology_gate if mode.startswith("v12") else False,
                    **risk_diagnostics,
                    **diagnostics,
                }
            )
            if mode != "core":
                for sector, group in evaluation.groupby("broad_sector"):
                    value_5 = group["portfolio_score"].corr(group["label_5"], method="spearman")
                    value_20 = group["portfolio_score"].corr(group["v10_target_20"], method="spearman")
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
                for rank, row in enumerate(selected.sort_values("portfolio_score", ascending=False).itertuples(), 1):
                    signals.append(
                        {
                            "date": date,
                            "test_year": year,
                            "mode": mode,
                            "rank": rank,
                            "symbol": row.symbol,
                            "broad_sector": row.broad_sector,
                            "score": row.portfolio_score,
                            "benchmark_weight": base.get(row.symbol, 0.0),
                            "target_weight": desired.get(row.symbol, 0.0),
                            "equity_exposure": mode_exposure,
                        }
                    )
            previous[mode] = executed
            previous_active[mode] = active
    return pd.DataFrame(rows), pd.DataFrame(signals), pd.DataFrame(sector_ics), pd.DataFrame(validation_rows)

