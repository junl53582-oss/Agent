import numpy as np
import pandas as pd

from research_v10.backtest import _execute
from research_v10.portfolio import benchmark_weights
from research_v16.model import fit_v16_models, score_v16
from research_v16.portfolio import optimize_v16
from stockpilot.portfolio import turnover

from .config import V20Settings
from .timing import historical_market_state, weights_for_momentum


MODES = ("v16_control", "v20_adaptive", "v20_timing")


def run_backtest(dataset, corpus, settings=None, progress=None):
    settings = settings or V20Settings()
    progress = progress or (lambda *args, **kwargs: None)
    market = historical_market_state(dataset, settings)
    scope = dataset[
        dataset["in_universe"].eq(True)
        & pd.to_datetime(dataset["date"]).dt.year.isin(settings.test_years)
    ].copy()
    scope["date"] = pd.to_datetime(scope["date"])
    dates = scope["date"].drop_duplicates().sort_values().iloc[::settings.rebalance_every]
    previous = {mode: {} for mode in MODES}
    active_previous = {mode: set() for mode in MODES}
    cache, rows, decision_rows = {}, [], []
    # Fit/evaluate one outer year at a time instead of retaining all yearly models.
    for year in settings.test_years:
        progress("fitting", test_year=int(year))
        models = fit_v16_models(dataset, corpus, year, settings, cache)
        _, v5, v4 = cache[year]
        for date in dates[dates.dt.year.eq(year)]:
            progress("portfolio_evaluation", test_year=int(year), date=str(date.date()))
            state = market.loc[date]
            momentum = float(state["market_momentum"])
            baseline_weight, regime = weights_for_momentum(momentum, settings)
            frame = score_v16(scope[scope["date"].eq(date)].copy(), models, v5, v4, settings)
            base = benchmark_weights(frame)
            if frame.loc[frame["symbol"].isin(base), "future_return_20"].isna().any():
                raise ValueError(f"incomplete benchmark outcome at {date}")
            benchmark_return = sum(base.get(row.symbol, 0) * row.future_return_20 for row in frame.itertuples())
            for mode in MODES:
                current = frame.copy()
                current["portfolio_score"] = (
                    baseline_weight * current["v13_comparable_score"]
                    + (1 - baseline_weight) * current["text_event_score"]
                    if mode == "v20_adaptive" else current["v16_score"]
                )
                in_market = mode != "v20_timing" or momentum > settings.timing_threshold
                if in_market:
                    desired, active, diagnostics = optimize_v16(current, active_previous[mode], True, True, settings)
                else:
                    desired, active, diagnostics = {}, set(), {}
                # Actually liquidate/re-enter and charge turnover, never zero an
                # already net-of-cost equity curve as V17 did.
                executed, realized = _execute(current, desired, previous[mode])
                buys, sells = turnover(previous[mode], executed)
                cost = buys * (settings.fee_rate + settings.slippage) + sells * (
                    settings.fee_rate + settings.slippage + settings.stamp_duty
                )
                gross = sum(weight * realized[symbol] for symbol, weight in executed.items())
                evaluation = current[current["eligible"].eq(True) & current["label_5"].notna()]
                tech = evaluation[evaluation["broad_sector"].eq("technology")]
                ic5 = evaluation["portfolio_score"].corr(evaluation["label_5"], method="spearman")
                ic20 = evaluation["portfolio_score"].corr(evaluation["v10_target_20"], method="spearman")
                tech_ic = tech["portfolio_score"].corr(tech["label_5"], method="spearman") if len(tech) >= 3 else np.nan
                rows.append({
                    "date": date, "test_year": year, "mode": mode,
                    "period_return": gross - cost, "benchmark_return": benchmark_return,
                    "excess_period_return": gross - cost - benchmark_return,
                    "rank_ic_5": ic5, "rank_ic_20": ic20, "technology_rank_ic_5": tech_ic,
                    "in_market": bool(in_market), "market_momentum": momentum,
                    "market_data_end": state["market_data_end"], "market_regime": regime,
                    "baseline_weight": baseline_weight if mode == "v20_adaptive" else settings.baseline_share,
                    "buy_turnover": buys, "sell_turnover": sells, "transaction_cost": cost,
                    "cash_weight": 1 - sum(executed.values()), **diagnostics,
                })
                for symbol, weight in executed.items():
                    decision_rows.append({"date": date, "mode": mode, "symbol": symbol, "weight": weight})
                previous[mode], active_previous[mode] = executed, active
    return pd.DataFrame(rows), pd.DataFrame(decision_rows)
