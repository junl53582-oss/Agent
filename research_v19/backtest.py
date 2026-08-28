from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from research_v10.backtest import _execute
from research_v10.portfolio import benchmark_weights
from research_v16.model import fit_v16_models, score_v16
from research_v16.portfolio import optimize_v16
from stockpilot.portfolio import turnover

from .config import V19Settings
from .model import apply_v19_weights, regime_name


MODES = ("v16_ungated", "v19_adaptive")


def record_progress(settings, stage, **values):
    report = {"stage": stage, "at_utc": datetime.now(timezone.utc).isoformat(), **values}
    (settings.artifact_dir / "runtime_status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def max_drawdown(returns):
    equity = pd.concat([pd.Series([1.0]), (1 + returns).cumprod()], ignore_index=True)
    return float((equity / equity.cummax() - 1).min())


def run_v19_backtest(
    dataset: pd.DataFrame,
    corpus,
    settings: V19Settings | None = None,
):
    settings = settings or V19Settings()
    yearly, baseline_cache = {}, {}
    for year in settings.test_years:
        record_progress(settings, "fitting", test_year=int(year))
        print(f"V19 fitting test_year={year}", flush=True)
        models = fit_v16_models(dataset, corpus, year, settings, baseline_cache)
        _, v5_models, v4_specs = baseline_cache[year]
        yearly[year] = (v5_models, v4_specs, models)

    scope = dataset[
        dataset["in_universe"].fillna(False)
        & dataset["future_return_20"].notna()
        & pd.to_datetime(dataset["date"]).dt.year.isin(settings.test_years)
    ].copy()
    scope["date"] = pd.to_datetime(scope["date"])
    dates = scope["date"].drop_duplicates().sort_values().reset_index(drop=True)
    previous = {mode: {} for mode in MODES}
    previous_active = {mode: set() for mode in MODES}
    rows = []
    buy_rate = settings.fee_rate + settings.slippage
    sell_rate = settings.fee_rate + settings.slippage + settings.stamp_duty
    benchmark_history = deque(maxlen=1)

    for period_index, date in enumerate(dates.iloc[:: settings.rebalance_every]):
        if period_index % 10 == 0:
            record_progress(settings, "portfolio_evaluation", period=period_index, date=str(date.date()))
            print(f"V19 portfolio period={period_index} date={date.date()}", flush=True)
        year = int(date.year)
        v5_models, v4_specs, models = yearly[year]
        raw = scope[scope["date"] == date].copy()
        scored = score_v16(raw, models, v5_models, v4_specs, settings)
        base = benchmark_weights(scored)
        benchmark_return = float(
            sum(base.get(row.symbol, 0.0) * float(row.future_return_20) for row in scored.itertuples())
        )
        momentum = float(benchmark_history[-1]) if benchmark_history else 0.0
        adaptive = apply_v19_weights(scored, momentum, settings)

        for mode in MODES:
            frame = adaptive if mode == "v19_adaptive" else scored
            frame = frame.copy()
            frame["portfolio_score"] = (
                frame["v19_score"] if mode == "v19_adaptive" else frame["v16_score"]
            )
            desired, active, diagnostics = optimize_v16(
                frame, previous_active[mode], True, True, settings
            )
            executed, realized = _execute(frame, desired, previous[mode])
            buys, sells = turnover(previous[mode], executed)
            cost = buys * buy_rate + sells * sell_rate
            gross = float(sum(weight * realized[symbol] for symbol, weight in executed.items()))
            evaluation = frame[frame["eligible"] & frame["label_5"].notna()]
            ic5 = evaluation["portfolio_score"].corr(evaluation["label_5"], method="spearman")
            ic20 = evaluation["portfolio_score"].corr(evaluation["v10_target_20"], method="spearman")
            selected = frame[frame["symbol"].isin(active)]
            rows.append({
                "date": date,
                "test_year": year,
                "mode": mode,
                "period_return": gross - cost,
                "benchmark_return": benchmark_return,
                "excess_period_return": gross - cost - benchmark_return,
                "rank_ic_5": float(ic5) if pd.notna(ic5) else np.nan,
                "rank_ic_20": float(ic20) if pd.notna(ic20) else np.nan,
                "market_regime": regime_name(momentum, settings),
                "buy_turnover": buys,
                "sell_turnover": sells,
                "transaction_cost": cost,
                "cash_weight": 1 - sum(executed.values()),
                **diagnostics,
            })
            previous[mode] = executed
            previous_active[mode] = active

        benchmark_history.append(benchmark_return)

    record_progress(settings, "backtest_complete", periods=len(rows) // len(MODES))
    return pd.DataFrame(rows)
