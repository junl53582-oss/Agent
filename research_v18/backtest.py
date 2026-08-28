from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from research_v10.backtest import _execute
from research_v10.portfolio import benchmark_weights
from research_v16.portfolio import optimize_v16
from stockpilot.portfolio import turnover

from .config import V18Settings
from .model import fit_v18_models, score_v18


MODES = ("v13_comparable", "v18_text_ungated")


def record_progress(settings, stage, **values):
    report = {"stage": stage, "at_utc": datetime.now(timezone.utc).isoformat(), **values}
    (settings.artifact_dir / "runtime_status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def max_drawdown(returns):
    equity = pd.concat([pd.Series([1.0]), (1 + returns).cumprod()], ignore_index=True)
    return float((equity / equity.cummax() - 1).min())


def run_v18_backtest(
    dataset: pd.DataFrame,
    events: pd.DataFrame,
    embeddings: np.ndarray,
    settings: V18Settings | None = None,
):
    settings = settings or V18Settings()
    yearly, baseline_cache = {}, {}
    for year in settings.test_years:
        record_progress(settings, "fitting", test_year=int(year))
        print(f"V18 fitting test_year={year}", flush=True)
        models = fit_v18_models(dataset, events, embeddings, year, settings, baseline_cache)
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
    rows, signals, sector_ics = [], [], []
    buy_rate = settings.fee_rate + settings.slippage
    sell_rate = settings.fee_rate + settings.slippage + settings.stamp_duty
    for period_index, date in enumerate(dates.iloc[:: settings.rebalance_every]):
        if period_index % 10 == 0:
            record_progress(settings, "portfolio_evaluation", period=period_index, date=str(date.date()))
            print(f"V18 portfolio period={period_index} date={date.date()}", flush=True)
        year = int(date.year)
        v5_models, v4_specs, models = yearly[year]
        current = score_v18(
            scope[scope["date"] == date].copy(), models, v5_models, v4_specs, settings
        )
        base = benchmark_weights(current)
        benchmark_return = float(
            sum(base.get(row.symbol, 0.0) * float(row.future_return_20) for row in current.itertuples())
        )
        mode_scores = {
            "v13_comparable": current["v13_comparable_score"],
            "v18_text_ungated": current["v18_score"],
        }
        for mode in MODES:
            frame = current.copy()
            frame["portfolio_score"] = mode_scores[mode]
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
            selected_excess = float(selected["future_return_20"].mean() - benchmark_return) if not selected.empty else np.nan
            rows.append({
                "date": date,
                "test_year": year,
                "mode": mode,
                "period_return": gross - cost,
                "benchmark_return": benchmark_return,
                "excess_period_return": gross - cost - benchmark_return,
                "rank_ic_5": float(ic5) if pd.notna(ic5) else np.nan,
                "rank_ic_20": float(ic20) if pd.notna(ic20) else np.nan,
                "selected_excess_return": selected_excess,
                "buy_turnover": buys,
                "sell_turnover": sells,
                "transaction_cost": cost,
                "cash_weight": 1 - sum(executed.values()),
                "training_text_events": models.training_events,
                "raw_event_years": len(models.raw_event_years),
                **diagnostics,
            })
            if mode == "v18_text_ungated":
                for sector, group in evaluation.groupby("broad_sector"):
                    value5 = group["portfolio_score"].corr(group["label_5"], method="spearman")
                    value20 = group["portfolio_score"].corr(group["v10_target_20"], method="spearman")
                    if len(group) >= 10 and (pd.notna(value5) or pd.notna(value20)):
                        sector_ics.append({
                            "date": date, "test_year": year, "mode": mode,
                            "broad_sector": sector, "rank_ic_5": value5, "rank_ic_20": value20,
                        })
                for rank, row in enumerate(selected.sort_values("portfolio_score", ascending=False).itertuples(), 1):
                    signals.append({
                        "date": date, "test_year": year, "mode": mode, "rank": rank,
                        "symbol": row.symbol, "broad_sector": row.broad_sector,
                        "score": row.portfolio_score, "text_event_score": row.text_event_score,
                        "recent_text_events": row.recent_text_events,
                        "benchmark_weight": base.get(row.symbol, 0.0),
                        "target_weight": desired.get(row.symbol, 0.0),
                    })
            previous[mode] = executed
            previous_active[mode] = active
    record_progress(settings, "backtest_complete", periods=len(rows) // len(MODES))
    return (
        pd.DataFrame(rows),
        pd.DataFrame(signals),
        pd.DataFrame(sector_ics),
    )
