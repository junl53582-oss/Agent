"""Unchanged frozen predictors with a common-calendar, persistent NAV ledger."""
import numpy as np
import pandas as pd

from research_v16.model import fit_v16_models, score_v16
from research_v16.portfolio import optimize_v16
from research_v20.backtest import MODES
from research_v20.timing import historical_market_state, weights_for_momentum
from .config import V20R2Settings
from .ledger import Ledger, evaluation_schedule, snapshot_weights


def run_backtest(dataset, corpus, book, membership, settings=None, progress=None, checkpoint=None):
    settings = settings or V20R2Settings()
    progress = progress or (lambda *a, **k: None)
    checkpoint = checkpoint or (lambda *a: None)
    market = historical_market_state(dataset, settings)
    schedule = evaluation_schedule(dataset, settings)
    # These checks are independent of scores, labels and realized performance.
    for date, _, _ in schedule:
        weights_for_momentum(float(market.loc[date, "market_momentum"]), settings)
    scope = dataset[dataset.in_universe.eq(True) & pd.to_datetime(dataset.date).isin([r[0] for r in schedule])]
    ledgers = {mode: Ledger(book, settings) for mode in MODES}
    benchmark = Ledger(book, settings, charge_costs=False)
    active_previous = {mode: set() for mode in MODES}
    cache, rows, holdings, daily, settlements = {}, [], [], [], []
    for year in settings.test_years:
        progress("fitting", test_year=int(year))
        models = fit_v16_models(dataset, corpus, year, settings, cache)
        _, v5, v4 = cache[year]
        for date, start, end in [r for r in schedule if r[0].year == year]:
            progress("portfolio_evaluation", test_year=int(year), date=str(date.date()))
            momentum = float(market.loc[date, "market_momentum"])
            baseline_weight, regime = weights_for_momentum(momentum, settings)
            frame = score_v16(scope[scope.date.eq(date)].copy(), models, v5, v4, settings)
            base = book.canonical(snapshot_weights(membership, date), date)
            # Restore missing snapshot members' core budget without fabricating
            # feature rows for them or selecting on future-label availability.
            frame["benchmark_weight"] = frame.symbol.map(base).fillna(0.0)
            available_budget = float(frame.benchmark_weight.sum())
            absent_core = {s: w for s, w in base.items() if s not in set(frame.symbol)}
            benchmark.settle(start)
            bench_before = benchmark.nav(start)
            benchmark.rebalance(base, start)
            benchmark.advance(start, end)
            benchmark_return = benchmark.nav(end) / bench_before - 1
            for mode, ledger in ledgers.items():
                current = frame.copy()
                current["portfolio_score"] = (
                    baseline_weight * current.v13_comparable_score + (1 - baseline_weight) * current.text_event_score
                    if mode == "v20_adaptive" else current.v16_score
                )
                in_market = mode != "v20_timing" or momentum > settings.timing_threshold
                if in_market:
                    desired, active, diagnostics = optimize_v16(current, active_previous[mode], True, True, settings)
                    desired = {s: w * available_budget for s, w in desired.items()}
                    desired.update(absent_core)
                else:
                    desired, active, diagnostics = {}, set(), {}
                ledger.settle(start)
                before = ledger.nav(start)
                old_stale = ledger.stale_observations
                execution = ledger.rebalance(desired, start)
                nav_after_trade = ledger.nav(start)
                daily.append({"date": book.dates[start], "mode": mode, "nav": nav_after_trade, "point": "after_rebalance"})
                for symbol, units in ledger.units.items():
                    holdings.append({"date": date, "execution_date": book.dates[start], "mode": mode, "symbol": symbol,
                                     "units": units, "weight": units * book.mark(symbol, start) / nav_after_trade,
                                     "target_weight": desired.get(symbol, 0.0)})
                for day, nav in ledger.advance(start, end):
                    daily.append({"date": day, "mode": mode, "nav": nav, "point": "before_rebalance"})
                after = ledger.nav(end)
                eligible = current[current.eligible.eq(True)]
                evaluation = eligible[eligible.label_5.notna()]
                tech = evaluation[evaluation.broad_sector.eq("technology")]
                ic = lambda data, target: float(data.portfolio_score.corr(data[target], method="spearman")) if len(data) >= 3 else np.nan
                rows.append({"date": date, "entry_date": book.dates[start], "end_date": book.dates[end],
                             "test_year": year, "mode": mode, "period_return": after / before - 1,
                             "benchmark_return": benchmark_return, "excess_period_return": after / before - 1 - benchmark_return,
                             "rank_ic_5": ic(evaluation, "label_5"), "rank_ic_20": ic(evaluation, "v10_target_20"),
                             "technology_rank_ic_5": ic(tech, "label_5"), "ic_eligible_rows": len(eligible),
                             "ic_labelled_rows": len(evaluation),
                             "ic20_labelled_rows": int(evaluation.v10_target_20.notna().sum()),
                             "technology_ic_labelled_rows": len(tech),
                             "risk_on_signal": bool(in_market), "in_market": bool(ledger.units),
                             "market_momentum": momentum, "market_data_end": date, "market_regime": regime,
                             "baseline_weight": baseline_weight if mode == "v20_adaptive" else settings.baseline_share,
                             "cash_weight": ledger.cash / after, "nav": after,
                             "stale_position_observations": ledger.stale_observations - old_stale,
                             **{k: v for k, v in execution.items() if k not in ("nav_before", "blocked")}, **diagnostics})
                active_previous[mode] = set(active) & set(ledger.units)
        checkpoint(year, pd.DataFrame(rows), pd.DataFrame(holdings), pd.DataFrame(daily))
        progress("year_complete", test_year=int(year), completed_years=sum(y <= year for y in settings.test_years),
                 total_years=len(settings.test_years))
        # Frozen fit helper caches large past baseline objects; only future nested
        # validation windows need the most recent two years.
        for old_year in list(cache):
            if old_year < year - settings.validation_years + 1:
                del cache[old_year]
    for mode, ledger in {**ledgers, "benchmark": benchmark}.items():
        settlements.extend({"mode": mode, **event} for event in ledger.action_log)
    return pd.DataFrame(rows), pd.DataFrame(holdings), pd.DataFrame(daily), settlements
