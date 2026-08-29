"""Replay already-frozen scores through the unchanged V20r2 ledger."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_v16.portfolio import optimize_v16
from research_v20r2.ledger import Ledger, PriceBook, snapshot_weights
from stockpilot.membership import load_membership_history


MODES = ("v16_replay", "global_only")
SCORE_COLUMNS = {"v16_replay": "v16_score", "global_only": "global_model_score"}
MARKET_PATH = Path("data/market_history_v10_hfq.csv")
MEMBERSHIP_PATH = Path("data/universes/000300/history_v10.csv")


def load_scores(folder: Path, years) -> pd.DataFrame:
    frames = [pd.read_csv(folder / f"scores_{year}.csv", dtype={"symbol": str}) for year in years]
    scores = pd.concat(frames, ignore_index=True)
    scores["symbol"] = scores.symbol.str.zfill(6)
    required = {"date", "symbol", "eligible", "broad_sector", "benchmark_weight", "label_5", "v10_target_20", *SCORE_COLUMNS.values()}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"invalid frozen score trace; missing={sorted(missing)}")
    for column in ("date", "label_end_date_5", "label_end_date_20"):
        if column in scores:
            scores[column] = pd.to_datetime(scores[column])
    if scores.duplicated(["date", "symbol"]).any():
        raise ValueError("invalid frozen score trace; duplicate keys")
    if set(scores.date.dt.year.unique()) != set(years):
        raise ValueError("frozen score years differ from protocol")
    return scores


def attach_volatility(scores: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    feature = dataset.loc[dataset.date.isin(scores.date.unique()), ["date", "symbol", "volatility_60"]]
    if feature.duplicated(["date", "symbol"]).any():
        raise ValueError("duplicate feature keys")
    merged = scores.merge(feature, on=["date", "symbol"], how="left", validate="one_to_one")
    if len(merged) != len(scores):
        raise AssertionError("score/feature join changed rows")
    return merged


def schedule_from_parent(parent: pd.DataFrame, book: PriceBook):
    control = parent[parent.mode.eq("v16_control")].copy()
    if control.date.duplicated().any() or len(control) != 73:
        raise ValueError("unexpected parent control schedule")
    rows = []
    for row in control.sort_values("date").itertuples():
        signal = pd.Timestamp(row.date)
        start, end = book.index(row.entry_date), book.index(row.end_date)
        if start != book.index(signal) + 1 or end - start != 20:
            raise ValueError("parent schedule is not the frozen common-calendar schedule")
        rows.append((signal, start, end))
    return rows


def portfolio_input(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    # Deliberately excludes every realized-return and label column.
    result = frame[["symbol", "eligible", "broad_sector", "benchmark_weight", "volatility_60", score_column]].copy()
    return result.rename(columns={score_column: "portfolio_score"})


def run_replay(scores, book, membership, schedule, settings, progress=None, checkpoint=None):
    progress = progress or (lambda *args, **kwargs: None)
    checkpoint = checkpoint or (lambda *args: None)
    ledgers = {mode: Ledger(book, settings) for mode in MODES}
    benchmark = Ledger(book, settings, charge_costs=False)
    previous = {mode: set() for mode in MODES}
    rows, holdings, daily, settlements = [], [], [], []
    for year in settings.test_years:
        for date, start, end in [row for row in schedule if row[0].year == year]:
            progress("portfolio_replay", test_year=int(year), date=str(date.date()))
            trace = scores[scores.date.eq(date)].copy()
            base = book.canonical(snapshot_weights(membership, date), date)
            trace["benchmark_weight"] = trace.symbol.map(base).fillna(0.0)
            available_budget = float(trace.benchmark_weight.sum())
            absent_core = {symbol: weight for symbol, weight in base.items() if symbol not in set(trace.symbol)}
            benchmark.settle(start)
            bench_before = benchmark.nav(start)
            benchmark.rebalance(base, start)
            benchmark.advance(start, end)
            benchmark_return = benchmark.nav(end) / bench_before - 1
            for mode in MODES:
                current = portfolio_input(trace, SCORE_COLUMNS[mode])
                desired, active, diagnostics = optimize_v16(current, previous[mode], True, True, settings)
                desired = {symbol: weight * available_budget for symbol, weight in desired.items()}
                desired.update(absent_core)
                ledger = ledgers[mode]
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
                evaluated = trace[trace.eligible.eq(True) & trace.label_5.notna()].copy()
                evaluated["portfolio_score"] = evaluated[SCORE_COLUMNS[mode]]
                tech = evaluated[evaluated.broad_sector.eq("technology")]
                ic = lambda frame, target: float(frame.portfolio_score.corr(frame[target], method="spearman")) if len(frame) >= 3 else np.nan
                rows.append({"date": date, "entry_date": book.dates[start], "end_date": book.dates[end], "test_year": year,
                             "mode": mode, "period_return": after / before - 1, "benchmark_return": benchmark_return,
                             "excess_period_return": after / before - 1 - benchmark_return,
                             "rank_ic_5": ic(evaluated, "label_5"), "rank_ic_20": ic(evaluated[evaluated.v10_target_20.notna()], "v10_target_20"),
                             "technology_rank_ic_5": ic(tech, "label_5"), "ic_eligible_rows": int(trace.eligible.sum()),
                             "ic_labelled_rows": len(evaluated), "ic20_labelled_rows": int(evaluated.v10_target_20.notna().sum()),
                             "technology_ic_labelled_rows": len(tech), "risk_on_signal": True, "in_market": bool(ledger.units),
                             "market_data_end": date, "baseline_weight": settings.baseline_share, "cash_weight": ledger.cash / after,
                             "nav": after, "stale_position_observations": ledger.stale_observations - old_stale,
                             **{key: value for key, value in execution.items() if key not in ("nav_before", "blocked")}, **diagnostics})
                previous[mode] = set(active) & set(ledger.units)
        checkpoint(year, pd.DataFrame(rows), pd.DataFrame(holdings), pd.DataFrame(daily))
    for mode, ledger in {**ledgers, "benchmark": benchmark}.items():
        settlements.extend({"mode": mode, **event} for event in ledger.action_log)
    return pd.DataFrame(rows), pd.DataFrame(holdings), pd.DataFrame(daily), settlements


def compare_control(equity, holdings, daily, settlements, parent_dir: Path):
    mapping = {"v16_replay": "v16_control"}
    checks = {}
    for name, candidate, keys, numeric in (
        ("equity", equity, ["date"], ["period_return", "benchmark_return", "nav", "buy_turnover", "sell_turnover", "transaction_cost"]),
        ("holdings", holdings, ["date", "symbol"], ["units", "weight", "target_weight"]),
        ("daily_nav", daily, ["date", "point"], ["nav"]),
    ):
        parent = pd.read_csv(parent_dir / f"{name}.csv", dtype={"symbol": str})
        parent = parent[parent["mode"].eq("v16_control")].copy().sort_values(keys).reset_index(drop=True)
        current = candidate[candidate["mode"].eq("v16_replay")].copy().sort_values(keys).reset_index(drop=True)
        if len(parent) != len(current) or not parent[keys].astype(str).equals(current[keys].astype(str)):
            raise AssertionError(f"{name} control keys differ")
        maximum = 0.0
        for column in numeric:
            delta = np.abs(pd.to_numeric(parent[column]) - pd.to_numeric(current[column]))
            maximum = max(maximum, float(delta.max()) if len(delta) else 0.0)
            if not np.allclose(pd.to_numeric(parent[column]), pd.to_numeric(current[column]), rtol=1e-10, atol=1e-12):
                raise AssertionError(f"{name}.{column} control mismatch")
        checks[name] = {"rows": len(current), "maximum_absolute_difference": maximum}
    parent_events = [event for event in json.loads((parent_dir / "settlements.json").read_text(encoding="utf-8"))["events"] if event["mode"] == "v16_control"]
    current_events = [{**event, "mode": "v16_control"} for event in settlements if event["mode"] == "v16_replay"]
    if parent_events != current_events:
        raise AssertionError("settlement control mismatch")
    checks["settlements"] = {"rows": len(current_events), "exact": True}
    return checks
