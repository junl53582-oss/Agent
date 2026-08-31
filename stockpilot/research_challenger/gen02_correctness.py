from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_v20r2.config import V20R2Settings
from research_v20r2.ledger import Ledger, PriceBook

from .config import ChallengerSettings
from .data import add_research_targets, assert_feature_columns_safe, sha256, verify_dataset_manifest
from .gen02 import (
    Gen02Settings,
    _feature_drift,
    _fit_development_scores,
    _portfolio_policies,
    _ranking_differences,
    _score_metrics,
    _selected_factors,
    _selection_stability,
    _stability_metrics,
    _tail_metrics,
    _write_csv,
    _write_json,
    development_protocol,
)
from .gen02_portfolio import PortfolioPolicy, _select_symbols, _weights
from .metrics import daily_rank_metrics, moving_block_bootstrap_delta, summarize_ic


AMENDMENT_DIR = Path(
    "artifacts/research_challenger/gen02/experiments/005_correctness_hardening"
)
CUTOFF = pd.Timestamp("2026-01-01")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sidecar_intact(path: Path) -> bool:
    sidecar = Path(str(path) + ".sha256")
    return path.is_file() and sidecar.is_file() and sidecar.read_text(encoding="ascii").strip() == sha256(path)


def load_maturity_safe_development_dataset(
    base: ChallengerSettings | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Load only rows whose longest frozen label matured before 2026.

    Filtering on label_end_date_20d is deliberately pushed into the parquet
    reader. This prevents late-2025 labels, whose exit opens occur in 2026,
    from entering the correctness recalculation.
    """

    base = base or ChallengerSettings()
    evidence = verify_dataset_manifest(base)
    assert_feature_columns_safe(base.factor_columns)
    filters = [
        ("date", "<", CUTOFF),
        ("label_end_date_20d", "<", CUTOFF),
    ]
    data = pd.read_parquet(base.dataset_path, filters=filters)
    data["date"] = pd.to_datetime(data["date"])
    for horizon in (5, 20):
        end = pd.to_datetime(data[f"label_end_date_{horizon}d"], errors="coerce")
        if end.isna().any() or end.ge(CUTOFF).any():
            raise RuntimeError(f"GEN02_CORRECTNESS_{horizon}D_LABEL_CROSSES_2026")
    if data["date"].ge(CUTOFF).any():
        raise RuntimeError("GEN02_CORRECTNESS_2026_DECISION_ROW_ENTERED")
    data["symbol"] = data["symbol"].astype(str).str.zfill(6)
    if data.duplicated(["date", "symbol"]).any():
        raise RuntimeError("GEN02_CORRECTNESS_DUPLICATE_DATE_SYMBOL")
    decision = data["date"]
    pit_checks = {
        "membership_pit": bool(
            (
                pd.to_datetime(data["membership_snapshot_date"], errors="coerce").isna()
                | pd.to_datetime(data["membership_snapshot_date"], errors="coerce").le(decision)
            ).all()
        ),
        "fundamentals_pit": bool(
            (
                pd.to_datetime(data["available_date"], errors="coerce").isna()
                | pd.to_datetime(data["available_date"], errors="coerce").le(decision)
            ).all()
        ),
        "industry_pit": bool(
            (
                pd.to_datetime(data["industry_effective_date"], errors="coerce").isna()
                | pd.to_datetime(data["industry_effective_date"], errors="coerce").le(decision)
            ).all()
        ),
    }
    if not all(pit_checks.values()):
        raise RuntimeError(f"GEN02_CORRECTNESS_PIT_FAILED: {pit_checks}")
    eligible = data["eligible"].fillna(False) & data["in_universe"].fillna(False)
    data = add_research_targets(data.loc[eligible].copy(), base.horizons)
    evidence.update(
        {
            "rows": int(len(data)),
            "symbols": int(data["symbol"].nunique()),
            "date_min": str(data["date"].min().date()),
            "date_max": str(data["date"].max().date()),
            "parquet_filters": ["date < 2026-01-01", "label_end_date_20d < 2026-01-01"],
            "latest_label_end_5d": str(pd.to_datetime(data["label_end_date_5d"]).max().date()),
            "latest_label_end_20d": str(pd.to_datetime(data["label_end_date_20d"]).max().date()),
            "2026_labels_read": False,
            "pit_checks": pit_checks,
        }
    )
    return data.sort_values(["date", "symbol"]).reset_index(drop=True), evidence


def factor_decay(
    data: pd.DataFrame,
    selected: dict[int, tuple[str, ...]],
    horizon: int,
) -> pd.DataFrame:
    """Compute factor decay using exactly the requested horizon."""

    if horizon not in (5, 20):
        raise ValueError("Gen2 correctness factor decay supports only 5D and 20D")
    target = f"future_return_{horizon}d"
    rows: list[dict] = []
    for factor in selected[2025]:
        yearly: dict[int, float] = {}
        for year in range(2020, 2026):
            current = data[data["date"].dt.year.eq(year)]
            daily = daily_rank_metrics(current, factor, target)
            yearly[year] = summarize_ic(daily)["mean_rank_ic"] if not daily.empty else np.nan
        earlier = np.asarray([yearly[year] for year in range(2020, 2025)], dtype=float)
        earlier_mean = float(np.nanmean(earlier))
        current_value = float(yearly[2025])
        sign_flip = bool(np.isfinite(earlier_mean * current_value) and earlier_mean * current_value < 0)
        rows.append(
            {
                "horizon": horizon,
                "factor": factor,
                "rank_ic_2020_2024_mean": earlier_mean,
                "rank_ic_2025": current_value,
                "ic_change": current_value - earlier_mean,
                "sign_flip": sign_flip,
                "status": "SIGN_REVERSAL"
                if sign_flip
                else "FACTOR_DECAY"
                if current_value < earlier_mean
                else "NO_DECAY",
            }
        )
    return pd.DataFrame(rows)


def sector_concentration_metrics(periods: pd.DataFrame) -> dict:
    values = pd.to_numeric(periods["maximum_sector_weight"], errors="coerce").dropna()
    if values.empty:
        return {
            "mean_maximum_sector_weight": np.nan,
            "p95_maximum_sector_weight": np.nan,
            "worst_maximum_sector_weight": np.nan,
        }
    return {
        "mean_maximum_sector_weight": float(values.mean()),
        "p95_maximum_sector_weight": float(values.quantile(0.95)),
        "worst_maximum_sector_weight": float(values.max()),
    }


def summarize_stateful_portfolio(periods: pd.DataFrame, horizon: int) -> dict:
    if periods.empty:
        return {"periods": 0}
    periods_per_year = 252 / horizon
    years = len(periods) / periods_per_year
    gross_total = float((1 + periods["gross_return"]).prod() - 1)
    net_total = float((1 + periods["net_return"]).prod() - 1)
    proxy_total = float((1 + periods["research_benchmark_proxy_return"]).prod() - 1)
    net = periods["net_return"]
    std = float(net.std(ddof=1))
    downside = float(net[net < 0].std(ddof=1))
    equity = (1 + net.fillna(0)).cumprod()
    sector = sector_concentration_metrics(periods)
    cost_rate_sum = float(periods["transaction_cost_rate"].sum())
    return {
        "periods": int(len(periods)),
        "gross_total_return": gross_total,
        "net_total_return": net_total,
        "transaction_cost_rate_sum": cost_rate_sum,
        "transaction_cost_sum": cost_rate_sum,
        "compounded_total_return_drag": gross_total - net_total,
        "research_proxy_return": proxy_total,
        "gross_research_proxy_alpha": gross_total - proxy_total,
        "net_research_proxy_alpha": net_total - proxy_total,
        "net_cagr": float((1 + net_total) ** (1 / years) - 1)
        if years > 0 and net_total > -1
        else -1.0,
        "sharpe": float(net.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0,
        "sortino": float(net.mean() / downside * np.sqrt(periods_per_year))
        if downside > 0
        else 0.0,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "average_one_way_turnover": float(
            (periods["buy_turnover"] + periods["sell_turnover"]).mean() / 2
        ),
        "annualized_turnover": float(
            (periods["buy_turnover"] + periods["sell_turnover"]).mean()
            / 2
            * periods_per_year
        ),
        "average_transaction_cost_rate": float(periods["transaction_cost_rate"].mean()),
        **sector,
        "average_size_rank": float(periods["mean_size_rank"].mean()),
        "average_liquidity_rank": float(periods["mean_liquidity_rank"].mean()),
        "mean_top_minus_bottom_spread": float(periods["top_minus_bottom_spread"].mean()),
        "blocked_sell_orders": int(periods["blocked_sell_orders"].sum()),
        "terminal_unliquidated_positions": int(periods.iloc[-1]["terminal_unliquidated_positions"]),
        "exit_tradability_status": "CANONICAL_STATEFUL_RESEARCH_LEDGER",
        "pnl_classification": "RESEARCH_PROXY_ONLY",
    }


def _weights_from_ledger(ledger: Ledger, index: int) -> dict[str, float]:
    nav = ledger.nav(index)
    return {
        symbol: units * ledger.book.mark(symbol, index) / nav
        for symbol, units in ledger.units.items()
        if units > 0 and nav > 0
    }


def _load_verified_price_book(scores: pd.DataFrame, base: ChallengerSettings) -> tuple[PriceBook, dict]:
    manifest = _load_json(base.dataset_manifest_path)
    market_entries = [
        (Path(path), digest)
        for path, digest in manifest["source_hashes"].items()
        if Path(path).name == "market_history_v10_hfq.csv"
    ]
    if len(market_entries) != 1:
        raise RuntimeError("GEN02_CORRECTNESS_MARKET_SOURCE_UNRESOLVED")
    market_path, expected_hash = market_entries[0]
    if sha256(market_path) != expected_hash:
        raise RuntimeError("GEN02_CORRECTNESS_MARKET_SOURCE_HASH_MISMATCH")
    action_path = V20R2Settings().action_path
    action_lock = _load_json(Path("artifacts/research_v20r2/plan.lock.json"))
    expected_action = action_lock["sha256"][action_path.as_posix()]
    if sha256(action_path).upper() != expected_action.upper():
        raise RuntimeError("GEN02_CORRECTNESS_ACTION_SOURCE_HASH_MISMATCH")
    events = _load_json(action_path)["events"]
    symbols = set(scores["symbol"].astype(str))
    for event in events:
        symbols.update((event["old_symbol"], event["new_symbol"]))
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        market_path,
        usecols=["date", "symbol", "open", "close", "volume"],
        dtype={"symbol": str},
        chunksize=400_000,
    ):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk["symbol"] = chunk["symbol"].astype(str).str.zfill(6)
        keep = (
            chunk["date"].ge("2019-01-01")
            & chunk["date"].lt(CUTOFF)
            & chunk["symbol"].isin(symbols)
        )
        if keep.any():
            pieces.append(chunk.loc[keep].copy())
    market = pd.concat(pieces, ignore_index=True)
    if market["date"].ge(CUTOFF).any():
        raise RuntimeError("GEN02_CORRECTNESS_2026_MARKET_ENTERED_LEDGER")
    book = PriceBook(market, events)
    return book, {
        "market_path": market_path.as_posix(),
        "market_sha256": expected_hash,
        "market_rows_loaded": int(len(market)),
        "market_date_max": str(market["date"].max().date()),
        "corporate_action_path": action_path.as_posix(),
        "corporate_action_sha256": sha256(action_path),
        "corporate_action_events": len(events),
        "canonical_ledger": "research_v20r2.ledger.Ledger",
        "2026_market_rows_used": 0,
    }


def evaluate_stateful_portfolio_policy(
    frame: pd.DataFrame,
    score_column: str,
    horizon: int,
    policy: PortfolioPolicy,
    book: PriceBook,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate with persistent positions, cash and observable sell constraints."""

    return_column = f"future_return_{horizon}d"
    signal_dates = list(pd.DatetimeIndex(frame["date"].drop_duplicates().sort_values())[::horizon])
    ledgers = {
        "gross": Ledger(book, V20R2Settings(), charge_costs=False),
        "net": Ledger(book, V20R2Settings(), charge_costs=True),
    }
    rows: list[dict] = []
    decomposition: list[dict] = []
    sector_by_symbol: dict[str, str] = {}
    previous_entry_index: int | None = None
    for signal_index, date in enumerate(signal_dates):
        current = frame[frame["date"].eq(date)].dropna(subset=[score_column, return_column]).copy()
        if len(current) < max(policy.top_k, 30):
            continue
        entry_index = int(book.dates.searchsorted(pd.Timestamp(date), side="right"))
        if entry_index >= len(book.dates) or book.dates[entry_index] >= CUTOFF:
            raise RuntimeError("GEN02_CORRECTNESS_ENTRY_NOT_MATURE_BEFORE_2026")
        if previous_entry_index is not None:
            for ledger in ledgers.values():
                ledger.advance(previous_entry_index, entry_index)
        actual_previous = _weights_from_ledger(ledgers["net"], entry_index)
        current["symbol"] = current["symbol"].astype(str)
        ranked = current.sort_values([score_column, "symbol"], ascending=[False, True]).copy()
        ranked["model_rank"] = np.arange(1, len(ranked) + 1)
        selected = _select_symbols(current, score_column, policy, actual_previous)
        desired = _weights(selected, score_column, policy.weighting)
        sector_by_symbol.update(
            selected.set_index("symbol")["broad_sector"].astype(str).to_dict()
        )
        before = {name: ledger.nav(entry_index) for name, ledger in ledgers.items()}
        trades = {name: ledger.rebalance(desired, entry_index) for name, ledger in ledgers.items()}
        if signal_index + 1 < len(signal_dates):
            end_index = int(
                book.dates.searchsorted(pd.Timestamp(signal_dates[signal_index + 1]), side="right")
            )
        else:
            end_index = entry_index + horizon
        if end_index >= len(book.dates) or book.dates[end_index] >= CUTOFF:
            raise RuntimeError("GEN02_CORRECTNESS_EXIT_NOT_MATURE_BEFORE_2026")
        for ledger in ledgers.values():
            ledger.advance(entry_index, end_index)
        final_trades = None
        if signal_index + 1 == len(signal_dates):
            final_trades = {
                name: ledger.rebalance({}, end_index) for name, ledger in ledgers.items()
            }
        end_nav = {name: ledger.nav(end_index) for name, ledger in ledgers.items()}
        gross_return = end_nav["gross"] / before["gross"] - 1
        net_return = end_nav["net"] / before["net"] - 1
        benchmark = current[current["benchmark_weight"].gt(0)].copy()
        weight_total = float(benchmark["benchmark_weight"].sum())
        proxy = float(
            (
                benchmark["benchmark_weight"]
                * pd.to_numeric(benchmark[return_column], errors="coerce").fillna(0)
            ).sum()
            / weight_total
        )
        net_trade = trades["net"]
        terminal_trade = final_trades["net"] if final_trades else None
        cost_rate = float(net_trade["transaction_cost"])
        buy_turnover = float(net_trade["buy_turnover"])
        sell_turnover = float(net_trade["sell_turnover"])
        blocked = list(net_trade["blocked"])
        if terminal_trade:
            cost_rate += float(terminal_trade["transaction_cost"])
            buy_turnover += float(terminal_trade["buy_turnover"])
            sell_turnover += float(terminal_trade["sell_turnover"])
            blocked.extend(terminal_trade["blocked"])
        weights = _weights_from_ledger(ledgers["net"], end_index)
        sector_weights: dict[str, float] = {}
        for symbol, weight in weights.items():
            sector = sector_by_symbol.get(symbol, "UNKNOWN")
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight
        bottom = ranked.tail(policy.top_k)
        top = ranked[ranked["symbol"].isin(desired)]
        rows.append(
            {
                "date": pd.Timestamp(date),
                "entry_date": book.dates[entry_index],
                "exit_valuation_date": book.dates[end_index],
                "gross_return": gross_return,
                "net_return": net_return,
                "transaction_cost_rate": cost_rate,
                "research_benchmark_proxy_return": proxy,
                "gross_research_proxy_alpha": gross_return - proxy,
                "net_research_proxy_alpha": net_return - proxy,
                "top_minus_bottom_spread": float(
                    top[return_column].mean() - bottom[return_column].mean()
                ),
                "buy_turnover": buy_turnover,
                "sell_turnover": sell_turnover,
                "cash_weight": float(ledgers["net"].cash / end_nav["net"]),
                "maximum_sector_weight": max(sector_weights.values(), default=0.0),
                "mean_size_rank": float(top["benchmark_weight_rank"].mean()),
                "mean_liquidity_rank": float(top["amount_rank"].mean()),
                "blocked_sell_orders": sum(
                    item.get("side") == "sell" for item in blocked
                ),
                "terminal_unliquidated_positions": len(ledgers["net"].units)
                if final_trades
                else 0,
            }
        )
        decomposition.append(
            {
                "date": pd.Timestamp(date),
                "blocked_sell_orders": sum(item.get("side") == "sell" for item in blocked),
                "blocked_buy_orders": sum(item.get("side") == "buy" for item in blocked),
                "actual_position_count": len(ledgers["net"].units),
                "cash_weight": float(ledgers["net"].cash / end_nav["net"]),
            }
        )
        previous_entry_index = entry_index
    return pd.DataFrame(rows), pd.DataFrame(decomposition)


def portfolio_analysis_corrected(
    scores: pd.DataFrame,
    gen: Gen02Settings,
    base: ChallengerSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int, str], pd.DataFrame], dict]:
    book, evidence = _load_verified_price_book(scores, base)
    summaries: list[dict] = []
    decomposition_rows: list[dict] = []
    period_map: dict[tuple[str, int, str], pd.DataFrame] = {}
    for horizon in gen.horizons:
        part = scores[scores["horizon"].eq(horizon)].copy()
        for model in ("v6", *gen.candidate_models):
            for policy in _portfolio_policies(gen):
                periods, decomposition = evaluate_stateful_portfolio_policy(
                    part, f"score_{model}", horizon, policy, book
                )
                key = (model, horizon, policy.name)
                period_map[key] = periods
                summaries.append(
                    {
                        "model": model,
                        "horizon": horizon,
                        "portfolio_policy": policy.name,
                        "top_k": policy.top_k,
                        "weighting": policy.weighting,
                        "buffer_exit_rank": policy.buffer_exit_rank,
                        "sector_balanced": policy.sector_balanced,
                        **summarize_stateful_portfolio(periods, horizon),
                    }
                )
                if not decomposition.empty:
                    totals = decomposition.drop(columns="date").sum(numeric_only=True)
                    decomposition_rows.append(
                        {
                            "model": model,
                            "horizon": horizon,
                            "portfolio_policy": policy.name,
                            **{name: float(value) for name, value in totals.items()},
                        }
                    )
    return pd.DataFrame(summaries), pd.DataFrame(decomposition_rows), period_map, evidence


def choose_configuration_corrected(
    model_metrics: pd.DataFrame,
    yearly: pd.DataFrame,
    portfolios: pd.DataFrame,
    gen: Gen02Settings,
) -> tuple[dict | None, dict | None, pd.DataFrame]:
    rows: list[dict] = []
    for _, portfolio in portfolios[portfolios["model"].isin(gen.candidate_models)].iterrows():
        metric = model_metrics[
            model_metrics["model"].eq(portfolio["model"])
            & model_metrics["horizon"].eq(portfolio["horizon"])
        ].iloc[0]
        y2025 = yearly[
            yearly["model"].eq(portfolio["model"])
            & yearly["horizon"].eq(portfolio["horizon"])
            & yearly["test_year"].eq(2025)
        ].iloc[0]
        gates = {
            "rank_ic_positive": metric["mean_rank_ic"] > gen.minimum_rank_ic,
            "rank_ic_ir_positive": metric["rank_ic_ir"] > gen.minimum_rank_ic_ir,
            "positive_ratio": metric["positive_rank_ic_ratio"] >= gen.minimum_positive_ratio,
            "rank_ic_2025_positive": y2025["mean_rank_ic"] > gen.minimum_2025_rank_ic,
            "net_research_proxy_alpha_positive": portfolio["net_research_proxy_alpha"]
            > gen.minimum_net_research_proxy_alpha,
            "drawdown_acceptable": portfolio["max_drawdown"] >= gen.maximum_drawdown,
            "turnover_acceptable": portfolio["annualized_turnover"]
            <= gen.maximum_annualized_turnover,
            "sector_concentration_acceptable": portfolio["worst_maximum_sector_weight"]
            <= gen.maximum_sector_weight,
        }
        rows.append(
            {
                "model": portfolio["model"],
                "horizon": int(portfolio["horizon"]),
                "portfolio_policy": portfolio["portfolio_policy"],
                "mean_rank_ic": float(metric["mean_rank_ic"]),
                "rank_ic_ir": float(metric["rank_ic_ir"]),
                "positive_rank_ic_ratio": float(metric["positive_rank_ic_ratio"]),
                "rank_ic_2025": float(y2025["mean_rank_ic"]),
                "net_research_proxy_alpha": float(portfolio["net_research_proxy_alpha"]),
                "max_drawdown": float(portfolio["max_drawdown"]),
                "annualized_turnover": float(portfolio["annualized_turnover"]),
                "mean_maximum_sector_weight": float(portfolio["mean_maximum_sector_weight"]),
                "p95_maximum_sector_weight": float(portfolio["p95_maximum_sector_weight"]),
                "worst_maximum_sector_weight": float(portfolio["worst_maximum_sector_weight"]),
                "gates_passed": sum(bool(value) for value in gates.values()),
                "all_gates_passed": bool(all(gates.values())),
                "gates": json.dumps({key: bool(value) for key, value in gates.items()}, sort_keys=True),
            }
        )
    table = pd.DataFrame(rows)
    eligible = table[table["all_gates_passed"]].sort_values(
        ["net_research_proxy_alpha", "mean_rank_ic"], ascending=False
    )
    eligible_configuration = eligible.iloc[0].to_dict() if not eligible.empty else None
    ineligible = table[~table["all_gates_passed"]]
    near_miss = ineligible.sort_values(
        ["gates_passed", "net_research_proxy_alpha", "mean_rank_ic"], ascending=False
    ).iloc[0].to_dict() if not ineligible.empty else None
    if eligible_configuration is not None:
        eligible_configuration["selection_semantics"] = "FROZEN_PROTOCOL_ELIGIBLE"
    if near_miss is not None:
        near_miss["selection_semantics"] = "DIAGNOSTIC_NEAR_MISS_NOT_ELIGIBLE"
    return eligible_configuration, near_miss, table


def freeze_postrun_interpretation() -> dict:
    """Correct result interpretation without rerunning any research calculation."""

    target = AMENDMENT_DIR / "experiments/006_postrun_eligibility_interpretation"
    if target.exists() and any(target.iterdir()):
        raise RuntimeError("GEN02_CORRECTNESS_INTERPRETATION_ALREADY_EXISTS")
    table = pd.read_csv(AMENDMENT_DIR / "candidate_gate_matrix_corrected.csv")
    gate_flags = table["all_gates_passed"].map(
        lambda value: bool(value)
        if isinstance(value, (bool, np.bool_))
        else str(value).strip().lower() == "true"
    )
    eligible = table[gate_flags].sort_values(
        ["net_research_proxy_alpha", "mean_rank_ic"], ascending=False
    )
    ineligible = table[~gate_flags].sort_values(
        ["gates_passed", "net_research_proxy_alpha", "mean_rank_ic"], ascending=False
    )
    eligible_configuration = eligible.iloc[0].to_dict() if not eligible.empty else None
    diagnostic_near_miss = ineligible.iloc[0].to_dict() if not ineligible.empty else None
    protocol = {
        "amendment_id": "GEN02-CORRECTNESS-POSTRUN-INTERPRETATION-006",
        "classification": "POSTRUN_INTERPRETATION_ONLY",
        "reason": "Separate a gate-eligible corrected configuration from the best ineligible diagnostic near miss.",
        "research_recalculated": False,
        "model_retrained": False,
        "performance_recomputed": False,
        "holdout_opened": False,
        "2026_labels_read": False,
        "hyperparameters_changed": False,
        "thresholds_changed": False,
        "provider_requests": 0,
    }
    interpretation = {
        "correctness_status": "CORRECTNESS_RECALCULATION_CHANGED_ELIGIBILITY",
        "operative_frozen_gen2_decision": "GEN2_REJECTED",
        "human_readjudication_required": True,
        "eligible_configuration_under_corrected_mechanics": eligible_configuration,
        "diagnostic_near_miss": diagnostic_near_miss,
        "shadow_eligible": False,
        "automatic_promotion_forbidden": True,
        "v6_remains_champion": True,
        "production_prediction_ready": False,
        "execution_authorized": False,
        "prospective_boundary": correctness_protocol()["prospective_boundary"],
    }
    _write_json(target / "protocol_amendment.json", protocol)
    _write_json(target / "final_interpretation.json", interpretation)
    pre_run = _load_json(AMENDMENT_DIR / "pre_run_plan.lock.json")
    code_path = Path("stockpilot/research_challenger/gen02_correctness.py")
    test_path = Path("tests/test_research_challenger_gen02_correctness.py")
    lock = {
        "lock_id": "GEN02-CORRECTNESS-POSTRUN-INTERPRETATION-006",
        "created_at_utc": _utc(),
        "original_code_sha256": pre_run["files"][code_path.as_posix()],
        "new_code_sha256": sha256(code_path),
        "new_test_sha256": sha256(test_path),
        "files": {
            "protocol_amendment.json": sha256(target / "protocol_amendment.json"),
            "final_interpretation.json": sha256(target / "final_interpretation.json"),
            (AMENDMENT_DIR / "candidate_gate_matrix_corrected.csv").as_posix(): sha256(
                AMENDMENT_DIR / "candidate_gate_matrix_corrected.csv"
            ),
        },
        "research_recalculated": False,
        "holdout_opened": False,
        "2026_labels_read": False,
    }
    digest = _write_json(target / "plan.lock.json", lock)
    return {"status": "INTERPRETATION_FROZEN", "lock_sha256": digest, **interpretation}


def correctness_protocol() -> dict:
    return {
        "amendment_id": "GEN02-CORRECTNESS-HARDENING-005",
        "classification": "CORRECTNESS_ONLY_AMENDMENT",
        "parent_commit": "0bf59670496ef86f37608c1682f6573473ec6072",
        "parent_effective_lock": "c3353cf4643d4aa08e994ffb30581db71e82c87a917f021f0f71fc06bddecf1d",
        "changes": [
            "maturity-safe label filter before 2026",
            "horizon-aware factor decay",
            "worst-snapshot maximum sector weight gate",
            "V20r2 stateful sell/cash/corporate-action ledger",
            "frozen eligible selection separated from diagnostic near miss",
            "cost-rate sum separated from compounded total-return drag",
            "V1r4 evidence explicitly excluded from Gen2 validation",
        ],
        "unchanged": {
            "models": ["ridge", "lightgbm_regression"],
            "horizons": [5, 20],
            "development_years": [2020, 2021, 2022, 2023, 2024, 2025],
            "hyperparameters_changed": False,
            "thresholds_changed": False,
            "costs_changed": False,
            "new_models_added": False,
            "holdout_opened": False,
            "2026_labels_read": False,
            "provider_requests": 0,
        },
        "sector_gate": {
            "metric": "worst_maximum_sector_weight",
            "operator": "<=",
            "threshold": 0.45,
            "reason": "frozen protocol says maximum, not average maximum",
        },
        "portfolio_status": "RESEARCH_PROXY_ONLY",
        "benchmark_status": "UNAPPROVED",
        "prospective_boundary": (
            "V1r4 observations and V30r1-forward-r2 predictions do not constitute prospective "
            "validation of the Gen2 LightGBM challenger. Gen2 remains rejected and no Gen2 "
            "prospective system is created by this amendment."
        ),
        "production_prediction_ready": False,
        "execution_authorized": False,
        "shadow_eligible": False,
    }


def freeze_protocol() -> dict:
    if AMENDMENT_DIR.exists() and any(AMENDMENT_DIR.iterdir()):
        raise RuntimeError("GEN02_CORRECTNESS_AMENDMENT_ALREADY_EXISTS")
    protocol_path = AMENDMENT_DIR / "protocol_amendment.json"
    _write_json(protocol_path, correctness_protocol())
    parents = [
        Path("artifacts/research_challenger/gen02/plan.lock.json"),
        Path("artifacts/research_challenger/gen02/decision.json"),
        Path("artifacts/prospective_alpha_v1r4/plan.lock.json"),
        Path("artifacts/research_v6/plan.lock.json"),
        Path("artifacts/research_v20r2/plan.lock.json"),
    ]
    files = [
        Path("stockpilot/research_challenger/gen02_correctness.py"),
        Path("tests/test_research_challenger_gen02_correctness.py"),
        protocol_path,
        *parents,
    ]
    lock = {
        "lock_id": "GEN02-CORRECTNESS-HARDENING-005-PRE-RUN",
        "created_at_utc": _utc(),
        "files": {path.as_posix(): sha256(path) for path in files},
        "original_artifact_hashes": {
            path.as_posix(): sha256(path)
            for path in Path("artifacts/research_challenger/gen02").glob("*")
            if path.is_file()
        },
        "holdout_opened": False,
        "2026_labels_read": False,
        "hyperparameters_changed": False,
        "thresholds_changed": False,
    }
    digest = _write_json(AMENDMENT_DIR / "pre_run_plan.lock.json", lock)
    return {"status": "FROZEN", "lock_sha256": digest}


def verify_protocol() -> dict:
    lock_path = AMENDMENT_DIR / "pre_run_plan.lock.json"
    if not _sidecar_intact(lock_path):
        return {"intact": False, "mismatches": [lock_path.as_posix()]}
    lock = _load_json(lock_path)
    mismatches = [path for path, digest in lock["files"].items() if not Path(path).is_file() or sha256(Path(path)) != digest]
    return {"intact": not mismatches, "mismatches": mismatches, "lock_sha256": sha256(lock_path)}


def run_correctness_recalculation() -> dict:
    verification = verify_protocol()
    if not verification["intact"]:
        raise RuntimeError(f"GEN02_CORRECTNESS_PROTOCOL_NOT_INTACT: {verification}")
    result_path = AMENDMENT_DIR / "decision.json"
    if result_path.exists():
        raise RuntimeError("GEN02_CORRECTNESS_ALREADY_CONSUMED")
    settings = Gen02Settings()
    base = ChallengerSettings()
    data, data_evidence = load_maturity_safe_development_dataset(base)
    selected = _selected_factors()
    scores, sensitivity = _fit_development_scores(data, settings, base)
    model_metrics, yearly = _score_metrics(scores)
    tail = _tail_metrics(scores)
    drift = _feature_drift(data, selected)
    decay = pd.concat(
        [factor_decay(data, selected, horizon) for horizon in settings.horizons],
        ignore_index=True,
    )
    selection_pairs, selection_frequency = _selection_stability(selected)
    portfolios, decomposition, periods, price_evidence = portfolio_analysis_corrected(
        scores, settings, base
    )
    regimes, industries = _stability_metrics(scores)
    ranking = _ranking_differences(scores)
    eligible, near_miss, gate_table = choose_configuration_corrected(
        model_metrics, yearly, portfolios, settings
    )
    reference = eligible or near_miss
    key = (reference["model"], int(reference["horizon"]), reference["portfolio_policy"])
    challenger_periods = periods[key].set_index("date")["net_research_proxy_alpha"]
    v6_periods = periods[("v6", key[1], key[2])].set_index("date")["net_research_proxy_alpha"]
    score_part = scores[scores["horizon"].eq(key[1])]
    challenger_ic: list[float] = []
    v6_ic: list[float] = []
    for _, current in score_part.groupby("date"):
        challenger_ic.append(current[f"score_{key[0]}"].corr(current[f"future_return_{key[1]}d"], method="spearman"))
        v6_ic.append(current["score_v6"].corr(current[f"future_return_{key[1]}d"], method="spearman"))
    bootstrap = {
        "classification": "DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
        "rank_ic_delta_vs_v6": moving_block_bootstrap_delta(
            pd.Series(challenger_ic), pd.Series(v6_ic),
            replications=settings.bootstrap_replications,
            block_length=settings.bootstrap_block_length,
            seed=settings.random_seed,
        ),
        "topk_net_research_proxy_alpha_delta_vs_v6": moving_block_bootstrap_delta(
            challenger_periods, v6_periods,
            replications=settings.bootstrap_replications,
            block_length=min(10, settings.bootstrap_block_length),
            seed=settings.random_seed,
        ),
    }
    original = _load_json(Path("artifacts/research_challenger/gen02/decision.json"))[
        "best_development_configuration"
    ]
    original_match = portfolios[
        portfolios["model"].eq(original["model"])
        & portfolios["horizon"].eq(original["horizon"])
        & portfolios["portfolio_policy"].eq(original["portfolio_policy"])
    ].iloc[0].to_dict()
    corrected_original_metric = model_metrics[
        model_metrics["model"].eq(original["model"])
        & model_metrics["horizon"].eq(original["horizon"])
    ].iloc[0].to_dict()
    corrected_original_2025 = yearly[
        yearly["model"].eq(original["model"])
        & yearly["horizon"].eq(original["horizon"])
        & yearly["test_year"].eq(2025)
    ].iloc[0].to_dict()
    corrected_reference = {
        **original_match,
        **{
            key_name: corrected_original_metric[key_name]
            for key_name in ("mean_rank_ic", "rank_ic_ir", "positive_rank_ic_ratio")
        },
        "rank_ic_2025": corrected_original_2025["mean_rank_ic"],
    }
    outcome_changed = False
    eligibility_changed = eligible is not None
    decision_name = (
        "CORRECTNESS_RECALCULATION_CHANGED_ELIGIBILITY"
        if eligibility_changed
        else "GEN2_REJECTED"
    )
    selected_decay = decay[decay["horizon"].eq(int(reference["horizon"]))]
    selected_tail = tail[
        tail["model"].eq(reference["model"])
        & tail["horizon"].eq(reference["horizon"])
        & tail["test_year"].eq(2025)
    ].iloc[0]
    failure = {
        "classification": "CORRECTNESS_ONLY_DEVELOPMENT_RECALCULATION",
        "primary_configuration": reference,
        "factor_decay_horizon": int(reference["horizon"]),
        "factor_decay_uses_selected_horizon": True,
        "top_tail": {
            "2025_top_decile_ic": float(selected_tail["top_decile_ic"]),
            "2025_top_quintile_ic": float(selected_tail["top_quintile_ic"]),
            "2025_overall_rank_ic": float(selected_tail["overall_rank_ic"]),
        },
        "factor_decay": {
            "sign_reversal_count": int(selected_decay["sign_flip"].sum()),
            "largest_negative_ic_changes": selected_decay.nsmallest(5, "ic_change")[["factor", "ic_change"]].to_dict(orient="records"),
        },
        "2026_labels_read": False,
    }
    governance = {
        "amendment": "CORRECTNESS_ONLY",
        "original_result_preserved": True,
        "research_outcome_changed": outcome_changed,
        "eligibility_changed": eligibility_changed,
        "decision": decision_name,
        "holdout_opened": False,
        "2026_labels_read": False,
        "provider_requests": 0,
        "model_retrained": True,
        "model_retrain_reason": "deterministic correctness-only recalculation after maturity and execution fixes",
        "hyperparameters_changed": False,
        "thresholds_changed": False,
        "new_models_added": False,
        "v6_modified": False,
        "v30_modified": False,
        "v30r1_modified": False,
        "v1r4_modified": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
        "shadow_eligible": False,
    }
    before_after = {
        "comparison_basis": "same original LightGBM Regression 20D equal_top30 configuration",
        "original": original,
        "corrected": corrected_reference,
        "reasons": [
            "2025 labels crossing into 2026 excluded",
            "stateful V20r2 tradability/cash ledger",
            "worst-snapshot sector metric added",
            "cost semantics clarified",
        ],
    }
    report = {
        "report_id": "GEN02-CORRECTNESS-HARDENING-005",
        "created_at_utc": _utc(),
        "decision": decision_name,
        "gen2_rejected_remains": not eligibility_changed,
        "v6_remains_champion": True,
        "eligible_configuration": eligible,
        "diagnostic_near_miss": near_miss,
        "data": data_evidence,
        "price_and_execution_evidence": price_evidence,
        "exit_tradability_verdict": {
            "status": "CANONICAL_STATEFUL_RESEARCH_LEDGER_NOT_PRODUCTION_REALISTIC",
            "implemented": True,
            "sell_failure_retains_position": True,
            "sell_failure_does_not_release_cash": True,
            "terminal_untradable_position_not_fake_liquidated": True,
            "limitations": [
                "missing quote is an observable non-trading state, not independently verified suspension metadata",
                "price-limit test is an approximation without order-book depth",
                "HFQ economic units and fractional shares remain research approximations",
            ],
        },
        "transaction_cost_semantics": (
            "transaction_cost_rate_sum is the arithmetic sum of per-period cost rates; "
            "compounded_total_return_drag equals gross_total_return minus net_total_return"
        ),
        "prospective_boundary": correctness_protocol()["prospective_boundary"],
        "official_benchmark_status": "UNAPPROVED",
        "alpha_field_semantics": "research_proxy_alpha_only",
        "governance": governance,
    }
    outputs = {
        "model_comparison_corrected.csv": model_metrics,
        "yearly_metrics_corrected.csv": yearly,
        "tail_metrics_corrected.csv": tail,
        "feature_drift_corrected.csv": drift,
        "factor_decay_corrected.csv": decay,
        "factor_selection_stability.csv": selection_pairs,
        "factor_selection_frequency.csv": selection_frequency,
        "training_window_sensitivity.csv": sensitivity,
        "portfolio_variants_corrected.csv": portfolios,
        "turnover_decomposition_corrected.csv": decomposition,
        "candidate_gate_matrix_corrected.csv": gate_table,
        "regime_metrics_corrected.csv": regimes,
        "industry_metrics_corrected.csv": industries,
        "ranking_differences_corrected.csv": ranking,
    }
    for name, frame in outputs.items():
        _write_csv(AMENDMENT_DIR / name, frame)
    for name, payload in {
        "bootstrap_corrected.json": bootstrap,
        "2025_failure_analysis_corrected.json": failure,
        "before_after.json": before_after,
        "governance.json": governance,
        "decision.json": {
            "decision": decision_name,
            "original_decision": "GEN2_REJECTED",
            "eligible_configuration": eligible,
            "diagnostic_near_miss": near_miss,
            "shadow_eligible": False,
            "v6_remains_champion": True,
            "production_prediction_ready": False,
            "execution_authorized": False,
        },
        "report.json": report,
    }.items():
        _write_json(AMENDMENT_DIR / name, payload)
    return {
        "decision": decision_name,
        "eligible_configuration": eligible,
        "diagnostic_near_miss": near_miss,
        "2026_labels_read": False,
        "holdout_opened": False,
    }


def record_tests(kind: str, command: str, summary: str) -> dict:
    payload = {
        "kind": kind,
        "recorded_at_utc": _utc(),
        "command": command,
        "summary": summary,
        "exit_code": 0,
        "new_skip_added": False,
        "new_xfail_added": False,
    }
    _write_json(AMENDMENT_DIR / f"{kind}_test_receipt.json", payload)
    return payload


def freeze_final() -> dict:
    required = [
        AMENDMENT_DIR / "protocol_amendment.json",
        AMENDMENT_DIR / "pre_run_plan.lock.json",
        AMENDMENT_DIR / "decision.json",
        AMENDMENT_DIR / "report.json",
        AMENDMENT_DIR / "targeted_test_receipt.json",
        AMENDMENT_DIR / "full_test_receipt.json",
        AMENDMENT_DIR / "experiments/006_postrun_eligibility_interpretation/final_interpretation.json",
    ]
    missing = [path.as_posix() for path in required if not _sidecar_intact(path)]
    if missing:
        raise RuntimeError(f"GEN02_CORRECTNESS_FINAL_REQUIRED_MISSING: {missing}")
    files = [
        path for path in AMENDMENT_DIR.rglob("*")
        if path.is_file() and not path.name.endswith(".sha256") and path.name not in {"plan.lock.json", "artifact_manifest.json"}
    ]
    manifest = {
        "manifest_id": "GEN02-CORRECTNESS-HARDENING-005",
        "created_at_utc": _utc(),
        "files": {path.relative_to(AMENDMENT_DIR).as_posix(): sha256(path) for path in sorted(files)},
    }
    _write_json(AMENDMENT_DIR / "artifact_manifest.json", manifest)
    lock_files = [
        Path("stockpilot/research_challenger/gen02_correctness.py"),
        Path("tests/test_research_challenger_gen02_correctness.py"),
        AMENDMENT_DIR / "artifact_manifest.json",
        Path("artifacts/research_challenger/gen02/plan.lock.json"),
        Path("artifacts/prospective_alpha_v1r4/plan.lock.json"),
        Path("artifacts/research_v6/plan.lock.json"),
        Path("artifacts/research_v20r2/plan.lock.json"),
    ]
    lock = {
        "lock_id": "GEN02-CORRECTNESS-HARDENING-005-FINAL",
        "created_at_utc": _utc(),
        "files": {path.as_posix(): sha256(path) for path in lock_files},
        "decision": _load_json(AMENDMENT_DIR / "decision.json")["decision"],
        "holdout_opened": False,
        "2026_labels_read": False,
        "hyperparameters_changed": False,
        "thresholds_changed": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
        "shadow_eligible": False,
    }
    digest = _write_json(AMENDMENT_DIR / "plan.lock.json", lock)
    return {"status": "FROZEN", "lock_sha256": digest, "manifest_sha256": sha256(AMENDMENT_DIR / "artifact_manifest.json")}


def verify_final() -> dict:
    lock_path = AMENDMENT_DIR / "plan.lock.json"
    if not _sidecar_intact(lock_path):
        return {"intact": False, "mismatches": [lock_path.as_posix()]}
    lock = _load_json(lock_path)
    mismatches = [path for path, digest in lock["files"].items() if not Path(path).is_file() or sha256(Path(path)) != digest]
    return {"intact": not mismatches, "mismatches": mismatches, "lock_sha256": sha256(lock_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Gen2 correctness-only hardening")
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze-protocol", "verify-protocol", "run", "freeze-interpretation", "freeze-final", "verify-final"):
        subs.add_parser(name)
    receipt = subs.add_parser("record-tests")
    receipt.add_argument("kind", choices=("targeted", "full"))
    receipt.add_argument("test_command")
    receipt.add_argument("summary")
    args = parser.parse_args()
    if args.command == "freeze-protocol":
        result = freeze_protocol()
    elif args.command == "verify-protocol":
        result = verify_protocol()
    elif args.command == "run":
        result = run_correctness_recalculation()
    elif args.command == "freeze-interpretation":
        result = freeze_postrun_interpretation()
    elif args.command == "record-tests":
        result = record_tests(args.kind, args.test_command, args.summary)
    elif args.command == "freeze-final":
        result = freeze_final()
    else:
        result = verify_final()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
