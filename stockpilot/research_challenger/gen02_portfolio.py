from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from research_v6.model import _sector_quotas
from stockpilot.portfolio import turnover


@dataclass(frozen=True)
class PortfolioPolicy:
    name: str
    top_k: int
    weighting: str = "equal"
    buffer_exit_rank: int | None = None
    sector_balanced: bool = False


def _execution_columns(horizon: int) -> tuple[str, str]:
    if horizon == 5:
        return "entry_tradable", "execution_return"
    if horizon == 20:
        return "entry_tradable_20", "execution_return_20"
    raise ValueError("Gen02 portfolios only support the frozen 5D and 20D horizons")


def _select_symbols(
    current: pd.DataFrame,
    score_column: str,
    policy: PortfolioPolicy,
    previous: dict[str, float],
) -> pd.DataFrame:
    ranked = current.sort_values([score_column, "symbol"], ascending=[False, True]).copy()
    ranked["model_rank"] = np.arange(1, len(ranked) + 1)
    if policy.sector_balanced:
        quotas = _sector_quotas(ranked, policy.top_k)
        pieces = [
            ranked[ranked["broad_sector"].astype(str).eq(sector)].head(quota)
            for sector, quota in quotas.items()
        ]
        selected = pd.concat(pieces, ignore_index=False) if pieces else ranked.iloc[0:0]
        return selected.sort_values([score_column, "symbol"], ascending=[False, True]).head(
            policy.top_k
        )
    if policy.buffer_exit_rank is None or not previous:
        return ranked.head(policy.top_k)
    retained = ranked[
        ranked["symbol"].astype(str).isin(previous)
        & ranked["model_rank"].le(policy.buffer_exit_rank)
    ].head(policy.top_k)
    remaining = ranked[~ranked.index.isin(retained.index)].head(policy.top_k - len(retained))
    return pd.concat([retained, remaining]).sort_values(
        [score_column, "symbol"], ascending=[False, True]
    )


def _weights(selected: pd.DataFrame, score_column: str, weighting: str) -> dict[str, float]:
    if selected.empty:
        return {}
    if weighting == "equal":
        raw = np.ones(len(selected), dtype=float)
    elif weighting == "rank":
        raw = np.arange(len(selected), 0, -1, dtype=float)
    elif weighting == "rank_decay":
        raw = np.exp(-np.arange(len(selected), dtype=float) / max(1.0, len(selected) / 3))
    else:
        raise ValueError(f"unsupported transparent weighting policy: {weighting}")
    raw /= raw.sum()
    return {
        str(symbol): float(weight)
        for symbol, weight in zip(selected["symbol"].astype(str), raw)
    }


def evaluate_portfolio_policy(
    frame: pd.DataFrame,
    score_column: str,
    horizon: int,
    policy: PortfolioPolicy,
    *,
    rebalance_every: int,
    buy_rate: float,
    sell_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one transparent policy continuously across all development years.

    The function intentionally uses the existing execution-return and tradability
    columns. It does not introduce a new price, cost, or benchmark convention.
    """

    return_column = f"future_return_{horizon}d"
    tradable_column, execution_column = _execution_columns(horizon)
    dates = pd.DatetimeIndex(frame["date"].drop_duplicates().sort_values())[::rebalance_every]
    previous: dict[str, float] = {}
    previous_sector: dict[str, str] = {}
    rows: list[dict] = []
    decomposition: list[dict] = []
    for date in dates:
        current = frame[frame["date"].eq(date)].copy()
        current = current.dropna(subset=[score_column, return_column])
        if len(current) < max(policy.top_k, 30):
            continue
        current["symbol"] = current["symbol"].astype(str)
        ranked = current.sort_values([score_column, "symbol"], ascending=[False, True]).copy()
        ranked["model_rank"] = np.arange(1, len(ranked) + 1)
        selected = _select_symbols(current, score_column, policy, previous)
        desired = _weights(selected, score_column, policy.weighting)
        lookup = selected.set_index("symbol")
        executed: dict[str, float] = {}
        realized: dict[str, float] = {}
        for symbol, weight in desired.items():
            row = lookup.loc[symbol]
            continuing = symbol in previous
            can_enter = bool(row.get(tradable_column, False))
            value = row[return_column] if continuing else row[execution_column]
            if (continuing or can_enter) and pd.notna(value) and np.isfinite(float(value)):
                executed[symbol] = float(weight)
                realized[symbol] = float(value)

        buys, sells = turnover(previous, executed)
        cost = buys * buy_rate + sells * sell_rate
        gross = float(sum(executed[symbol] * realized[symbol] for symbol in executed))
        benchmark = current[current["benchmark_weight"].gt(0)].copy()
        weight_total = float(benchmark["benchmark_weight"].sum())
        proxy = (
            float(
                (
                    benchmark["benchmark_weight"]
                    * pd.to_numeric(benchmark[return_column], errors="coerce").fillna(0)
                ).sum()
                / weight_total
            )
            if weight_total > 0
            else float(current[return_column].mean())
        )
        bottom = ranked.tail(policy.top_k)
        selected_raw = ranked[ranked["symbol"].isin(executed)]
        sector_weights: dict[str, float] = {}
        for symbol, weight in executed.items():
            sector = str(lookup.loc[symbol, "broad_sector"])
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight

        previous_symbols = set(previous)
        current_symbols = set(ranked["symbol"])
        exited = previous_symbols.difference(executed)
        absent = exited.difference(current_symbols)
        forced = {
            symbol
            for symbol in exited.intersection(current_symbols)
            if symbol not in desired
            and (
                pd.isna(ranked.set_index("symbol").loc[symbol, return_column])
                or not np.isfinite(
                    float(ranked.set_index("symbol").loc[symbol, return_column])
                )
            )
        }
        ranking_exits = exited.difference(absent).difference(forced)
        rank_lookup = ranked.set_index("symbol")["model_rank"].to_dict()
        near_cutoff = {
            symbol
            for symbol in ranking_exits
            if rank_lookup.get(symbol, policy.top_k + 100) <= policy.top_k + 10
        }
        entered = set(executed).difference(previous_symbols)
        entered_sectors = {str(lookup.loc[symbol, "broad_sector"]) for symbol in entered}
        cross_sector_exits = {
            symbol
            for symbol in ranking_exits
            if previous_sector.get(symbol) not in entered_sectors
        }

        rows.append(
            {
                "date": date,
                "gross_return": gross,
                "transaction_cost": cost,
                "net_return": gross - cost,
                "research_benchmark_proxy_return": proxy,
                "gross_research_proxy_alpha": gross - proxy,
                "net_research_proxy_alpha": gross - cost - proxy,
                "top_minus_bottom_spread": float(
                    selected_raw[return_column].mean() - bottom[return_column].mean()
                ),
                "buy_turnover": buys,
                "sell_turnover": sells,
                "executed_symbols": len(executed),
                "cash_weight": 1 - sum(executed.values()),
                "maximum_sector_weight": max(sector_weights.values(), default=0.0),
                "mean_size_rank": float(
                    selected_raw["benchmark_weight_rank"].mean()
                ),
                "mean_liquidity_rank": float(selected_raw["amount_rank"].mean()),
            }
        )
        decomposition.append(
            {
                "date": date,
                "exited_count": len(exited),
                "entered_count": len(entered),
                "universe_churn_count": len(absent),
                "forced_exit_count": len(forced),
                "ranking_churn_count": len(ranking_exits),
                "near_cutoff_score_noise_count": len(near_cutoff),
                "cross_sector_ranking_exit_count": len(cross_sector_exits),
            }
        )
        previous = executed
        previous_sector = {
            symbol: str(lookup.loc[symbol, "broad_sector"]) for symbol in executed
        }

    periods = pd.DataFrame(rows)
    if not periods.empty and previous:
        liquidation = sum(previous.values()) * sell_rate
        idx = periods.index[-1]
        periods.loc[idx, "transaction_cost"] += liquidation
        periods.loc[idx, "net_return"] -= liquidation
        periods.loc[idx, "net_research_proxy_alpha"] -= liquidation
        periods.loc[idx, "sell_turnover"] += sum(previous.values())
    return periods, pd.DataFrame(decomposition)


def summarize_portfolio(periods: pd.DataFrame, horizon: int) -> dict:
    if periods.empty:
        return {"periods": 0}
    periods_per_year = 252 / horizon
    years = len(periods) / periods_per_year
    gross_total = float((1 + periods["gross_return"]).prod() - 1)
    net_total = float((1 + periods["net_return"]).prod() - 1)
    proxy_total = float((1 + periods["research_benchmark_proxy_return"]).prod() - 1)
    cost_free_net = periods["net_return"]
    std = float(cost_free_net.std(ddof=1))
    downside = float(cost_free_net[cost_free_net < 0].std(ddof=1))
    equity = (1 + cost_free_net.fillna(0)).cumprod()
    max_drawdown = float((equity / equity.cummax() - 1).min())
    return {
        "periods": int(len(periods)),
        "gross_total_return": gross_total,
        "transaction_cost_sum": float(periods["transaction_cost"].sum()),
        "net_total_return": net_total,
        "research_proxy_return": proxy_total,
        "gross_research_proxy_alpha": gross_total - proxy_total,
        "net_research_proxy_alpha": net_total - proxy_total,
        "net_cagr": float((1 + net_total) ** (1 / years) - 1)
        if years > 0 and net_total > -1
        else -1.0,
        "sharpe": float(cost_free_net.mean() / std * math.sqrt(periods_per_year))
        if std > 0
        else 0.0,
        "sortino": float(cost_free_net.mean() / downside * math.sqrt(periods_per_year))
        if downside > 0
        else 0.0,
        "max_drawdown": max_drawdown,
        "average_one_way_turnover": float(
            (periods["buy_turnover"] + periods["sell_turnover"]).mean() / 2
        ),
        "annualized_turnover": float(
            (periods["buy_turnover"] + periods["sell_turnover"]).mean()
            / 2
            * periods_per_year
        ),
        "average_transaction_cost": float(periods["transaction_cost"].mean()),
        "average_maximum_sector_weight": float(periods["maximum_sector_weight"].mean()),
        "average_size_rank": float(periods["mean_size_rank"].mean()),
        "average_liquidity_rank": float(periods["mean_liquidity_rank"].mean()),
        "mean_top_minus_bottom_spread": float(periods["top_minus_bottom_spread"].mean()),
    }


def staggered_sleeve_membership(
    scores_by_date: list[list[str]], horizon: int
) -> list[dict[str, float]]:
    """Return deterministic overlapping sleeve weights for algorithm tests/diagnostics.

    This helper intentionally does not claim P&L because the research cache lacks
    daily mark-to-market paths for every open sleeve. It proves the membership and
    turnover mechanism without inventing returns.
    """

    if horizon < 1:
        raise ValueError("horizon must be positive")
    sleeves: list[list[str]] = [[] for _ in range(horizon)]
    snapshots: list[dict[str, float]] = []
    for index, symbols in enumerate(scores_by_date):
        sleeves[index % horizon] = list(dict.fromkeys(map(str, symbols)))
        weights: dict[str, float] = {}
        for sleeve in sleeves:
            if not sleeve:
                continue
            for symbol in sleeve:
                weights[symbol] = weights.get(symbol, 0.0) + 1 / horizon / len(sleeve)
        snapshots.append(weights)
    return snapshots
