from __future__ import annotations

import math

import numpy as np
import pandas as pd

from stockpilot.portfolio import turnover


def daily_rank_metrics(
    frame: pd.DataFrame, score_column: str, return_column: str
) -> pd.DataFrame:
    rows = []
    for date, group in frame.groupby("date", sort=True):
        valid = group[[score_column, return_column]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 20 or valid[score_column].nunique() < 2:
            continue
        rows.append(
            {
                "date": pd.Timestamp(date),
                "rank_ic": float(valid[score_column].corr(valid[return_column], method="spearman")),
                "pearson_ic": float(valid[score_column].corr(valid[return_column], method="pearson")),
                "sample_size": int(len(valid)),
            }
        )
    return pd.DataFrame(rows)


def summarize_ic(daily: pd.DataFrame) -> dict:
    values = pd.to_numeric(daily["rank_ic"], errors="coerce").dropna()
    pearson = pd.to_numeric(daily["pearson_ic"], errors="coerce").dropna()
    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    return {
        "dates": int(len(values)),
        "mean_rank_ic": float(values.mean()) if len(values) else float("nan"),
        "rank_ic_std": std,
        "rank_ic_ir": float(values.mean() / std) if np.isfinite(std) and std > 0 else 0.0,
        "positive_rank_ic_ratio": float((values > 0).mean()) if len(values) else float("nan"),
        "mean_pearson_ic": float(pearson.mean()) if len(pearson) else float("nan"),
    }


def _execution_columns(horizon: int) -> tuple[str, str]:
    if horizon == 5:
        return "entry_tradable", "execution_return"
    if horizon == 20:
        return "entry_tradable_20", "execution_return_20"
    return "entry_tradable", "future_return_1d"


def evaluate_topk(
    frame: pd.DataFrame,
    score_column: str,
    horizon: int,
    k: int,
    *,
    rebalance_every: int,
    buy_rate: float,
    sell_rate: float,
) -> pd.DataFrame:
    return_column = f"future_return_{horizon}d"
    tradable_column, execution_column = _execution_columns(horizon)
    dates = pd.DatetimeIndex(frame["date"].drop_duplicates().sort_values())[::rebalance_every]
    previous: dict[str, float] = {}
    rows = []
    for date in dates:
        current = frame[frame["date"].eq(date)].copy()
        current = current.dropna(subset=[score_column, return_column])
        if len(current) < max(k, 30):
            continue
        selected = current.nlargest(k, score_column).copy()
        bottom = current.nsmallest(k, score_column)
        desired = {symbol: 1.0 / k for symbol in selected["symbol"]}
        lookup = selected.set_index("symbol")
        executed: dict[str, float] = {}
        realized: dict[str, float] = {}
        for symbol, weight in desired.items():
            row = lookup.loc[symbol]
            continuing = symbol in previous
            can_enter = bool(row.get(tradable_column, False))
            value = row[return_column] if continuing else row[execution_column]
            if (continuing or can_enter) and pd.notna(value) and np.isfinite(float(value)):
                executed[str(symbol)] = float(weight)
                realized[str(symbol)] = float(value)
        buys, sells = turnover(previous, executed)
        cost = buys * buy_rate + sells * sell_rate
        gross_return = float(sum(executed[s] * realized[s] for s in executed))
        benchmark = current[current["benchmark_weight"].gt(0)].copy()
        weight_total = float(benchmark["benchmark_weight"].sum())
        proxy_return = (
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
        top_raw = float(selected[return_column].mean())
        bottom_raw = float(bottom[return_column].mean())
        rows.append(
            {
                "date": date,
                "gross_return": gross_return,
                "net_return": gross_return - cost,
                "research_benchmark_proxy_return": proxy_return,
                "gross_alpha": gross_return - proxy_return,
                "net_alpha": gross_return - cost - proxy_return,
                "top_minus_bottom_spread": top_raw - bottom_raw,
                "buy_turnover": buys,
                "sell_turnover": sells,
                "transaction_cost": cost,
                "executed_symbols": len(executed),
                "cash_weight": 1 - sum(executed.values()),
                "maximum_sector_weight": float(
                    selected["broad_sector"].value_counts(normalize=True).max()
                ),
                "mean_size_rank": float(selected["benchmark_weight_rank"].mean()),
                "mean_liquidity_rank": float(selected["amount_rank"].mean()),
            }
        )
        previous = executed
    result = pd.DataFrame(rows)
    if not result.empty and previous:
        liquidation = sum(previous.values()) * sell_rate
        idx = result.index[-1]
        result.loc[idx, "transaction_cost"] += liquidation
        result.loc[idx, "net_return"] -= liquidation
        result.loc[idx, "net_alpha"] -= liquidation
        result.loc[idx, "sell_turnover"] += sum(previous.values())
    return result


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns.fillna(0)).cumprod()
    return float((equity / equity.cummax() - 1).min()) if len(equity) else float("nan")


def summarize_topk(periods: pd.DataFrame, horizon: int) -> dict:
    if periods.empty:
        return {"periods": 0}
    net = periods["net_return"]
    gross = periods["gross_return"]
    periods_per_year = 252 / horizon
    years = len(periods) / periods_per_year
    gross_total = float((1 + gross).prod() - 1)
    net_total = float((1 + net).prod() - 1)
    benchmark_total = float((1 + periods["research_benchmark_proxy_return"]).prod() - 1)
    std = float(net.std(ddof=1))
    downside = float(net[net < 0].std(ddof=1))
    monthly = periods.set_index("date")["net_return"].resample("ME").apply(
        lambda values: (1 + values).prod() - 1
    )
    annual = periods.assign(year=periods["date"].dt.year).groupby("year")["net_return"].apply(
        lambda values: (1 + values).prod() - 1
    )
    return {
        "periods": int(len(periods)),
        "gross_total_return": gross_total,
        "net_total_return": net_total,
        "research_benchmark_proxy_return": benchmark_total,
        "gross_alpha_total": gross_total - benchmark_total,
        "net_alpha_total": net_total - benchmark_total,
        "net_cagr": float((1 + net_total) ** (1 / years) - 1) if years > 0 and net_total > -1 else -1.0,
        "sharpe": float(net.mean() / std * math.sqrt(periods_per_year)) if std > 0 else 0.0,
        "sortino": float(net.mean() / downside * math.sqrt(periods_per_year)) if downside > 0 else 0.0,
        "max_drawdown": _max_drawdown(net),
        "volatility": float(std * math.sqrt(periods_per_year)) if std > 0 else 0.0,
        "worst_month": float(monthly.min()) if len(monthly) else float("nan"),
        "worst_year": float(annual.min()) if len(annual) else float("nan"),
        "mean_net_alpha": float(periods["net_alpha"].mean()),
        "mean_top_minus_bottom_spread": float(periods["top_minus_bottom_spread"].mean()),
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
    }


def quantile_returns(
    frame: pd.DataFrame, score_column: str, return_column: str, quantiles: int = 5
) -> pd.DataFrame:
    rows = []
    for date, group in frame.groupby("date", sort=True):
        valid = group[[score_column, return_column]].dropna().copy()
        if len(valid) < quantiles * 10 or valid[score_column].nunique() < quantiles:
            continue
        valid["quantile"] = pd.qcut(
            valid[score_column].rank(method="first"), quantiles, labels=False
        ) + 1
        for quantile, part in valid.groupby("quantile"):
            rows.append(
                {
                    "date": date,
                    "quantile": int(quantile),
                    "actual_return": float(part[return_column].mean()),
                    "sample_size": int(len(part)),
                }
            )
    return pd.DataFrame(rows)


def moving_block_bootstrap_delta(
    challenger: pd.Series,
    champion: pd.Series,
    *,
    replications: int,
    block_length: int,
    seed: int,
) -> dict:
    paired = pd.concat([challenger.rename("challenger"), champion.rename("champion")], axis=1).dropna()
    delta = (paired["challenger"] - paired["champion"]).to_numpy(dtype=float)
    if len(delta) < block_length * 2:
        return {"samples": int(len(delta)), "mean_delta": float(np.mean(delta)) if len(delta) else np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan}
    rng = np.random.default_rng(seed)
    starts = np.arange(0, len(delta) - block_length + 1)
    estimates = np.empty(replications, dtype=float)
    blocks_needed = int(np.ceil(len(delta) / block_length))
    for index in range(replications):
        selected = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([delta[start : start + block_length] for start in selected])[: len(delta)]
        estimates[index] = sample.mean()
    return {
        "samples": int(len(delta)),
        "mean_delta": float(delta.mean()),
        "ci_lower": float(np.quantile(estimates, 0.025)),
        "ci_upper": float(np.quantile(estimates, 0.975)),
        "replications": replications,
        "block_length": block_length,
        "seed": seed,
    }
