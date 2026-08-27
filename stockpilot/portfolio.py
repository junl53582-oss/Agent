from __future__ import annotations

import numpy as np
import pandas as pd


def select_with_buffer_and_cap(
    current: pd.DataFrame,
    top_n: int,
    previous_symbols: set[str] | None = None,
    hold_buffer: int = 0,
    industry_cap: float = 1.0,
) -> pd.DataFrame:
    """Keep sufficiently ranked holdings, then fill by score subject to a group cap."""
    previous_symbols = previous_symbols or set()
    ranked = current.sort_values("pred_rank").copy()
    retained = ranked[
        ranked["symbol"].isin(previous_symbols)
        & (ranked["pred_rank"] <= top_n + max(hold_buffer, 0))
    ]
    candidates = (
        pd.concat([retained, ranked[~ranked.index.isin(retained.index)]])
        if not retained.empty
        else ranked
    )
    candidates["risk_group"] = (
        candidates["industry"].fillna(candidates["board"])
        if "industry" in candidates.columns
        else candidates["board"]
    )
    max_per_group = top_n if industry_cap >= 1 else max(1, int(np.floor(top_n * industry_cap)))
    selected_indexes: list[int] = []
    group_counts: dict[str, int] = {}
    for row in candidates.itertuples():
        group = str(row.risk_group)
        if group_counts.get(group, 0) >= max_per_group:
            continue
        selected_indexes.append(row.Index)
        group_counts[group] = group_counts.get(group, 0) + 1
        if len(selected_indexes) == top_n:
            break
    return current.loc[selected_indexes].sort_values("score", ascending=False).copy()


def portfolio_weights(selected: pd.DataFrame, method: str = "equal") -> pd.Series:
    if selected.empty:
        return pd.Series(dtype=float)
    if method == "equal":
        return pd.Series(1 / len(selected), index=selected.index)
    if method == "inverse_volatility":
        volatility = selected["volatility_20"].clip(lower=1e-6)
        raw = 1 / volatility
        return raw / raw.sum()
    raise ValueError("weighting 必须是 equal 或 inverse_volatility")


def turnover(previous: dict[str, float], current: dict[str, float]) -> tuple[float, float]:
    symbols = set(previous) | set(current)
    buys = sum(max(current.get(symbol, 0) - previous.get(symbol, 0), 0) for symbol in symbols)
    sells = sum(max(previous.get(symbol, 0) - current.get(symbol, 0), 0) for symbol in symbols)
    return float(buys), float(sells)
