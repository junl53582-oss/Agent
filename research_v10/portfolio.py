from __future__ import annotations

import numpy as np
import pandas as pd

from .research_config import V10Settings


def _allocate_capped(
    capacities: pd.Series, total: float, preference: pd.Series
) -> pd.Series:
    allocation = pd.Series(0.0, index=capacities.index)
    remaining = min(float(total), float(capacities.clip(lower=0).sum()))
    for _ in range(20):
        available = capacities - allocation
        active = available > 1e-12
        if remaining <= 1e-12 or not active.any():
            break
        weights = preference.loc[active].clip(lower=0)
        if weights.sum() <= 0:
            weights = pd.Series(1.0, index=weights.index)
        proposal = weights / weights.sum() * remaining
        addition = proposal.clip(upper=available.loc[active])
        allocation.loc[active] += addition
        consumed = float(addition.sum())
        remaining -= consumed
        if consumed <= 1e-12:
            break
    return allocation


def benchmark_weights(current: pd.DataFrame) -> dict[str, float]:
    base = current[current["benchmark_weight"] > 0].set_index("symbol")["benchmark_weight"]
    total = float(base.sum())
    return (base / total).to_dict() if total > 0 else {}


def optimize_benchmark_relative(
    current: pd.DataFrame,
    previous_active: set[str],
    confidence: float,
    technology_enabled: bool,
    settings: V10Settings | None = None,
) -> tuple[dict[str, float], set[str], dict]:
    settings = settings or V10Settings()
    base_map = benchmark_weights(current)
    if not base_map:
        return {}, set(), {"active_budget": 0.0, "ex_ante_tracking_error": 0.0}
    indexed = current.drop_duplicates("symbol").set_index("symbol").copy()
    base = pd.Series(base_map).reindex(indexed.index).fillna(0.0)
    candidates = indexed[indexed["eligible"].fillna(False)].copy()
    if not technology_enabled:
        candidates = candidates[candidates["broad_sector"] != "technology"]
    candidates["selection_score"] = candidates["portfolio_score"] + (
        candidates.index.to_series().isin(previous_active).astype(float) * settings.holding_bonus
    )
    active_budget = settings.maximum_active_budget * float(np.clip(confidence, 0.0, 1.0))
    if candidates.empty or active_budget <= 0:
        return base_map, set(), {"active_budget": 0.0, "ex_ante_tracking_error": 0.0}

    sector_base = base.groupby(indexed["broad_sector"]).sum()
    selected_parts = []
    for sector, sector_weight in sector_base.items():
        group = candidates[candidates["broad_sector"] == sector]
        if group.empty or sector_weight <= 0:
            continue
        count = max(1, int(round(settings.active_top_n * float(sector_weight))))
        selected_parts.append(group.nlargest(min(count, len(group)), "selection_score"))
    selected = (
        pd.concat(selected_parts).sort_values("selection_score", ascending=False)
        if selected_parts
        else candidates.iloc[0:0]
    )
    selected = selected[~selected.index.duplicated(keep="first")].head(settings.active_top_n)
    tilt = pd.Series(0.0, index=indexed.index)
    for sector, sector_weight in sector_base.items():
        chosen = selected[selected["broad_sector"] == sector]
        donors = indexed[
            (indexed["broad_sector"] == sector) & ~indexed.index.isin(chosen.index)
        ]
        if chosen.empty or donors.empty:
            continue
        target = active_budget * float(sector_weight)
        positive_capacity = pd.Series(
            settings.maximum_stock_active_weight, index=chosen.index, dtype=float
        )
        volatility = pd.to_numeric(chosen.get("volatility_60"), errors="coerce").clip(lower=1e-4)
        preference = 1 / volatility.fillna(volatility.median() if volatility.notna().any() else 0.02)
        additions = _allocate_capped(positive_capacity, target, preference)
        negative_capacity = pd.concat(
            [
                base.loc[donors.index],
                pd.Series(settings.maximum_stock_active_weight, index=donors.index),
            ],
            axis=1,
        ).min(axis=1)
        deductions = _allocate_capped(
            negative_capacity, float(additions.sum()), base.loc[donors.index]
        )
        matched = min(float(additions.sum()), float(deductions.sum()))
        if matched <= 0:
            continue
        if additions.sum() > matched:
            additions *= matched / additions.sum()
        if deductions.sum() > matched:
            deductions *= matched / deductions.sum()
        tilt.loc[additions.index] += additions
        tilt.loc[deductions.index] -= deductions

    volatility = pd.to_numeric(indexed.get("volatility_60"), errors="coerce").fillna(0.02)
    ex_ante_te = float(np.sqrt(np.sum((tilt * volatility) ** 2) * 252))
    if ex_ante_te > settings.maximum_ex_ante_tracking_error and ex_ante_te > 0:
        scale = settings.maximum_ex_ante_tracking_error / ex_ante_te
        tilt *= scale
        ex_ante_te = settings.maximum_ex_ante_tracking_error
    desired = (base + tilt).clip(lower=0)
    desired /= desired.sum()
    active_symbols = set(tilt[tilt > 1e-10].index.astype(str))
    sector_deviation = tilt.groupby(indexed["broad_sector"]).sum().abs().max()
    diagnostics = {
        "active_budget": float(tilt.clip(lower=0).sum()),
        "ex_ante_tracking_error": ex_ante_te,
        "maximum_stock_active_weight": float(tilt.abs().max()),
        "maximum_sector_deviation": float(sector_deviation),
        "active_holdings": len(active_symbols),
    }
    return desired[desired > 0].to_dict(), active_symbols, diagnostics

