from __future__ import annotations

from dataclasses import replace

import pandas as pd

from research_v10.portfolio import benchmark_weights, optimize_benchmark_relative
from research_v10.research_config import V10Settings

from .config import V11Settings


def optimize_v11(
    current: pd.DataFrame,
    previous_active: set[str],
    global_gate: bool,
    technology_gate: bool,
    settings: V11Settings | None = None,
) -> tuple[dict[str, float], set[str], dict]:
    settings = settings or V11Settings()
    if not global_gate:
        return benchmark_weights(current), set(), {
            "active_budget": 0.0,
            "ex_ante_tracking_error": 0.0,
            "maximum_stock_active_weight": 0.0,
            "maximum_sector_deviation": 0.0,
            "active_holdings": 0,
        }
    v10_settings = replace(
        V10Settings(),
        maximum_active_budget=settings.maximum_active_budget,
        maximum_stock_active_weight=settings.maximum_stock_active_weight,
        maximum_ex_ante_tracking_error=settings.maximum_ex_ante_tracking_error,
        active_top_n=settings.active_top_n,
        holding_bonus=settings.holding_bonus,
    )
    return optimize_benchmark_relative(
        current, previous_active, 1.0, technology_gate, v10_settings
    )


def apply_exposure(weights: dict[str, float], exposure: float) -> dict[str, float]:
    return {symbol: float(weight) * exposure for symbol, weight in weights.items()}

