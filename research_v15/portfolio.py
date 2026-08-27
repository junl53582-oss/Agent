from __future__ import annotations

from dataclasses import replace

from research_v10.portfolio import benchmark_weights, optimize_benchmark_relative
from research_v10.research_config import V10Settings

from .config import V15Settings


def optimize_v15(current, previous_active, enabled, technology_enabled, settings=None):
    settings = settings or V15Settings()
    if not enabled:
        return benchmark_weights(current), set(), {
            "active_budget": 0.0,
            "ex_ante_tracking_error": 0.0,
            "maximum_stock_active_weight": 0.0,
            "maximum_sector_deviation": 0.0,
            "active_holdings": 0,
        }
    compatible = replace(
        V10Settings(),
        maximum_active_budget=settings.maximum_active_budget,
        maximum_stock_active_weight=settings.maximum_stock_active_weight,
        maximum_ex_ante_tracking_error=settings.maximum_ex_ante_tracking_error,
        active_top_n=settings.active_top_n,
        holding_bonus=settings.holding_bonus,
    )
    return optimize_benchmark_relative(
        current, previous_active, 1.0, technology_enabled, compatible
    )
