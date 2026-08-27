from __future__ import annotations

import pandas as pd

from .config import V11Settings


def defensive_exposure(
    current: pd.DataFrame, settings: V11Settings | None = None
) -> tuple[float, dict[str, float | str]]:
    settings = settings or V11Settings()
    universe = current[current["in_universe"].fillna(False)].copy()
    weights = pd.to_numeric(universe["benchmark_weight"], errors="coerce").clip(lower=0)
    momentum = pd.to_numeric(universe["momentum_60"], errors="coerce")
    ret20 = pd.to_numeric(universe["ret_20"], errors="coerce")
    valid_momentum = momentum.notna() & (weights > 0)
    market_momentum = (
        float((momentum[valid_momentum] * weights[valid_momentum]).sum() / weights[valid_momentum].sum())
        if valid_momentum.any()
        else 0.0
    )
    breadth = float((ret20.dropna() > 0).mean()) if ret20.notna().any() else 0.5
    if market_momentum < settings.risk_off_momentum and breadth < settings.risk_off_breadth:
        exposure, regime = settings.risk_off_exposure, "risk_off"
    elif market_momentum < settings.weak_momentum or breadth < settings.weak_breadth:
        exposure, regime = settings.weak_exposure, "weak"
    else:
        exposure, regime = settings.risk_on_exposure, "risk_on"
    return exposure, {
        "market_momentum_60": market_momentum,
        "positive_breadth_20": breadth,
        "risk_regime": regime,
    }

