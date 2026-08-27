from __future__ import annotations

import numpy as np
import pandas as pd

from .config import V12Settings


def risk_budget_exposure(
    current: pd.DataFrame, settings: V12Settings | None = None
) -> tuple[float, dict[str, float | str]]:
    settings = settings or V12Settings()
    volatility = pd.to_numeric(current["market_volatility_60"], errors="coerce").dropna()
    momentum = pd.to_numeric(current["v12_market_momentum_60"], errors="coerce").dropna()
    market_volatility = float(volatility.iloc[0]) if len(volatility) else settings.risk_target_annual_volatility
    market_momentum = float(momentum.iloc[0]) if len(momentum) else 0.0
    if market_momentum >= 0 or not np.isfinite(market_volatility) or market_volatility <= 0:
        exposure = settings.maximum_equity_exposure
        regime = "full"
    else:
        exposure = float(
            np.clip(
                settings.risk_target_annual_volatility / market_volatility,
                settings.minimum_equity_exposure,
                settings.maximum_equity_exposure,
            )
        )
        regime = "risk_budget"
    return exposure, {
        "market_volatility_60": market_volatility,
        "market_momentum_60": market_momentum,
        "risk_regime": regime,
    }
