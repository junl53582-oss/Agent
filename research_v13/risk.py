from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import V13Settings


@dataclass
class DrawdownState:
    regime: str = "full"

    def update(self, current: pd.DataFrame, settings: V13Settings | None = None):
        settings = settings or V13Settings()
        drawdown_values = pd.to_numeric(current["v13_market_drawdown_120"], errors="coerce").dropna()
        momentum_values = pd.to_numeric(current["v12_market_momentum_60"], errors="coerce").dropna()
        drawdown = float(drawdown_values.iloc[0]) if len(drawdown_values) else 0.0
        momentum = float(momentum_values.iloc[0]) if len(momentum_values) else 0.0
        if drawdown <= settings.defensive_drawdown:
            self.regime = "defensive"
        elif self.regime == "defensive":
            if drawdown > settings.defensive_recovery_drawdown and momentum > 0:
                self.regime = "cautious"
        elif drawdown <= settings.cautious_drawdown:
            self.regime = "cautious"
        elif self.regime == "cautious" and drawdown > settings.cautious_recovery_drawdown and momentum > 0:
            self.regime = "full"
        exposure = {"full": 1.0, "cautious": settings.cautious_exposure, "defensive": settings.defensive_exposure}[self.regime]
        return exposure, {"market_drawdown_120": drawdown, "market_momentum_60": momentum, "risk_regime": self.regime}

