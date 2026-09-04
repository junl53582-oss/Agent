from __future__ import annotations

from dataclasses import dataclass

from .lineage import ModelOutput


@dataclass(frozen=True)
class ProductEvidence:
    """Optional, already-normalized product evidence; absent evidence stays absent."""

    model_outputs: tuple[ModelOutput, ...] = ()
    historical_calibration: float | None = None
    data_completeness: float | None = None
    feature_completeness: float | None = None
    drift_ood_quality: float | None = None
    regime_familiarity: float | None = None
    historical_stability: float | None = None
    volatility_risk: float | None = None
    drawdown_risk: float | None = None
    liquidity_risk: float | None = None
    drift_ood_risk: float | None = None
    regime_risk: float | None = None
    data_quality_risk: float | None = None
    industry_score: float | None = None
    regime_score: float | None = None

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if name == "model_outputs" or value is None:
                continue
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
