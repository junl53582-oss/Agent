from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PredictionRecord:
    date: str
    symbol: str
    name: str
    close: float
    p_up_1d_raw: float
    p_up_1d: float
    p_up_5d_raw: float
    p_up_5d: float
    p_up_20d_raw: float
    p_up_20d: float
    expected_return_5d: float
    expected_return_20d: float
    rank_1d: int
    rank_5d: int
    rank_20d: int
    confidence_score: float
    confidence_level: str
    risk_level: str
    regime: str
    broad_sector: str
    prediction_ready: bool
    calibration_status: str
    drift_status: str
    candidate_score: float
    ranking_component: float
    probability_component: float
    expected_return_component: float
    risk_penalty: float
    model_version: str
    training_cutoff: str
    generated_at_utc: str
    execution_authorized: bool = False

    def to_dict(self) -> dict:
        return asdict(self)
