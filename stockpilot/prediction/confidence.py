from __future__ import annotations

import numpy as np
import pandas as pd


def confidence_scores(
    probability: pd.Series,
    *,
    oos_skill: float,
    calibration_quality: float,
    regime_consistency: float,
    sector_stability: pd.Series | float,
    feature_completeness: pd.Series,
    drift_multiplier: float,
    low_upper: float = 0.40,
    medium_upper: float = 0.70,
) -> tuple[pd.Series, pd.Series]:
    """Confidence is evidence quality, deliberately distinct from win probability."""
    p = pd.to_numeric(probability, errors="coerce").clip(0, 1)
    distance = ((p - 0.5).abs() * 2).clip(0, 1)
    sector = pd.Series(sector_stability, index=p.index) if np.isscalar(sector_stability) else sector_stability.reindex(p.index)
    score = (
        0.22 * float(np.clip(oos_skill, 0, 1))
        + 0.18 * float(np.clip(calibration_quality, 0, 1))
        + 0.15 * float(np.clip(regime_consistency, 0, 1))
        + 0.10 * sector.fillna(0).clip(0, 1)
        + 0.15 * distance.fillna(0)
        + 0.20 * feature_completeness.fillna(0).clip(0, 1)
    ) * float(np.clip(drift_multiplier, 0, 1))
    score = score.clip(0, 1)
    level = pd.cut(
        score,
        bins=[-np.inf, low_upper, medium_upper, np.inf],
        labels=["LOW", "MEDIUM", "HIGH"],
        right=False,
    ).astype(str)
    return score, level

