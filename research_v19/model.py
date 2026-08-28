from __future__ import annotations

import pandas as pd

from research_v16.model import fit_v16_models, score_v16

from .config import V19Settings


def regime_weight(momentum: float, settings: V19Settings | None = None) -> float:
    """市场状态 → 基线权重。上涨偏文本进攻，下跌偏基线防守。"""
    settings = settings or V19Settings()
    if momentum > settings.bull_threshold:
        return settings.weight_bull
    if momentum < settings.bear_threshold:
        return settings.weight_bear
    return settings.weight_neutral


def regime_name(momentum: float, settings: V19Settings | None = None) -> str:
    settings = settings or V19Settings()
    if momentum > settings.bull_threshold:
        return "bull"
    if momentum < settings.bear_threshold:
        return "bear"
    return "neutral"


def apply_v19_weights(
    scored: pd.DataFrame, momentum: float, settings: V19Settings | None = None
) -> pd.DataFrame:
    """在已打分的截面(V16)上，按市场状态用自适应权重组合成 v19_score。"""
    settings = settings or V19Settings()
    weight = regime_weight(momentum, settings)
    scored = scored.copy()
    scored["baseline_weight"] = weight
    scored["market_regime"] = regime_name(momentum, settings)
    scored["v19_score"] = (
        weight * scored["v13_comparable_score"]
        + (1.0 - weight) * scored["text_event_score"]
    )
    return scored


__all__ = ["fit_v16_models", "score_v16", "apply_v19_weights", "regime_weight", "regime_name"]
