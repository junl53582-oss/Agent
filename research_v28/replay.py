from dataclasses import replace

import pandas as pd

from research_v16.portfolio import optimize_v16 as base_optimize
from research_v22 import replay as parent
from research_v22r1.schedule import schedule_from_parent


MODES = ("v16_replay", "v28_confidence_tail")
SCORE_COLUMNS = {"v16_replay": "v16_score", "v28_confidence_tail": "v28_score"}


def portfolio_input(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    columns = ["symbol", "eligible", "broad_sector", "benchmark_weight", "volatility_60", score_column]
    result = frame[columns].copy().rename(columns={score_column: "portfolio_score"})
    result["model_confidence"] = 1.0 if score_column == "v16_score" else float(frame["model_confidence"].iloc[0])
    return result


def confidence_optimize(current, previous_active, enabled, technology_enabled, settings):
    values = pd.to_numeric(current.pop("model_confidence"), errors="raise")
    if values.nunique() != 1 or not values.between(0.0, 1.0).all():
        raise ValueError("portfolio confidence must be one value in [0,1]")
    confidence = float(values.iloc[0])
    scaled = replace(settings, maximum_active_budget=settings.maximum_active_budget * confidence)
    desired, active, diagnostics = base_optimize(current, previous_active, enabled and confidence > 0, technology_enabled, scaled)
    diagnostics["model_confidence"] = confidence
    diagnostics["maximum_active_budget_after_confidence"] = scaled.maximum_active_budget
    return desired, active, diagnostics


def run_replay(scores, book, membership, schedule, settings, progress=None, checkpoint=None):
    old = (parent.MODES, parent.SCORE_COLUMNS, parent.portfolio_input, parent.optimize_v16)
    try:
        parent.MODES, parent.SCORE_COLUMNS = MODES, SCORE_COLUMNS
        parent.portfolio_input, parent.optimize_v16 = portfolio_input, confidence_optimize
        return parent.run_replay(scores, book, membership, schedule, settings, progress, checkpoint)
    finally:
        parent.MODES, parent.SCORE_COLUMNS, parent.portfolio_input, parent.optimize_v16 = old


load_scores = parent.load_scores
attach_volatility = parent.attach_volatility
compare_control = parent.compare_control

