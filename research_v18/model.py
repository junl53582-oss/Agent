from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research_v10.model import score_v10
from research_v10.research_config import V10Settings
from research_v12.model import mature_embargoed_training
from research_v13.config import V13Settings
from research_v13.model import TwoStageModel
from research_v16.model import _sector_rank, baseline_parts_for_year

from .config import V18Settings
from .text_model import EmbeddingTextModel


@dataclass
class V18Models:
    text_model: EmbeddingTextModel
    baseline_model: TwoStageModel
    v10: object
    training_events: int
    raw_event_years: list[int]


def fit_v18_models(
    dataset: pd.DataFrame,
    events: pd.DataFrame,
    embeddings: np.ndarray,
    test_year: int,
    settings: V18Settings | None = None,
    baseline_cache: dict | None = None,
) -> V18Models:
    settings = settings or V18Settings()
    earliest = test_year - settings.training_window_years
    train = mature_embargoed_training(
        dataset, test_year, earliest, settings.embargo_calendar_days
    )
    baseline_cache = {} if baseline_cache is None else baseline_cache
    text_model = EmbeddingTextModel.fit(events, embeddings, test_year, earliest, settings)
    return V18Models(
        text_model=text_model,
        baseline_model=TwoStageModel().fit(train, V13Settings()),
        v10=baseline_parts_for_year(dataset, test_year, settings, baseline_cache)[0],
        training_events=text_model.training_events,
        raw_event_years=text_model.event_years,
    )


def score_v18(
    current: pd.DataFrame,
    models: V18Models,
    v5_models,
    v4_specs,
    settings: V18Settings | None = None,
) -> pd.DataFrame:
    settings = settings or V18Settings()
    return _score_components(
        current, models.baseline_model, models.text_model,
        (models.v10, v5_models, v4_specs), settings,
    )


def _score_components(current, baseline_model, text_model, baseline_parts, settings):
    v10, v5_models, v4_specs = baseline_parts
    scored = score_v10(current, v10, v5_models, v4_specs, V10Settings())
    baseline_probability, baseline_magnitude = baseline_model.predict_components(scored)
    baseline_raw = baseline_probability * np.clip(baseline_magnitude, 0.0, 0.10)
    scored["v13_baseline_score"] = _sector_rank(scored, baseline_raw)
    v13_settings = V13Settings()
    scored["v13_comparable_score"] = (
        v13_settings.two_stage_share * scored["v13_baseline_score"]
        + v13_settings.v10_global_share * scored["global_model_score"]
    )
    recent = text_model.recent_scores(scored, settings).set_index("symbol")
    scored["text_event_score"] = scored["symbol"].map(recent["text_score"]).fillna(0.0)
    scored["recent_text_events"] = scored["symbol"].map(recent["text_events"]).fillna(0).astype(int)
    scored["v18_score"] = (
        settings.baseline_share * scored["v13_comparable_score"]
        + settings.text_share * scored["text_event_score"]
    )
    scored["score"] = scored["v18_score"]
    return scored
