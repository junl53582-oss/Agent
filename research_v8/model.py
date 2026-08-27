from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from research_v4.stability import FactorSpec
from research_v5.models import V5Models
from research_v6.config import V6Settings
from research_v6.model import score_v6
from stockpilot.model import RidgeRanker

from .config import V8Settings
from .features import ENHANCED_FEATURES


@dataclass
class V8Models:
    global_model: RidgeRanker
    technology_model: RidgeRanker
    technology_fallback: bool
    training_rows: int
    training_end: pd.Timestamp


def mature_training(
    dataset: pd.DataFrame, test_year: int, settings: V8Settings
) -> pd.DataFrame:
    cutoff = pd.Timestamp(test_year, 1, 1)
    return dataset[
        dataset["eligible"]
        & dataset["label_5"].notna()
        & (pd.to_datetime(dataset["label_end_date_5"]) < cutoff)
        & (pd.to_datetime(dataset["date"]) < cutoff)
        & (pd.to_datetime(dataset["date"]).dt.year >= test_year - settings.training_window_years)
    ].sort_values(["date", "symbol"])


def fit_v8_models(
    dataset: pd.DataFrame, test_year: int, settings: V8Settings | None = None
) -> V8Models:
    settings = settings or V8Settings()
    train = mature_training(dataset, test_year, settings)
    if train.empty:
        raise RuntimeError(f"{test_year}测试折没有成熟训练数据")
    global_model = RidgeRanker(settings.ridge_alpha).fit(
        train[ENHANCED_FEATURES], train["label_5"]
    )
    technology = train[train["broad_sector"] == "technology"]
    enough = (
        len(technology) >= settings.minimum_technology_rows
        and technology["date"].nunique() >= settings.minimum_technology_dates
    )
    technology_model = (
        RidgeRanker(settings.ridge_alpha).fit(
            technology[ENHANCED_FEATURES], technology["label_5"]
        )
        if enough
        else global_model
    )
    return V8Models(
        global_model=global_model,
        technology_model=technology_model,
        technology_fallback=not enough,
        training_rows=len(train),
        training_end=pd.to_datetime(train["label_end_date_5"]).max(),
    )


def _centered_rank(values: pd.Series) -> pd.Series:
    return values.rank(pct=True, method="average").sub(0.5).fillna(0.0)


def score_v8(
    current: pd.DataFrame,
    v8_models: V8Models,
    v5_models: V5Models,
    v4_specs: list[FactorSpec],
    settings: V8Settings | None = None,
) -> pd.DataFrame:
    settings = settings or V8Settings()
    scored = score_v6(current, v5_models, v4_specs, V6Settings())
    scored["v6_score"] = scored["score"]
    global_raw = pd.Series(
        v8_models.global_model.predict(scored[ENHANCED_FEATURES]), index=scored.index
    )
    scored["enhanced_global"] = _centered_rank(global_raw)
    scored["technology_specialist"] = scored["enhanced_global"]
    tech = scored["broad_sector"] == "technology"
    if tech.any():
        specialist_raw = pd.Series(
            v8_models.technology_model.predict(scored.loc[tech, ENHANCED_FEATURES]),
            index=scored.index[tech],
        )
        scored.loc[tech, "technology_specialist"] = _centered_rank(specialist_raw)
    scored["enhanced_score"] = scored["enhanced_global"]
    scored.loc[tech, "enhanced_score"] = (
        settings.technology_specialist_share * scored.loc[tech, "technology_specialist"]
        + settings.technology_global_share * scored.loc[tech, "enhanced_global"]
    )
    scored["model_score"] = (
        settings.v6_base_share * scored["v6_score"]
        + settings.enhanced_global_share * scored["enhanced_score"]
    )
    regime = str(scored["regime"].iloc[0])
    if regime == "risk_on":
        scored["model_score"] += settings.risk_on_momentum_tilt * (
            0.5 * scored["momentum"] + 0.5 * scored["industry_momentum"]
        )
    elif regime == "risk_off":
        scored["model_score"] += settings.risk_off_quality_tilt * (
            0.5 * scored["quality"] + 0.5 * scored["low_volatility"]
        )
    scored["score"] = scored["model_score"]
    return scored

