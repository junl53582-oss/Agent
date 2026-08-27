from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from research_v4.stability import FactorSpec
from research_v5.models import V5Models
from research_v6.config import V6Settings
from research_v6.model import score_v6
from stockpilot.model import LightGBMRanker, RidgeRanker

from .config import V9Settings
from .features import V9_FEATURES


@dataclass
class V9Models:
    ridge: RidgeRanker
    lightgbm: LightGBMRanker
    technology: RidgeRanker
    technology_fallback: bool
    training_rows: int
    training_end: pd.Timestamp


def mature_training(
    dataset: pd.DataFrame, test_year: int, settings: V9Settings
) -> pd.DataFrame:
    cutoff = pd.Timestamp(test_year, 1, 1)
    earliest = test_year - settings.training_window_years
    return dataset[
        dataset["eligible"]
        & dataset["v9_target_5"].notna()
        & (pd.to_datetime(dataset["label_end_date_5"]) < cutoff)
        & (pd.to_datetime(dataset["date"]) < cutoff)
        & (pd.to_datetime(dataset["date"]).dt.year >= earliest)
    ].sort_values(["date", "symbol"])


def fit_v9_models(
    dataset: pd.DataFrame, test_year: int, settings: V9Settings | None = None
) -> V9Models:
    settings = settings or V9Settings()
    train = mature_training(dataset, test_year, settings)
    if train.empty:
        raise RuntimeError(f"{test_year}测试折没有成熟训练数据")
    ridge = RidgeRanker(settings.ridge_alpha).fit(train[V9_FEATURES], train["v9_target_5"])
    lightgbm = LightGBMRanker().fit(train[V9_FEATURES], train["v9_target_5"])
    tech = train[train["broad_sector"] == "technology"]
    enough = (
        len(tech) >= settings.minimum_technology_rows
        and tech["date"].nunique() >= settings.minimum_technology_dates
    )
    technology = (
        RidgeRanker(settings.ridge_alpha).fit(tech[V9_FEATURES], tech["v9_target_5"])
        if enough
        else ridge
    )
    return V9Models(
        ridge=ridge,
        lightgbm=lightgbm,
        technology=technology,
        technology_fallback=not enough,
        training_rows=len(train),
        training_end=pd.to_datetime(train["label_end_date_5"]).max(),
    )


def _centered_rank(values: pd.Series) -> pd.Series:
    return values.rank(pct=True, method="average").sub(0.5).fillna(0)


def score_v9(
    current: pd.DataFrame,
    models: V9Models,
    v5_models: V5Models,
    v4_specs: list[FactorSpec],
    settings: V9Settings | None = None,
) -> pd.DataFrame:
    settings = settings or V9Settings()
    scored = score_v6(current, v5_models, v4_specs, V6Settings())
    scored["v6_score"] = scored["score"]
    ridge = _centered_rank(
        pd.Series(models.ridge.predict(scored[V9_FEATURES]), index=scored.index)
    )
    nonlinear = _centered_rank(
        pd.Series(models.lightgbm.predict(scored[V9_FEATURES]), index=scored.index)
    )
    scored["ridge_score"] = ridge
    scored["nonlinear_score"] = nonlinear
    scored["enhanced_global"] = settings.ridge_share * ridge + settings.lightgbm_share * nonlinear
    scored["enhanced_score"] = scored["enhanced_global"]
    tech = scored["broad_sector"] == "technology"
    if tech.any():
        specialist = _centered_rank(
            pd.Series(
                models.technology.predict(scored.loc[tech, V9_FEATURES]),
                index=scored.index[tech],
            )
        )
        scored.loc[tech, "technology_specialist"] = specialist
        scored.loc[tech, "enhanced_score"] = (
            settings.technology_global_share * scored.loc[tech, "enhanced_global"]
            + settings.technology_specialist_share * specialist
        )
    scored["alpha_model_score"] = (
        settings.v6_base_share * scored["v6_score"] + settings.enhanced_share * ridge
    )
    scored["nonlinear_model_score"] = (
        settings.v6_base_share * scored["v6_score"]
        + settings.enhanced_share * scored["enhanced_global"]
    )
    scored["model_score"] = (
        settings.v6_base_share * scored["v6_score"]
        + settings.enhanced_share * scored["enhanced_score"]
    )
    scored["score"] = scored["model_score"]
    return scored
