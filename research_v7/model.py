from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research_v5.features import MODEL_FEATURES
from stockpilot.model import RidgeRanker

from .config import V7Settings


@dataclass
class HorizonModels:
    global_model: RidgeRanker
    experts: dict[str, RidgeRanker]


def _training(dataset: pd.DataFrame, year: int, horizon: int, settings: V7Settings) -> pd.DataFrame:
    cutoff = pd.Timestamp(year, 1, 1)
    return dataset[
        dataset["eligible"]
        & dataset[f"label_{horizon}"].notna()
        & (pd.to_datetime(dataset[f"label_end_date_{horizon}"]) < cutoff)
        & (pd.to_datetime(dataset["date"]) < cutoff)
        & (pd.to_datetime(dataset["date"]).dt.year >= year - settings.training_window_years)
    ].sort_values(["date", "symbol"])


def fit_multihorizon_models(
    dataset: pd.DataFrame, year: int, settings: V7Settings | None = None
) -> dict[int, HorizonModels]:
    settings = settings or V7Settings()
    result = {}
    for horizon in settings.horizons:
        train = _training(dataset, year, horizon, settings)
        global_model = RidgeRanker(settings.ridge_alpha).fit(
            train[MODEL_FEATURES], train[f"label_{horizon}"]
        )
        experts = {}
        for sector, group in train.groupby("broad_sector"):
            if len(group) >= 5000 and group["date"].nunique() >= 100:
                experts[str(sector)] = RidgeRanker(settings.ridge_alpha).fit(
                    group[MODEL_FEATURES], group[f"label_{horizon}"]
                )
            else:
                experts[str(sector)] = global_model
        result[horizon] = HorizonModels(global_model, experts)
    return result


def _rank(values: pd.Series) -> pd.Series:
    return values.rank(pct=True, method="average").sub(0.5).fillna(0)


def score_multihorizon(
    current: pd.DataFrame,
    models: dict[int, HorizonModels],
    settings: V7Settings | None = None,
) -> pd.DataFrame:
    settings = settings or V7Settings()
    scored = current.copy()
    columns = []
    for horizon in settings.horizons:
        fitted = models[horizon]
        global_raw = pd.Series(fitted.global_model.predict(scored[MODEL_FEATURES]), index=scored.index)
        global_score = _rank(global_raw)
        expert_score = pd.Series(0.0, index=scored.index)
        for sector, indexes in scored.groupby("broad_sector").groups.items():
            model = fitted.experts.get(str(sector), fitted.global_model)
            raw = pd.Series(model.predict(scored.loc[indexes, MODEL_FEATURES]), index=indexes)
            expert_score.loc[indexes] = _rank(raw)
        column = f"horizon_{horizon}_score"
        scored[column] = settings.global_share * global_score + settings.expert_share * expert_score
        columns.append(column)
    matrix = scored[columns]
    weights = np.asarray(settings.horizon_weights)
    scored["multihorizon_score"] = matrix.to_numpy() @ weights
    scored["horizon_uncertainty"] = matrix.std(axis=1, ddof=0)
    direction = np.sign(scored["multihorizon_score"])
    scored["horizon_agreement"] = (np.sign(matrix).eq(direction, axis=0)).mean(axis=1)
    return scored
