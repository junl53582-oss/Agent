from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stockpilot.model import LightGBMRanker, RidgeRanker

from .features import V3_FEATURES


@dataclass
class FittedV3Models:
    ridge: dict[int, RidgeRanker]
    lightgbm: dict[int, LightGBMRanker]


def _mature_training(
    dataset: pd.DataFrame,
    date: pd.Timestamp,
    horizon: int,
    train_window_days: int,
) -> pd.DataFrame:
    label = f"label_{horizon}"
    end = f"label_end_date_{horizon}"
    train = dataset[
        dataset["eligible"]
        & dataset[label].notna()
        & (dataset[end] <= date)
        & (dataset["date"] < date)
    ]
    dates = train["date"].drop_duplicates().sort_values()
    if len(dates) > train_window_days:
        train = train[train["date"] >= dates.iloc[-train_window_days]]
    return train.sort_values(["date", "symbol"])


def fit_v3_models(
    dataset: pd.DataFrame,
    date: pd.Timestamp,
    horizons: tuple[int, ...],
    train_window_days: int,
) -> FittedV3Models:
    ridge: dict[int, RidgeRanker] = {}
    lightgbm: dict[int, LightGBMRanker] = {}
    for horizon in horizons:
        train = _mature_training(dataset, date, horizon, train_window_days)
        groups = train.groupby("date", sort=False).size().to_numpy()
        ridge[horizon] = RidgeRanker(alpha=20.0).fit(train[V3_FEATURES], train[f"label_{horizon}"])
        lightgbm[horizon] = LightGBMRanker().fit(
            train[V3_FEATURES], train[f"label_{horizon}"], group_sizes=groups
        )
    return FittedV3Models(ridge=ridge, lightgbm=lightgbm)


def score_v3_models(
    current: pd.DataFrame,
    models: FittedV3Models,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    scored = current.copy()
    component_columns = ["stable_factor_score"]
    for horizon in horizons:
        for name, model in [
            ("ridge", models.ridge[horizon]),
            ("lightgbm", models.lightgbm[horizon]),
        ]:
            raw = pd.Series(model.predict(scored[V3_FEATURES]), index=scored.index)
            column = f"{name}_{horizon}_score"
            scored[column] = raw.rank(pct=True, method="first") - 0.5
            component_columns.append(column)
    ridge_columns = [f"ridge_{horizon}_score" for horizon in horizons]
    lightgbm_columns = [f"lightgbm_{horizon}_score" for horizon in horizons]
    scored["ridge_score"] = scored[ridge_columns].mean(axis=1)
    scored["lightgbm_score"] = scored[lightgbm_columns].mean(axis=1)
    scored["ensemble_score"] = (
        0.20 * scored["stable_factor_score"]
        + 0.35 * scored["ridge_score"]
        + 0.45 * scored["lightgbm_score"]
    )
    scored["agreement"] = (scored[component_columns].rank(pct=True) >= 0.70).mean(axis=1)
    return scored
