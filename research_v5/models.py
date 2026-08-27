from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stockpilot.model import RidgeRanker

from .config import V5Settings
from .features import MODEL_FEATURES

COMPONENT_WEIGHTS = {
    "risk_on": {
        "fundamental": 0.20,
        "behavior": 0.25,
        "risk": 0.05,
        "liquidity_component": 0.05,
        "global_model": 0.25,
        "industry_expert": 0.20,
    },
    "neutral": {
        "fundamental": 0.25,
        "behavior": 0.15,
        "risk": 0.10,
        "liquidity_component": 0.05,
        "global_model": 0.25,
        "industry_expert": 0.20,
    },
    "risk_off": {
        "fundamental": 0.30,
        "behavior": 0.05,
        "risk": 0.20,
        "liquidity_component": 0.05,
        "global_model": 0.25,
        "industry_expert": 0.15,
    },
}


@dataclass
class V5Models:
    global_model: RidgeRanker
    experts: dict[str, RidgeRanker]
    expert_fallback: set[str]
    training_rows: int
    training_end: pd.Timestamp


def mature_training(dataset: pd.DataFrame, test_year: int, settings: V5Settings) -> pd.DataFrame:
    cutoff = pd.Timestamp(test_year, 1, 1)
    earliest = test_year - settings.training_window_years
    return dataset[
        dataset["eligible"]
        & dataset["label_5"].notna()
        & (pd.to_datetime(dataset["label_end_date_5"]) < cutoff)
        & (pd.to_datetime(dataset["date"]) < cutoff)
        & (pd.to_datetime(dataset["date"]).dt.year >= earliest)
    ].sort_values(["date", "symbol"])


def fit_v5_models(
    dataset: pd.DataFrame, test_year: int, settings: V5Settings | None = None
) -> V5Models:
    settings = settings or V5Settings()
    train = mature_training(dataset, test_year, settings)
    if train.empty:
        raise RuntimeError(f"{test_year}测试折没有成熟训练数据")
    global_model = RidgeRanker(settings.ridge_alpha).fit(train[MODEL_FEATURES], train["label_5"])
    experts: dict[str, RidgeRanker] = {}
    fallback = set()
    for sector, group in train.groupby("broad_sector"):
        enough = (
            len(group) >= settings.minimum_expert_rows
            and group["date"].nunique() >= settings.minimum_expert_dates
        )
        if enough:
            experts[str(sector)] = RidgeRanker(settings.ridge_alpha).fit(
                group[MODEL_FEATURES], group["label_5"]
            )
        else:
            experts[str(sector)] = global_model
            fallback.add(str(sector))
    return V5Models(
        global_model=global_model,
        experts=experts,
        expert_fallback=fallback,
        training_rows=len(train),
        training_end=pd.to_datetime(train["label_end_date_5"]).max(),
    )


def _centered_rank(values: pd.Series) -> pd.Series:
    return values.rank(pct=True, method="average").sub(0.5).fillna(0)


def score_v5(current: pd.DataFrame, models: V5Models) -> pd.DataFrame:
    scored = current.copy()
    global_raw = pd.Series(
        models.global_model.predict(scored[MODEL_FEATURES]), index=scored.index
    )
    scored["global_model"] = _centered_rank(global_raw)
    scored["industry_expert"] = 0.0
    for sector, indexes in scored.groupby("broad_sector").groups.items():
        model = models.experts.get(str(sector), models.global_model)
        raw = pd.Series(model.predict(scored.loc[indexes, MODEL_FEATURES]), index=indexes)
        scored.loc[indexes, "industry_expert"] = _centered_rank(raw)
    scored["fundamental"] = 0.65 * scored["quality"] + 0.35 * scored["growth"]
    scored["behavior"] = (
        0.55 * scored["momentum"]
        + 0.25 * scored["short_reversal"]
        + 0.20 * scored["volume_attention"]
    )
    scored["risk"] = scored["low_volatility"]
    scored["liquidity_component"] = scored["liquidity"]
    regimes = scored["regime"].dropna().unique()
    if len(regimes) != 1:
        raise RuntimeError("单日截面出现多个市场状态")
    regime = str(regimes[0])
    weights = COMPONENT_WEIGHTS[regime]
    scored["score"] = sum(scored[column] * weight for column, weight in weights.items())
    return scored


def model_diagnostics(test_year: int, models: V5Models) -> list[dict]:
    rows = []
    all_models = {"global": models.global_model, **models.experts}
    for sector, model in all_models.items():
        weights = model.feature_weights(MODEL_FEATURES)
        for feature, coefficient in weights.items():
            rows.append(
                {
                    "test_year": test_year,
                    "model_scope": sector,
                    "feature": feature,
                    "coefficient": coefficient,
                    "global_fallback": sector in models.expert_fallback,
                    "training_rows": models.training_rows,
                    "training_end": models.training_end,
                }
            )
    return rows
