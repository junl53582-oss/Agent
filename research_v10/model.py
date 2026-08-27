from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research_v4.stability import FactorSpec
from research_v5.models import V5Models
from research_v6.config import V6Settings
from research_v6.model import score_v6
from stockpilot.model import LightGBMRanker, RidgeRanker

from .features import V10_FEATURES
from .research_config import V10Settings


@dataclass
class V10Models:
    ridge_5: RidgeRanker
    lightgbm_5: LightGBMRanker
    ridge_20: RidgeRanker
    lightgbm_20: LightGBMRanker
    technology_5: RidgeRanker
    technology_20: RidgeRanker
    technology_enabled: bool
    validation_year_ics: dict[int, dict[str, float]]
    validation_ic: float
    confidence: float
    training_rows_5: int
    training_rows_20: int
    training_end: pd.Timestamp


def mature_training(
    dataset: pd.DataFrame,
    cutoff_year: int,
    target: str,
    label_end: str,
    earliest_year: int,
) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff_year, 1, 1)
    dates = pd.to_datetime(dataset["date"])
    return dataset[
        dataset["eligible"].fillna(False)
        & dataset[target].notna()
        & (pd.to_datetime(dataset[label_end]) < cutoff)
        & (dates < cutoff)
        & (dates.dt.year >= earliest_year)
    ].sort_values(["date", "symbol"])


def _daily_ic(frame: pd.DataFrame, prediction: pd.Series, target: str) -> float:
    values = []
    evaluation = frame.copy()
    evaluation["prediction"] = prediction
    for _, group in evaluation.groupby("date"):
        valid = group[["prediction", target]].dropna()
        if len(valid) < 10 or valid["prediction"].nunique() < 2:
            continue
        value = valid["prediction"].corr(valid[target], method="spearman")
        if pd.notna(value):
            values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def _rank_by_date(frame: pd.DataFrame, prediction: np.ndarray) -> pd.Series:
    values = pd.Series(prediction, index=frame.index)
    return values.groupby(frame["date"]).rank(pct=True, method="average").sub(0.5).fillna(0)


def _nested_validation(
    dataset: pd.DataFrame, test_year: int, settings: V10Settings
) -> tuple[dict[int, dict[str, float]], float, bool]:
    earliest = test_year - settings.training_window_years
    diagnostics: dict[int, dict[str, float]] = {}
    for year in range(test_year - settings.validation_years, test_year):
        train_5 = mature_training(dataset, year, "v9_target_5", "label_end_date_5", earliest)
        train_20 = mature_training(dataset, year, "v10_target_20", "label_end_date_20", earliest)
        validation = dataset[
            dataset["eligible"].fillna(False)
            & (pd.to_datetime(dataset["date"]).dt.year == year)
        ].copy()
        if train_5.empty or train_20.empty or validation.empty:
            diagnostics[year] = {"global_ic": float("nan"), "technology_ic": float("nan")}
            continue
        global_5 = RidgeRanker(settings.ridge_alpha).fit(
            train_5[V10_FEATURES], train_5["v9_target_5"]
        )
        global_20 = RidgeRanker(settings.ridge_alpha).fit(
            train_20[V10_FEATURES], train_20["v10_target_20"]
        )
        prediction_5 = _rank_by_date(validation, global_5.predict(validation[V10_FEATURES]))
        prediction_20 = _rank_by_date(validation, global_20.predict(validation[V10_FEATURES]))
        global_prediction = (
            settings.horizon_5_share * prediction_5
            + settings.horizon_20_share * prediction_20
        )
        global_ic = settings.horizon_5_share * _daily_ic(
            validation, global_prediction, "label_5"
        ) + settings.horizon_20_share * _daily_ic(
            validation, global_prediction, "v10_target_20"
        )

        tech_train_5 = train_5[train_5["broad_sector"] == "technology"]
        tech_train_20 = train_20[train_20["broad_sector"] == "technology"]
        tech_validation = validation[validation["broad_sector"] == "technology"]
        enough = (
            len(tech_train_5) >= settings.minimum_technology_rows
            and len(tech_train_20) >= settings.minimum_technology_rows
            and tech_validation["date"].nunique() >= 50
        )
        technology_ic = float("nan")
        if enough:
            technology_5 = RidgeRanker(settings.ridge_alpha).fit(
                tech_train_5[V10_FEATURES], tech_train_5["v9_target_5"]
            )
            technology_20 = RidgeRanker(settings.ridge_alpha).fit(
                tech_train_20[V10_FEATURES], tech_train_20["v10_target_20"]
            )
            tech_5 = _rank_by_date(
                tech_validation, technology_5.predict(tech_validation[V10_FEATURES])
            )
            tech_20 = _rank_by_date(
                tech_validation, technology_20.predict(tech_validation[V10_FEATURES])
            )
            tech_prediction = settings.horizon_5_share * tech_5 + settings.horizon_20_share * tech_20
            technology_ic = settings.horizon_5_share * _daily_ic(
                tech_validation, tech_prediction, "label_5"
            ) + settings.horizon_20_share * _daily_ic(
                tech_validation, tech_prediction, "v10_target_20"
            )
        diagnostics[year] = {
            "global_ic": float(global_ic),
            "technology_ic": float(technology_ic),
        }
    global_values = [
        value["global_ic"] for value in diagnostics.values() if np.isfinite(value["global_ic"])
    ]
    validation_ic = float(np.mean(global_values)) if global_values else 0.0
    technology_enabled = len(diagnostics) == settings.validation_years and all(
        np.isfinite(value["technology_ic"]) and value["technology_ic"] > 0
        for value in diagnostics.values()
    )
    return diagnostics, validation_ic, technology_enabled


def fit_v10_models(
    dataset: pd.DataFrame, test_year: int, settings: V10Settings | None = None
) -> V10Models:
    settings = settings or V10Settings()
    earliest = test_year - settings.training_window_years
    train_5 = mature_training(
        dataset, test_year, "v9_target_5", "label_end_date_5", earliest
    )
    train_20 = mature_training(
        dataset, test_year, "v10_target_20", "label_end_date_20", earliest
    )
    if train_5.empty or train_20.empty:
        raise RuntimeError(f"{test_year}没有足够的V10成熟训练数据")
    diagnostics, validation_ic, technology_enabled = _nested_validation(
        dataset, test_year, settings
    )
    ridge_5 = RidgeRanker(settings.ridge_alpha).fit(train_5[V10_FEATURES], train_5["v9_target_5"])
    lightgbm_5 = LightGBMRanker().fit(train_5[V10_FEATURES], train_5["v9_target_5"])
    ridge_20 = RidgeRanker(settings.ridge_alpha).fit(
        train_20[V10_FEATURES], train_20["v10_target_20"]
    )
    lightgbm_20 = LightGBMRanker().fit(
        train_20[V10_FEATURES], train_20["v10_target_20"]
    )
    tech_5 = train_5[train_5["broad_sector"] == "technology"]
    tech_20 = train_20[train_20["broad_sector"] == "technology"]
    enough = (
        len(tech_5) >= settings.minimum_technology_rows
        and len(tech_20) >= settings.minimum_technology_rows
        and tech_5["date"].nunique() >= settings.minimum_technology_dates
        and tech_20["date"].nunique() >= settings.minimum_technology_dates
    )
    technology_enabled = technology_enabled and enough
    technology_5 = (
        RidgeRanker(settings.ridge_alpha).fit(tech_5[V10_FEATURES], tech_5["v9_target_5"])
        if technology_enabled
        else ridge_5
    )
    technology_20 = (
        RidgeRanker(settings.ridge_alpha).fit(tech_20[V10_FEATURES], tech_20["v10_target_20"])
        if technology_enabled
        else ridge_20
    )
    confidence = float(
        np.clip(validation_ic / settings.validation_ic_full_confidence, 0.0, 1.0)
    )
    return V10Models(
        ridge_5=ridge_5,
        lightgbm_5=lightgbm_5,
        ridge_20=ridge_20,
        lightgbm_20=lightgbm_20,
        technology_5=technology_5,
        technology_20=technology_20,
        technology_enabled=technology_enabled,
        validation_year_ics=diagnostics,
        validation_ic=validation_ic,
        confidence=confidence,
        training_rows_5=len(train_5),
        training_rows_20=len(train_20),
        training_end=max(
            pd.to_datetime(train_5["label_end_date_5"]).max(),
            pd.to_datetime(train_20["label_end_date_20"]).max(),
        ),
    )


def _centered_rank(values: pd.Series) -> pd.Series:
    return values.rank(pct=True, method="average").sub(0.5).fillna(0)


def score_v10(
    current: pd.DataFrame,
    models: V10Models,
    v5_models: V5Models,
    v4_specs: list[FactorSpec],
    settings: V10Settings | None = None,
) -> pd.DataFrame:
    settings = settings or V10Settings()
    scored = score_v6(current, v5_models, v4_specs, V6Settings())
    scored["v6_score"] = scored["score"]
    predictions = {
        "ridge_5": models.ridge_5.predict(scored[V10_FEATURES]),
        "lightgbm_5": models.lightgbm_5.predict(scored[V10_FEATURES]),
        "ridge_20": models.ridge_20.predict(scored[V10_FEATURES]),
        "lightgbm_20": models.lightgbm_20.predict(scored[V10_FEATURES]),
    }
    for name, values in predictions.items():
        scored[name] = _centered_rank(pd.Series(values, index=scored.index))
    scored["ridge_multi"] = (
        settings.horizon_5_share * scored["ridge_5"]
        + settings.horizon_20_share * scored["ridge_20"]
    )
    scored["global_5"] = (
        settings.ridge_share * scored["ridge_5"]
        + settings.lightgbm_share * scored["lightgbm_5"]
    )
    scored["global_20"] = (
        settings.ridge_share * scored["ridge_20"]
        + settings.lightgbm_share * scored["lightgbm_20"]
    )
    scored["global_multi"] = (
        settings.horizon_5_share * scored["global_5"]
        + settings.horizon_20_share * scored["global_20"]
    )
    scored["enhanced_score"] = scored["global_multi"]
    tech = scored["broad_sector"] == "technology"
    if models.technology_enabled and tech.any():
        tech_5 = _centered_rank(
            pd.Series(
                models.technology_5.predict(scored.loc[tech, V10_FEATURES]),
                index=scored.index[tech],
            )
        )
        tech_20 = _centered_rank(
            pd.Series(
                models.technology_20.predict(scored.loc[tech, V10_FEATURES]),
                index=scored.index[tech],
            )
        )
        specialist = settings.horizon_5_share * tech_5 + settings.horizon_20_share * tech_20
        scored.loc[tech, "enhanced_score"] = (
            settings.technology_global_share * scored.loc[tech, "global_multi"]
            + settings.technology_specialist_share * specialist
        )
    scored["ridge_model_score"] = (
        settings.v6_base_share * scored["v6_score"]
        + settings.enhanced_share * scored["ridge_multi"]
    )
    scored["global_model_score"] = (
        settings.v6_base_share * scored["v6_score"]
        + settings.enhanced_share * scored["global_multi"]
    )
    scored["model_score"] = (
        settings.v6_base_share * scored["v6_score"]
        + settings.enhanced_share * scored["enhanced_score"]
    )
    scored["score"] = scored["model_score"]
    scored["model_confidence"] = models.confidence
    scored["technology_enabled"] = models.technology_enabled
    return scored

