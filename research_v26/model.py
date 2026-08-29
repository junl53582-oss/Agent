from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from research_v10.features import V10_FEATURES
from research_v10.model import mature_training
from stockpilot.model import LightGBMRanker

from .config import V26Settings


TARGETS = {
    "5": ("v9_target_5", "label_end_date_5"),
    "20": ("v10_target_20", "label_end_date_20"),
}


def direction_labels(values: pd.Series | np.ndarray) -> np.ndarray:
    target = np.asarray(values, dtype=float)
    if not np.isfinite(target).all():
        raise ValueError("direction labels must be finite")
    labels = (target > 0.0).astype(np.int8)
    if np.unique(labels).size != 2:
        raise ValueError("direction labels require both classes")
    return labels


@dataclass
class BinaryLightGBM:
    model_: object | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series | np.ndarray):
        import lightgbm as lgb

        features = np.asarray(x, dtype=float)
        target = np.asarray(y, dtype=float)
        valid = np.isfinite(features).all(axis=1) & np.isfinite(target)
        if valid.sum() < features.shape[1] * 3:
            raise ValueError("insufficient finite binary training rows")
        labels = direction_labels(target[valid])
        train = lgb.Dataset(features[valid], label=labels, feature_name=list(x.columns), free_raw_data=True)
        self.model_ = lgb.train(
            {
                "objective": "binary",
                "metric": "binary_logloss",
                "learning_rate": 0.04,
                "num_leaves": 15,
                "max_depth": 5,
                "min_data_in_leaf": 200,
                "feature_fraction": 0.8,
                "lambda_l1": 1.0,
                "lambda_l2": 5.0,
                "seed": 42,
                "num_threads": 4,
                "verbosity": -1,
            },
            train,
            num_boost_round=120,
        )
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("binary model is not fitted")
        return np.asarray(self.model_.predict(x), dtype=float)


@dataclass
class DirectionalModels:
    regression: dict[str, LightGBMRanker]
    probability: dict[str, BinaryLightGBM]
    training_rows: dict[str, int]
    positive_rates: dict[str, float]


def fit_directional_models(
    dataset: pd.DataFrame,
    test_year: int,
    settings: V26Settings,
    regression_factory: Callable[[], LightGBMRanker] = LightGBMRanker,
    binary_factory: Callable[[], BinaryLightGBM] = BinaryLightGBM,
) -> DirectionalModels:
    regression, probability, rows, rates = {}, {}, {}, {}
    earliest = test_year - settings.training_window_years
    for horizon, (target, label_end) in TARGETS.items():
        train = mature_training(dataset, test_year, target, label_end, earliest)
        if len(train) < len(V10_FEATURES) * 3:
            raise RuntimeError(f"insufficient mature training rows: year={test_year}, horizon={horizon}")
        cutoff = pd.Timestamp(test_year, 1, 1)
        if pd.to_datetime(train[label_end]).max() >= cutoff:
            raise AssertionError("immature label entered directional training")
        regression[horizon] = regression_factory().fit(train[V10_FEATURES], train[target])
        probability[horizon] = binary_factory().fit(train[V10_FEATURES], train[target])
        rows[horizon] = len(train)
        rates[horizon] = float((train[target] > 0).mean())
    return DirectionalModels(regression, probability, rows, rates)


def centered_rank_by_date(frame: pd.DataFrame, values: np.ndarray) -> pd.Series:
    series = pd.Series(np.asarray(values, dtype=float), index=frame.index)
    return series.groupby(pd.to_datetime(frame["date"])).rank(pct=True, method="average").sub(0.5).fillna(0.0)


def score_directional_delta(current: pd.DataFrame, models: DirectionalModels, settings: V26Settings) -> pd.DataFrame:
    required = {"date", "symbol", "global_model_score", *V10_FEATURES}
    missing = required - set(current.columns)
    if missing:
        raise ValueError(f"candidate input missing columns: {sorted(missing)}")
    scored = current.copy()
    delta = {}
    for horizon in TARGETS:
        regression = centered_rank_by_date(scored, models.regression[horizon].predict(scored[V10_FEATURES]))
        probability = centered_rank_by_date(scored, models.probability[horizon].predict(scored[V10_FEATURES]))
        delta[horizon] = probability - regression
    objective_delta = settings.horizon_5_share * delta["5"] + settings.horizon_20_share * delta["20"]
    changed_share = settings.enhanced_share * settings.lightgbm_share
    scored["directional_probability_score"] = scored["global_model_score"] + changed_share * objective_delta
    scored["directional_delta"] = changed_share * objective_delta
    if not np.isfinite(scored["directional_probability_score"]).all():
        raise ValueError("non-finite directional probability score")
    return scored


def build_candidate_scores(parent_scores, dataset, settings: V26Settings, progress=None):
    progress = progress or (lambda *args, **kwargs: None)
    features = dataset[["date", "symbol", *V10_FEATURES]].copy()
    features["date"] = pd.to_datetime(features["date"])
    if features.duplicated(["date", "symbol"]).any():
        raise ValueError("duplicate PIT feature keys")
    merged = parent_scores.merge(features, on=["date", "symbol"], how="left", validate="one_to_one")
    if len(merged) != len(parent_scores) or merged[V10_FEATURES].isna().any().any():
        raise ValueError("frozen score trace does not map completely to PIT features")
    pieces, diagnostics = [], {}
    for year in settings.test_years:
        progress("training_directional_probability", test_year=int(year))
        models = fit_directional_models(dataset, year, settings)
        current = merged[pd.to_datetime(merged["date"]).dt.year.eq(year)].copy()
        if current.empty:
            raise ValueError(f"missing evaluation scores for {year}")
        pieces.append(score_directional_delta(current, models, settings))
        diagnostics[str(year)] = {"training_rows": models.training_rows,
                                  "positive_rates": models.positive_rates,
                                  "evaluation_rows": len(current)}
    result = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    if result.duplicated(["date", "symbol"]).any() or len(result) != len(parent_scores):
        raise AssertionError("candidate generation changed frozen evaluation keys")
    return result, diagnostics

