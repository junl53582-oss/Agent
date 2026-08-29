from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from research_v10.features import V10_FEATURES
from stockpilot.model import LightGBMRanker

from .config import V25R1Settings


TARGETS = {
    "5": ("v9_target_5", "label_end_date_5"),
    "20": ("v10_target_20", "label_end_date_20"),
}


def mature_window(dataset: pd.DataFrame, test_year: int, window_years: int, target: str, label_end: str) -> pd.DataFrame:
    cutoff = pd.Timestamp(test_year, 1, 1)
    dates = pd.to_datetime(dataset["date"])
    ends = pd.to_datetime(dataset[label_end])
    result = dataset[
        dataset["eligible"].fillna(False)
        & dataset[target].notna()
        & (dates >= pd.Timestamp(test_year - window_years, 1, 1))
        & (dates < cutoff)
        & (ends < cutoff)
    ].sort_values(["date", "symbol"])
    if not result.empty and pd.to_datetime(result[label_end]).max() >= cutoff:
        raise AssertionError("immature label entered training")
    return result


def centered_rank_by_date(frame: pd.DataFrame, values: np.ndarray) -> pd.Series:
    ranked = pd.Series(np.asarray(values, dtype=float), index=frame.index)
    return ranked.groupby(pd.to_datetime(frame["date"])).rank(pct=True, method="average").sub(0.5).fillna(0.0)


@dataclass
class TemporalModels:
    by_horizon: dict[str, list[tuple[int, LightGBMRanker]]]
    training_rows: dict[str, dict[int, int]]


def fit_temporal_models(
    dataset: pd.DataFrame,
    test_year: int,
    settings: V25R1Settings,
    model_factory: Callable[[], LightGBMRanker] = LightGBMRanker,
) -> TemporalModels:
    models: dict[str, list[tuple[int, LightGBMRanker]]] = {}
    rows: dict[str, dict[int, int]] = {}
    for horizon, (target, label_end) in TARGETS.items():
        models[horizon] = []
        rows[horizon] = {}
        for window in settings.training_windows_years:
            train = mature_window(dataset, test_year, window, target, label_end)
            if len(train) < len(V10_FEATURES) * 3:
                raise RuntimeError(f"insufficient mature training rows: year={test_year}, horizon={horizon}, window={window}")
            model = model_factory().fit(train[V10_FEATURES], train[target])
            models[horizon].append((window, model))
            rows[horizon][window] = len(train)
    return TemporalModels(models, rows)


def score_temporal_delta(current: pd.DataFrame, models: TemporalModels, settings: V25R1Settings) -> pd.DataFrame:
    required = {"date", "symbol", "global_model_score", *V10_FEATURES}
    missing = required - set(current.columns)
    if missing:
        raise ValueError(f"candidate input missing columns: {sorted(missing)}")
    scored = current.copy()
    ensemble = {}
    original = {}
    for horizon in TARGETS:
        ranks = []
        for window, model in models.by_horizon[horizon]:
            rank = centered_rank_by_date(scored, model.predict(scored[V10_FEATURES]))
            ranks.append(rank)
            if window == max(settings.training_windows_years):
                original[horizon] = rank
        ensemble[horizon] = pd.concat(ranks, axis=1).mean(axis=1)
    if set(original) != set(TARGETS):
        raise AssertionError("full-window anchor is missing")
    horizon_delta = (
        settings.horizon_5_share * (ensemble["5"] - original["5"])
        + settings.horizon_20_share * (ensemble["20"] - original["20"])
    )
    lightgbm_delta_share = settings.enhanced_share * settings.lightgbm_share
    scored["temporal_ensemble_score"] = scored["global_model_score"] + lightgbm_delta_share * horizon_delta
    scored["temporal_delta"] = lightgbm_delta_share * horizon_delta
    if not np.isfinite(scored["temporal_ensemble_score"]).all():
        raise ValueError("non-finite temporal ensemble score")
    return scored


def build_candidate_scores(
    parent_scores: pd.DataFrame,
    dataset: pd.DataFrame,
    settings: V25R1Settings,
    progress: Callable[..., None] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    progress = progress or (lambda *args, **kwargs: None)
    feature_rows = dataset[["date", "symbol", *V10_FEATURES]].copy()
    feature_rows["date"] = pd.to_datetime(feature_rows["date"])
    if feature_rows.duplicated(["date", "symbol"]).any():
        raise ValueError("duplicate dataset feature keys")
    merged = parent_scores.merge(feature_rows, on=["date", "symbol"], how="left", validate="one_to_one")
    if len(merged) != len(parent_scores) or merged[V10_FEATURES].isna().any().any():
        raise ValueError("frozen score trace does not map completely to PIT features")
    pieces, diagnostics = [], {}
    for year in settings.test_years:
        progress("training_temporal_ensemble", test_year=int(year))
        models = fit_temporal_models(dataset, year, settings)
        current = merged[pd.to_datetime(merged["date"]).dt.year.eq(year)].copy()
        if current.empty:
            raise ValueError(f"missing evaluation scores for {year}")
        pieces.append(score_temporal_delta(current, models, settings))
        diagnostics[str(year)] = {"training_rows": models.training_rows, "evaluation_rows": len(current)}
    result = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    if result.duplicated(["date", "symbol"]).any() or len(result) != len(parent_scores):
        raise AssertionError("candidate score generation changed frozen evaluation keys")
    return result, diagnostics

