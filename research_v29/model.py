from __future__ import annotations

import numpy as np
import pandas as pd

from research_v10.features import V10_FEATURES
from research_v10.model import mature_training
from research_v26.model import BinaryLightGBM
from research_v28.model import Heads, TARGETS, _daily_metrics, centered_rank_by_date, confidence_from_validation, validation_slice
from stockpilot.model import LightGBMRanker

from .config import V29Settings


def sector_tail_labels(frame: pd.DataFrame, target: str, quantile: float) -> np.ndarray:
    if not 0.5 < quantile < 1.0:
        raise ValueError("tail quantile must be between 0.5 and 1")
    values = pd.to_numeric(frame[target], errors="coerce")
    sectors = frame["broad_sector"]
    if values.isna().any() or sectors.isna().any() or sectors.astype(str).str.strip().eq("").any():
        raise ValueError("sector tail labels require finite targets and point-in-time sectors")
    dates = pd.to_datetime(frame["date"])
    percentile = values.groupby([dates, sectors.astype(str)], sort=False).rank(pct=True, method="first")
    labels = (percentile > quantile).astype(np.int8).to_numpy()
    if np.unique(labels).size != 2:
        raise ValueError("sector tail labels require both classes")
    return labels


def fit_heads(dataset: pd.DataFrame, cutoff_year: int, settings: V29Settings) -> Heads:
    regression, direction, tail, rows = {}, {}, {}, {}
    earliest = cutoff_year - settings.training_window_years
    for horizon, (target, label_end) in TARGETS.items():
        train = mature_training(dataset, cutoff_year, target, label_end, earliest)
        if len(train) < len(V10_FEATURES) * 3:
            raise RuntimeError(f"insufficient mature rows: cutoff={cutoff_year}, horizon={horizon}")
        if pd.to_datetime(train[label_end]).max() >= pd.Timestamp(cutoff_year, 1, 1):
            raise AssertionError("immature label entered V29")
        regression[horizon] = LightGBMRanker().fit(train[V10_FEATURES], train[target])
        direction[horizon] = BinaryLightGBM().fit(train[V10_FEATURES], train[target])
        labels = sector_tail_labels(train, target, settings.tail_quantile)
        tail[horizon] = BinaryLightGBM().fit(train[V10_FEATURES], labels)
        rows[horizon] = len(train)
    return Heads(regression, direction, tail, rows)


def prediction_components(frame: pd.DataFrame, heads: Heads, settings: V29Settings):
    regression, blend = {}, {}
    for horizon in TARGETS:
        regression[horizon] = centered_rank_by_date(frame, heads.regression[horizon].predict(frame[V10_FEATURES]))
        direction = centered_rank_by_date(frame, heads.direction[horizon].predict(frame[V10_FEATURES]))
        tail = centered_rank_by_date(frame, heads.tail[horizon].predict(frame[V10_FEATURES]))
        blend[horizon] = settings.direction_share * direction + settings.tail_share * tail
    regression_multi = settings.horizon_5_share * regression["5"] + settings.horizon_20_share * regression["20"]
    blend_multi = settings.horizon_5_share * blend["5"] + settings.horizon_20_share * blend["20"]
    return regression_multi, blend_multi


def build_candidate_scores(parent_scores, dataset, settings: V29Settings, progress=None):
    progress = progress or (lambda *args, **kwargs: None)
    columns = ["date", "symbol", "eligible", "broad_sector", "v9_target_5", "v10_target_20", *V10_FEATURES]
    features = dataset[columns].copy()
    features["date"] = pd.to_datetime(features["date"])
    if features.duplicated(["date", "symbol"]).any():
        raise ValueError("duplicate PIT feature keys")
    eval_features = features[["date", "symbol", *V10_FEATURES]]
    merged = parent_scores.merge(eval_features, on=["date", "symbol"], how="left", validate="one_to_one")
    if len(merged) != len(parent_scores) or merged[V10_FEATURES].isna().any().any():
        raise ValueError("frozen evaluation trace does not map to PIT features")
    cutoffs = sorted(set(settings.test_years) | {year - 1 for year in settings.test_years} | {year - 2 for year in settings.test_years})
    cache = {}
    for cutoff in cutoffs:
        progress("training_sector_conditional_heads", cutoff_year=int(cutoff))
        cache[cutoff] = fit_heads(dataset, cutoff, settings)
    pieces, diagnostics = [], {}
    for year in settings.test_years:
        validation_metrics = []
        for validation_year in (year - 2, year - 1):
            validation = validation_slice(dataset, validation_year, settings.validation_rebalance_every)
            _, blend = prediction_components(validation, cache[validation_year], settings)
            validation_metrics.append({"year": validation_year, **_daily_metrics(validation, blend)})
        confidence = confidence_from_validation(validation_metrics)
        current = merged[pd.to_datetime(merged["date"]).dt.year.eq(year)].copy()
        regression, blend = prediction_components(current, cache[year], settings)
        changed_share = settings.enhanced_share * settings.lightgbm_share
        current["v29_score"] = current["global_model_score"] + changed_share * (blend - regression)
        current["model_confidence"] = confidence
        pieces.append(current)
        diagnostics[str(year)] = {"confidence": confidence, "validation": validation_metrics,
                                  "training_rows": cache[year].rows, "evaluation_rows": len(current)}
    result = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    if len(result) != len(parent_scores) or result.duplicated(["date", "symbol"]).any():
        raise AssertionError("V29 changed frozen evaluation keys")
    return result, diagnostics
