from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research_v10.features import V10_FEATURES
from research_v10.model import mature_training
from research_v26.model import BinaryLightGBM
from stockpilot.model import LightGBMRanker

from .config import V28Settings


TARGETS = {"5": ("v9_target_5", "label_end_date_5"), "20": ("v10_target_20", "label_end_date_20")}


def tail_labels(frame: pd.DataFrame, target: str, quantile: float) -> np.ndarray:
    if not 0.5 < quantile < 1.0:
        raise ValueError("tail quantile must be between 0.5 and 1")
    values = pd.to_numeric(frame[target], errors="coerce")
    if values.isna().any():
        raise ValueError("tail labels require finite targets")
    percentile = values.groupby(pd.to_datetime(frame["date"])).rank(pct=True, method="first")
    labels = (percentile > quantile).astype(np.int8).to_numpy()
    if np.unique(labels).size != 2:
        raise ValueError("tail labels require both classes")
    return labels


@dataclass
class Heads:
    regression: dict[str, LightGBMRanker]
    direction: dict[str, BinaryLightGBM]
    tail: dict[str, BinaryLightGBM]
    rows: dict[str, int]


def fit_heads(dataset: pd.DataFrame, cutoff_year: int, settings: V28Settings) -> Heads:
    regression, direction, tail, rows = {}, {}, {}, {}
    earliest = cutoff_year - settings.training_window_years
    for horizon, (target, label_end) in TARGETS.items():
        train = mature_training(dataset, cutoff_year, target, label_end, earliest)
        if len(train) < len(V10_FEATURES) * 3:
            raise RuntimeError(f"insufficient mature rows: cutoff={cutoff_year}, horizon={horizon}")
        if pd.to_datetime(train[label_end]).max() >= pd.Timestamp(cutoff_year, 1, 1):
            raise AssertionError("immature label entered V28")
        regression[horizon] = LightGBMRanker().fit(train[V10_FEATURES], train[target])
        direction[horizon] = BinaryLightGBM().fit(train[V10_FEATURES], train[target])
        labels = tail_labels(train, target, settings.tail_quantile)
        tail[horizon] = BinaryLightGBM().fit(train[V10_FEATURES], labels)
        rows[horizon] = len(train)
    return Heads(regression, direction, tail, rows)


def centered_rank_by_date(frame: pd.DataFrame, values) -> pd.Series:
    series = pd.Series(np.asarray(values, dtype=float), index=frame.index)
    return series.groupby(pd.to_datetime(frame["date"])).rank(pct=True, method="average").sub(0.5).fillna(0.0)


def prediction_components(frame: pd.DataFrame, heads: Heads, settings: V28Settings):
    regression, blend = {}, {}
    for horizon in TARGETS:
        regression[horizon] = centered_rank_by_date(frame, heads.regression[horizon].predict(frame[V10_FEATURES]))
        direction = centered_rank_by_date(frame, heads.direction[horizon].predict(frame[V10_FEATURES]))
        tail = centered_rank_by_date(frame, heads.tail[horizon].predict(frame[V10_FEATURES]))
        blend[horizon] = settings.direction_share * direction + settings.tail_share * tail
    regression_multi = settings.horizon_5_share * regression["5"] + settings.horizon_20_share * regression["20"]
    blend_multi = settings.horizon_5_share * blend["5"] + settings.horizon_20_share * blend["20"]
    return regression_multi, blend_multi


def _daily_metrics(frame: pd.DataFrame, score: pd.Series) -> dict:
    work = frame.copy()
    work["_score"] = score
    values = {"ic5": [], "ic20": [], "spread5": [], "spread20": []}
    for _, group in work.groupby("date"):
        valid = group[group["eligible"].fillna(False)]
        for key, target in (("ic5", "v9_target_5"), ("ic20", "v10_target_20")):
            sample = valid[["_score", target]].dropna()
            if len(sample) >= 20 and sample._score.nunique() > 1:
                values[key].append(float(sample._score.corr(sample[target], method="spearman")))
        for key, target in (("spread5", "v9_target_5"), ("spread20", "v10_target_20")):
            spreads = []
            for _, sector in valid.dropna(subset=[target]).groupby("broad_sector"):
                if len(sector) < 10:
                    continue
                cutoff = sector._score.quantile(0.8)
                top, rest = sector[sector._score >= cutoff], sector[sector._score < cutoff]
                if len(top) and len(rest):
                    spreads.append(float(top[target].mean() - rest[target].mean()))
            if spreads:
                values[key].append(float(np.mean(spreads)))
    return {key: float(np.mean(series)) if series else float("nan") for key, series in values.items()}


def confidence_from_validation(metrics: list[dict]) -> float:
    if len(metrics) != 2:
        raise ValueError("exactly two prior validation years are required")
    keys = ("ic5", "ic20", "spread5", "spread20")
    finite = all(np.isfinite(item[key]) for item in metrics for key in keys)
    if not finite:
        return 0.0
    positive_years = sum(all(item[key] > 0 for key in keys) for item in metrics)
    aggregate = {key: float(np.mean([item[key] for item in metrics])) for key in keys}
    all_aggregate_positive = all(value > 0 for value in aggregate.values())
    any_aggregate_positive = any(value > 0 for value in aggregate.values())
    if positive_years == 2 and all_aggregate_positive:
        return 1.0
    if positive_years >= 1 and all_aggregate_positive:
        return 0.5
    if any_aggregate_positive:
        return 0.25
    return 0.0


def validation_slice(dataset: pd.DataFrame, year: int, every: int) -> pd.DataFrame:
    frame = dataset[dataset["eligible"].fillna(False) & pd.to_datetime(dataset["date"]).dt.year.eq(year)].copy()
    dates = pd.DatetimeIndex(frame["date"].drop_duplicates().sort_values())
    selected = set(dates[::every])
    return frame[frame["date"].isin(selected)].copy()


def build_candidate_scores(parent_scores, dataset, settings: V28Settings, progress=None):
    progress = progress or (lambda *args, **kwargs: None)
    features = dataset[["date", "symbol", "eligible", "broad_sector", "v9_target_5", "v10_target_20", *V10_FEATURES]].copy()
    features["date"] = pd.to_datetime(features["date"])
    if features.duplicated(["date", "symbol"]).any():
        raise ValueError("duplicate PIT feature keys")
    eval_features = features[["date", "symbol", *V10_FEATURES]]
    merged = parent_scores.merge(eval_features, on=["date", "symbol"], how="left", validate="one_to_one")
    if len(merged) != len(parent_scores) or merged[V10_FEATURES].isna().any().any():
        raise ValueError("frozen evaluation trace does not map to PIT features")
    needed_cutoffs = sorted(set(settings.test_years) | {year - 1 for year in settings.test_years} | {year - 2 for year in settings.test_years})
    cache = {}
    for cutoff in needed_cutoffs:
        progress("training_crossfit_heads", cutoff_year=int(cutoff))
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
        current["v28_score"] = current["global_model_score"] + changed_share * (blend - regression)
        current["model_confidence"] = confidence
        pieces.append(current)
        diagnostics[str(year)] = {"confidence": confidence, "validation": validation_metrics,
                                  "training_rows": cache[year].rows, "evaluation_rows": len(current)}
    result = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    if len(result) != len(parent_scores) or result.duplicated(["date", "symbol"]).any():
        raise AssertionError("V28 changed frozen evaluation keys")
    return result, diagnostics
