from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


def residualize_cross_section(
    frame: pd.DataFrame,
    value: str,
    *,
    industry: str = "industry",
    size: str = "log_size",
) -> pd.Series:
    required = {value, industry, size}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"neutralization input missing {sorted(missing)}")
    usable = frame[[value, industry, size]].copy()
    valid = usable[value].notna() & usable[size].notna() & usable[industry].notna()
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    if valid.sum() < 10:
        return result
    design = pd.get_dummies(usable.loc[valid, industry].astype(str), drop_first=True, dtype=float)
    design.insert(0, "log_size", pd.to_numeric(usable.loc[valid, size], errors="coerce"))
    design.insert(0, "intercept", 1.0)
    y = pd.to_numeric(usable.loc[valid, value], errors="coerce").to_numpy(dtype=float)
    x = design.to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    result.loc[valid] = y - x @ coefficients
    return result


def _daily_ic(frame: pd.DataFrame, factor: str, target: str, method: str) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for date, group in frame.groupby("date", sort=True):
        pair = group[[factor, target]].dropna()
        if len(pair) >= 10 and pair[factor].nunique() > 1 and pair[target].nunique() > 1:
            values[pd.Timestamp(date)] = float(pair[factor].corr(pair[target], method=method))
    return pd.Series(values, dtype=float)


def factor_validation_metrics(frame: pd.DataFrame, factor: str, target: str) -> dict:
    required = {"date", "symbol", factor, target}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"factor validation input missing {sorted(missing)}")
    pearson = _daily_ic(frame, factor, target, "pearson")
    rank = _daily_ic(frame, factor, target, "spearman")
    std = float(rank.std(ddof=1)) if len(rank) > 1 else math.nan
    mean = float(rank.mean()) if len(rank) else math.nan
    return {
        "pearson_ic": float(pearson.mean()) if len(pearson) else math.nan,
        "spearman_rank_ic": mean,
        "ic_mean": mean,
        "ic_std": std,
        "icir": mean / std if np.isfinite(std) and std > 0 else math.nan,
        "positive_ic_ratio": float(rank.gt(0).mean()) if len(rank) else math.nan,
        "t_stat": mean / (std / math.sqrt(len(rank))) if np.isfinite(std) and std > 0 else math.nan,
        "coverage": float(frame[factor].notna().mean()),
        "cross_sectional_dispersion": float(frame.groupby("date")[factor].std(ddof=1).mean()),
        "dates": int(len(rank)),
    }


def neutralized_factor_metrics(
    frame: pd.DataFrame,
    factor: str,
    target: str,
    *,
    industry: str = "industry",
    size: str = "log_size",
) -> dict:
    pieces = []
    for _, group in frame.groupby("date", sort=True):
        copy = group.copy()
        copy["neutral_factor"] = residualize_cross_section(copy, factor, industry=industry, size=size)
        copy["neutral_target"] = residualize_cross_section(copy, target, industry=industry, size=size)
        pieces.append(copy)
    combined = pd.concat(pieces, ignore_index=True) if pieces else frame.head(0).copy()
    return factor_validation_metrics(combined, "neutral_factor", "neutral_target")


@dataclass(frozen=True)
class PurgedFold:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    validation_start: str
    validation_end: str

    def to_dict(self) -> dict:
        return asdict(self)


class PurgedWalkForwardSplit:
    def __init__(self, *, min_train_dates: int, validation_dates: int, gap_dates: int, step_dates: int | None = None):
        if min(min_train_dates, validation_dates, gap_dates) < 1:
            raise ValueError("walk-forward dimensions must be positive")
        self.min_train_dates = min_train_dates
        self.validation_dates = validation_dates
        self.gap_dates = gap_dates
        self.step_dates = step_dates or validation_dates

    def split(self, dates: pd.Series, label_end_dates: pd.Series):
        decision = pd.to_datetime(dates).dt.normalize()
        label_end = pd.to_datetime(label_end_dates).dt.normalize()
        unique = pd.DatetimeIndex(decision.drop_duplicates().sort_values())
        start = self.min_train_dates + self.gap_dates
        while start + self.validation_dates <= len(unique):
            validation_dates = unique[start : start + self.validation_dates]
            validation_start = validation_dates[0]
            train_dates = unique[: start - self.gap_dates]
            train_mask = decision.isin(train_dates) & label_end.lt(validation_start)
            validation_mask = decision.isin(validation_dates)
            if train_mask.any() and validation_mask.any():
                yield PurgedFold(
                    train_indices=tuple(np.flatnonzero(train_mask)),
                    validation_indices=tuple(np.flatnonzero(validation_mask)),
                    validation_start=str(validation_start.date()),
                    validation_end=str(validation_dates[-1].date()),
                )
            start += self.step_dates


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> pd.DataFrame:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    values = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite values in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted_ranked = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return pd.DataFrame({
        "hypothesis_index": np.arange(len(values)),
        "p_value": values,
        "adjusted_p_value": adjusted,
        "rejected": adjusted <= alpha,
        "family_size": len(values),
    })


def preregistration_manifest(definitions: list[dict]) -> dict:
    required = {"name", "formula", "family", "selection_criterion"}
    for item in definitions:
        missing = required - set(item)
        if missing:
            raise ValueError(f"factor preregistration missing {sorted(missing)}")
    return {
        "number_of_tested_factors": len(definitions),
        "factor_definitions": definitions,
        "selection_inside_training_window_only": True,
        "multiple_testing_diagnostic": "BENJAMINI_HOCHBERG_FDR",
        "real_prospective_results_read": False,
    }
