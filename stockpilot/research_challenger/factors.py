from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from .config import ChallengerSettings


def daily_rank_ic_matrix(
    frame: pd.DataFrame, features: list[str] | tuple[str, ...], target: str
) -> pd.DataFrame:
    columns = [*features, target]
    work = frame[["date", *columns]].dropna(subset=[target]).copy()
    ranks = work.groupby("date", sort=False)[columns].rank(pct=True, method="average")
    group = work["date"]
    centered = ranks - ranks.groupby(group, sort=False).transform("mean")
    target_centered = centered[target]
    target_ss = target_centered.pow(2).groupby(group, sort=False).sum()
    rows = {}
    for feature in features:
        numerator = (centered[feature] * target_centered).groupby(group, sort=False).sum()
        feature_ss = centered[feature].pow(2).groupby(group, sort=False).sum()
        values = numerator / np.sqrt(feature_ss * target_ss).replace(0, np.nan)
        rows[feature] = values
    return pd.DataFrame(rows).sort_index()


def _hac_t_stat(values: pd.Series, max_lag: int = 5) -> tuple[float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 20:
        return 0.0, 1.0
    mean = float(x.mean())
    demeaned = x - mean
    gamma0 = float(np.dot(demeaned, demeaned) / len(x))
    variance = gamma0
    for lag in range(1, min(max_lag, len(x) - 1) + 1):
        covariance = float(np.dot(demeaned[lag:], demeaned[:-lag]) / len(x))
        variance += 2 * (1 - lag / (max_lag + 1)) * covariance
    standard_error = float(np.sqrt(max(variance, 0.0) / len(x)))
    if standard_error == 0 and abs(mean) > 0:
        return float(np.sign(mean) * np.inf), 0.0
    t_stat = mean / standard_error if standard_error > 0 else 0.0
    p_value = 2 * (1 - NormalDist().cdf(abs(t_stat)))
    return float(t_stat), float(p_value)


def bh_fdr(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce").fillna(1.0).clip(0, 1)
    order = np.argsort(values.to_numpy())
    ranked = values.to_numpy()[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    result = np.empty(len(values), dtype=float)
    result[order] = adjusted
    return pd.Series(result, index=values.index)


def _turnover_proxy(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.Series:
    ordered = frame.sort_values(["symbol", "date"])
    return ordered.groupby("symbol", sort=False)[list(features)].diff().abs().mean()


@dataclass(frozen=True)
class FactorSelection:
    selected: tuple[str, ...]
    audit: pd.DataFrame
    correlation: pd.DataFrame


def select_factors_train_only(
    train: pd.DataFrame, settings: ChallengerSettings
) -> FactorSelection:
    target = f"return_rank_{settings.selection_horizon}d"
    daily = daily_rank_ic_matrix(train, settings.factor_columns, target)
    turnover = _turnover_proxy(train, settings.factor_columns)
    rows = []
    for feature in settings.factor_columns:
        values = daily[feature].dropna()
        t_stat, p_value = _hac_t_stat(values)
        annual = values.groupby(pd.to_datetime(values.index).year).mean()
        direction = np.sign(values.mean()) if len(values) else 0
        consistency = float((np.sign(annual) == direction).mean()) if direction else 0.0
        rows.append(
            {
                "factor_name": feature,
                "ic_dates": int(len(values)),
                "mean_rank_ic": float(values.mean()) if len(values) else 0.0,
                "rank_ic_std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                "rank_ic_ir": float(values.mean() / values.std(ddof=1))
                if len(values) > 1 and values.std(ddof=1) > 0
                else 0.0,
                "positive_ratio": float((values > 0).mean()) if len(values) else 0.0,
                "direction_consistency": consistency,
                "hac_t_stat": t_stat,
                "p_value": p_value,
                "turnover_proxy": float(turnover.get(feature, np.nan)),
            }
        )
    audit = pd.DataFrame(rows)
    audit["fdr_q_value"] = bh_fdr(audit["p_value"])
    audit["passes_statistical_gate"] = (
        audit["ic_dates"].ge(settings.minimum_ic_dates)
        & audit["mean_rank_ic"].abs().ge(settings.minimum_abs_rank_ic)
        & audit["positive_ratio"].where(audit["mean_rank_ic"].ge(0), 1 - audit["positive_ratio"]).ge(
            settings.minimum_positive_ratio
        )
        & audit["direction_consistency"].ge(settings.minimum_year_direction_consistency)
        & audit["fdr_q_value"].le(settings.fdr_q)
    )
    candidates = audit[audit["passes_statistical_gate"]].copy()
    candidates = candidates.sort_values(
        ["fdr_q_value", "direction_consistency", "turnover_proxy", "mean_rank_ic"],
        ascending=[True, False, True, False],
    )
    correlation = train[list(settings.factor_columns)].corr(method="spearman")
    selected: list[str] = []
    redundant_with: dict[str, str] = {}
    for feature in candidates["factor_name"]:
        conflict = next(
            (
                existing
                for existing in selected
                if abs(float(correlation.loc[feature, existing])) > settings.correlation_threshold
            ),
            None,
        )
        if conflict is not None:
            redundant_with[feature] = conflict
            continue
        selected.append(feature)
        if len(selected) >= settings.maximum_selected_factors:
            break
    audit["redundant_with"] = audit["factor_name"].map(redundant_with).fillna("")
    audit["selected"] = audit["factor_name"].isin(selected)
    audit["rejection_reason"] = np.select(
        [
            audit["selected"],
            audit["redundant_with"].ne(""),
            ~audit["passes_statistical_gate"],
        ],
        ["SELECTED", "REDUNDANT", "STATISTICAL_GATE"],
        default="MAX_FACTOR_CAP",
    )
    if not selected:
        raise RuntimeError("NO_FACTORS_PASSED_PRE_REGISTERED_TRAIN_ONLY_GATES")
    return FactorSelection(tuple(selected), audit, correlation)


def residualize_cross_section(
    frame: pd.DataFrame, values: pd.Series, mode: str
) -> pd.Series:
    if mode not in {"industry", "size", "industry_size"}:
        raise ValueError("unknown neutralization mode")
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for indexes in frame.groupby("date", sort=False).groups.values():
        group = frame.loc[indexes]
        y = pd.to_numeric(values.loc[indexes], errors="coerce")
        valid = y.notna()
        if valid.sum() < 20:
            continue
        idx = group.index[valid]
        pieces = [np.ones((len(idx), 1))]
        if mode in {"size", "industry_size"}:
            size = pd.to_numeric(group.loc[idx, "benchmark_weight_rank"], errors="coerce").fillna(0)
            pieces.append(size.to_numpy(dtype=float).reshape(-1, 1))
        if mode in {"industry", "industry_size"}:
            dummies = pd.get_dummies(
                group.loc[idx, "industry"].fillna("UNKNOWN").astype(str),
                drop_first=True,
                dtype=float,
            )
            pieces.append(dummies.to_numpy(dtype=float))
        design = np.column_stack(pieces)
        target = y.loc[idx].to_numpy(dtype=float)
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        result.loc[idx] = target - design @ coefficients
    return result
