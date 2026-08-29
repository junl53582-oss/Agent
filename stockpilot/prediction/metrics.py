from __future__ import annotations

import math

import numpy as np
import pandas as pd


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)
    keep = np.isfinite(y) & np.isfinite(s)
    y, s = y[keep], s[keep]
    positives, negatives = int((y == 1).sum()), int((y == 0).sum())
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def pr_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)
    keep = np.isfinite(y) & np.isfinite(s)
    y, s = y[keep], s[keep]
    positives = int((y == 1).sum())
    if positives == 0:
        return math.nan
    order = np.argsort(-s, kind="mergesort")
    sorted_y = y[order]
    precision = np.cumsum(sorted_y) / np.arange(1, len(sorted_y) + 1)
    return float(precision[sorted_y == 1].sum() / positives)


def rank_ic_by_date(frame: pd.DataFrame, score: str, actual_return: str) -> float:
    values: list[float] = []
    for _, group in frame.groupby("date", sort=False):
        pair = group[[score, actual_return]].dropna()
        if len(pair) >= 10 and pair[score].nunique() > 1 and pair[actual_return].nunique() > 1:
            values.append(float(pair[score].rank().corr(pair[actual_return].rank())))
    return float(np.mean(values)) if values else math.nan


def binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    keep = np.isfinite(y) & np.isfinite(p)
    y, p = y[keep], np.clip(p[keep], 1e-7, 1 - 1e-7)
    if len(y) == 0:
        return {key: math.nan for key in (
            "roc_auc", "pr_auc", "log_loss", "brier", "accuracy", "balanced_accuracy",
            "precision", "recall", "f1", "actual_up_rate", "mean_probability",
        )} | {"sample_size": 0}
    prediction = (p >= 0.5).astype(float)
    tp = int(((prediction == 1) & (y == 1)).sum())
    tn = int(((prediction == 0) & (y == 0)).sum())
    fp = int(((prediction == 1) & (y == 0)).sum())
    fn = int(((prediction == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "sample_size": int(len(y)),
        "roc_auc": roc_auc(y, p),
        "pr_auc": pr_auc(y, p),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean(prediction == y)),
        "balanced_accuracy": float((recall + specificity) / 2),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "actual_up_rate": float(y.mean()),
        "mean_probability": float(p.mean()),
    }


def calibration_table(
    y_true: np.ndarray,
    probability: np.ndarray,
    bins: tuple[float, ...] = (0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.0),
) -> pd.DataFrame:
    data = pd.DataFrame({"actual": y_true, "probability": probability}).dropna()
    data["probability_bucket"] = pd.cut(data["probability"], bins=bins, include_lowest=True, right=False)
    result = data.groupby("probability_bucket", observed=False).agg(
        sample_size=("actual", "size"),
        mean_predicted_probability=("probability", "mean"),
        actual_up_rate=("actual", "mean"),
    ).reset_index()
    result["probability_bucket"] = result["probability_bucket"].astype(str)
    result["absolute_calibration_error"] = (
        result["mean_predicted_probability"] - result["actual_up_rate"]
    ).abs()
    return result


def expected_calibration_error(table: pd.DataFrame) -> float:
    valid = table.dropna(subset=["absolute_calibration_error"])
    total = valid["sample_size"].sum()
    if total == 0:
        return math.nan
    return float((valid["sample_size"] * valid["absolute_calibration_error"]).sum() / total)

