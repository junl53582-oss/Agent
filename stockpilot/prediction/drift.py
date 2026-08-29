from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < bins or len(current) == 0:
        return math.nan
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference, bins=edges)[0] / len(reference)
    actual = np.histogram(current, bins=edges)[0] / len(current)
    expected, actual = np.clip(expected, 1e-6, None), np.clip(actual, 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def feature_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    *,
    psi_warning: float = 0.10,
    psi_severe: float = 0.25,
    zscore_warning: float = 1.0,
    zscore_severe: float = 2.0,
) -> tuple[pd.DataFrame, str, float]:
    rows: list[dict] = []
    for feature in features:
        train = pd.to_numeric(reference.get(feature), errors="coerce").to_numpy(dtype=float)
        now = pd.to_numeric(current.get(feature), errors="coerce").to_numpy(dtype=float)
        train_mean, train_std = np.nanmean(train), np.nanstd(train)
        current_mean = np.nanmean(now)
        zscore = abs(current_mean - train_mean) / train_std if train_std > 1e-12 else 0.0
        psi = _psi(train, now)
        rows.append({
            "feature": feature,
            "training_missing_rate": float(np.mean(~np.isfinite(train))),
            "current_missing_rate": float(np.mean(~np.isfinite(now))),
            "mean_zscore_drift": float(zscore),
            "psi": float(psi),
        })
    report = pd.DataFrame(rows)
    severe = (
        (report["psi"] >= psi_severe)
        | (report["mean_zscore_drift"] >= zscore_severe)
        | ((report["current_missing_rate"] - report["training_missing_rate"]) >= 0.20)
    )
    warning = (
        (report["psi"] >= psi_warning)
        | (report["mean_zscore_drift"] >= zscore_warning)
        | ((report["current_missing_rate"] - report["training_missing_rate"]) >= 0.10)
    )
    if severe.any():
        status, multiplier = "SEVERE", 0.45
    elif warning.any():
        status, multiplier = "WARNING", 0.75
    else:
        status, multiplier = "STABLE", 1.0
    return report, status, multiplier


def build_reference_profile(reference: pd.DataFrame, features: list[str]) -> dict:
    profile: dict[str, dict] = {}
    for feature in features:
        values = pd.to_numeric(reference[feature], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            profile[feature] = {"mean": 0.0, "std": 1.0, "missing_rate": 1.0, "edges": [], "shares": []}
            continue
        edges = np.unique(np.quantile(finite, np.linspace(0, 1, 11)))
        if len(edges) >= 3:
            edges[0], edges[-1] = -np.inf, np.inf
            shares = (np.histogram(finite, bins=edges)[0] / len(finite)).tolist()
            serial_edges = [None if not np.isfinite(value) else float(value) for value in edges]
        else:
            serial_edges, shares = [], []
        profile[feature] = {
            "mean": float(np.mean(finite)), "std": float(np.std(finite)),
            "missing_rate": float(np.mean(~np.isfinite(values))),
            "edges": serial_edges, "shares": shares,
        }
    return profile


def drift_from_profile(
    profile: dict,
    current: pd.DataFrame,
    *,
    psi_warning: float = 0.10,
    psi_severe: float = 0.25,
    zscore_warning: float = 1.0,
    zscore_severe: float = 2.0,
) -> tuple[pd.DataFrame, str, float]:
    rows = []
    for feature, reference in profile.items():
        values = pd.to_numeric(current[feature], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        std = float(reference["std"])
        zscore = abs(float(np.mean(finite)) - float(reference["mean"])) / std if len(finite) and std > 1e-12 else 0.0
        edges = [-np.inf if value is None and index == 0 else np.inf if value is None else value for index, value in enumerate(reference["edges"])]
        if len(edges) >= 3 and len(finite):
            actual = np.histogram(finite, bins=np.asarray(edges, dtype=float))[0] / len(finite)
            expected = np.asarray(reference["shares"], dtype=float)
            actual, expected = np.clip(actual, 1e-6, None), np.clip(expected, 1e-6, None)
            psi = float(np.sum((actual - expected) * np.log(actual / expected)))
        else:
            psi = 0.0
        rows.append({
            "feature": feature, "training_missing_rate": reference["missing_rate"],
            "current_missing_rate": float(np.mean(~np.isfinite(values))),
            "mean_zscore_drift": zscore, "psi": psi,
        })
    report = pd.DataFrame(rows)
    severe = (report["psi"] >= psi_severe) | (report["mean_zscore_drift"] >= zscore_severe) | ((report["current_missing_rate"] - report["training_missing_rate"]) >= 0.20)
    warning = (report["psi"] >= psi_warning) | (report["mean_zscore_drift"] >= zscore_warning) | ((report["current_missing_rate"] - report["training_missing_rate"]) >= 0.10)
    if severe.any():
        return report, "SEVERE", 0.45
    if warning.any():
        return report, "WARNING", 0.75
    return report, "STABLE", 1.0
