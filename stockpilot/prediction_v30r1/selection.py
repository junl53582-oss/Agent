from __future__ import annotations

import numpy as np
import pandas as pd

from stockpilot.prediction.metrics import binary_metrics, rank_ic_by_date


def select_direction_champion(history: pd.DataFrame) -> tuple[str, dict]:
    lgb = binary_metrics(history["actual"], history["raw_probability"])
    logistic = binary_metrics(history["actual"], history["logistic_probability"])
    lgb_wins = bool(
        lgb["brier"] < logistic["brier"]
        and lgb["log_loss"] < logistic["log_loss"]
        and lgb["roc_auc"] >= logistic["roc_auc"]
    )
    return ("lightgbm" if lgb_wins else "logistic_ridge"), {
        "lightgbm": lgb, "logistic_ridge": logistic, "lightgbm_retained": lgb_wins,
    }


def _return_metrics(frame: pd.DataFrame, column: str) -> dict:
    actual = pd.to_numeric(frame["actual_return"], errors="coerce").to_numpy(dtype=float)
    predicted = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(actual) & np.isfinite(predicted)
    error = predicted[keep] - actual[keep]
    return {
        "sample_size": int(keep.sum()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "rank_ic": rank_ic_by_date(frame, column, "actual_return"),
    }


def select_return_champion(history: pd.DataFrame) -> tuple[str, dict]:
    lgb = _return_metrics(history, "expected_return")
    ridge = _return_metrics(history, "ridge_expected_return")
    lgb_wins = bool(
        lgb["mae"] < ridge["mae"]
        and lgb["rmse"] < ridge["rmse"]
        and lgb["rank_ic"] >= ridge["rank_ic"]
    )
    return ("lightgbm" if lgb_wins else "ridge"), {
        "lightgbm": lgb, "ridge": ridge, "lightgbm_retained": lgb_wins,
    }
