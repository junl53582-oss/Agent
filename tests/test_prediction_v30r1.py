from __future__ import annotations

import numpy as np
import pandas as pd

from stockpilot.prediction.freeze import verify_validation_lock
from stockpilot.prediction_v30r1.calibration import MonotonicPlattCalibrator
from stockpilot.prediction_v30r1.pipeline import _rolling_history
from stockpilot.prediction_v30r1.selection import select_direction_champion, select_return_champion


def test_parent_v30_lock_remains_intact() -> None:
    assert verify_validation_lock()["intact"]


def test_negative_platt_slope_cannot_reverse_ranking() -> None:
    raw = np.linspace(0.1, 0.9, 100)
    target = np.r_[np.ones(50), np.zeros(50)]
    calibrator = MonotonicPlattCalibrator().fit(raw, target)
    calibrated = calibrator.predict(raw)
    assert calibrator.slope == 0
    assert calibrator.fallback_to_prevalence
    assert np.all(np.diff(calibrated) >= 0)


def test_direction_champion_requires_all_three_lightgbm_wins() -> None:
    actual = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    frame = pd.DataFrame({
        "actual": actual,
        "raw_probability": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        "logistic_probability": [0.4, 0.4, 0.4, 0.6, 0.6, 0.6],
    })
    source, evidence = select_direction_champion(frame)
    assert source == "lightgbm"
    assert evidence["lightgbm_retained"]
    frame["raw_probability"] = 1 - frame["raw_probability"]
    assert select_direction_champion(frame)[0] == "logistic_ridge"


def test_return_champion_falls_back_to_ridge() -> None:
    dates = pd.bdate_range("2020-01-01", periods=20).repeat(2)
    actual = np.tile([-0.01, 0.01], 20)
    frame = pd.DataFrame({
        "date": dates, "actual_return": actual,
        "expected_return": -actual,
        "ridge_expected_return": actual * 0.9,
    })
    source, evidence = select_return_champion(frame)
    assert source == "ridge"
    assert not evidence["lightgbm_retained"]


def test_rolling_champion_history_excludes_current_and_future_years() -> None:
    frame = pd.DataFrame({"horizon": [5] * 6, "test_year": range(2018, 2024)})
    selected = _rolling_history(frame, 2023, 5, 3)
    assert selected["test_year"].tolist() == [2020, 2021, 2022]
