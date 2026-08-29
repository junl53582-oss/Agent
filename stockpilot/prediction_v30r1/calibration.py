from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stockpilot.prediction.calibration import PlattCalibrator


@dataclass
class MonotonicPlattCalibrator:
    slope: float = 1.0
    intercept: float = 0.0
    fitted: bool = False
    fallback_to_prevalence: bool = False

    def fit(
        self,
        raw_probability: np.ndarray,
        target: np.ndarray,
        *,
        calibration_ids: set[str] | None = None,
        model_training_ids: set[str] | None = None,
    ) -> "MonotonicPlattCalibrator":
        fitted = PlattCalibrator().fit(
            raw_probability, target,
            calibration_ids=calibration_ids, model_training_ids=model_training_ids,
        )
        if fitted.slope <= 0:
            y = np.asarray(target, dtype=float)
            prevalence = float(np.clip(np.nanmean(y), 1e-6, 1 - 1e-6))
            self.slope = 0.0
            self.intercept = float(np.log(prevalence / (1 - prevalence)))
            self.fallback_to_prevalence = True
        else:
            self.slope, self.intercept = fitted.slope, fitted.intercept
        self.fitted = True
        return self

    def predict(self, raw_probability: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("calibrator is not fitted")
        clipped = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1 - 1e-6)
        logit = np.log(clipped / (1 - clipped))
        value = np.clip(self.slope * logit + self.intercept, -35, 35)
        return np.clip(1 / (1 + np.exp(-value)), 1e-7, 1 - 1e-7)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(vars(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "MonotonicPlattCalibrator":
        return cls(**json.loads(path.read_text(encoding="utf-8")))
