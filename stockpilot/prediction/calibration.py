from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


@dataclass
class PlattCalibrator:
    slope: float = 1.0
    intercept: float = 0.0
    fitted: bool = False

    def fit(
        self,
        raw_probability: np.ndarray,
        target: np.ndarray,
        *,
        calibration_ids: set[str] | None = None,
        model_training_ids: set[str] | None = None,
    ) -> "PlattCalibrator":
        if calibration_ids is not None and model_training_ids is not None:
            overlap = calibration_ids.intersection(model_training_ids)
            if overlap:
                raise ValueError("calibrator observations overlap model training observations")
        x = _logit(raw_probability)
        y = np.asarray(target, dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]
        if len(y) < 20 or np.unique(y).size < 2:
            raise ValueError("calibration requires at least 20 observations and both classes")
        slope = 1.0
        prevalence = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        intercept = float(np.log(prevalence / (1 - prevalence)))
        design = np.column_stack([x, np.ones(len(x))])
        for _ in range(50):
            logits = np.clip(design @ np.array([slope, intercept]), -35, 35)
            p = 1.0 / (1.0 + np.exp(-logits))
            weight = np.clip(p * (1 - p), 1e-6, None)
            gradient = design.T @ (p - y) + np.array([1e-6 * slope, 0.0])
            hessian = (design.T * weight) @ design + np.diag([1e-6, 1e-9])
            step = np.linalg.solve(hessian, gradient)
            slope -= float(step[0])
            intercept -= float(step[1])
            if np.max(np.abs(step)) < 1e-8:
                break
        self.slope, self.intercept, self.fitted = slope, intercept, True
        return self

    def predict(self, raw_probability: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("calibrator is not fitted")
        logits = np.clip(self.slope * _logit(raw_probability) + self.intercept, -35, 35)
        return np.clip(1.0 / (1.0 + np.exp(-logits)), 1e-7, 1 - 1e-7)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(vars(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PlattCalibrator":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

