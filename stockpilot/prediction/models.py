from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    values = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    return values.to_numpy(dtype=float)


def deterministic_sample(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Bound training cost without introducing a random or shuffled time split."""
    if cap <= 0 or len(frame) <= cap:
        return frame
    positions = np.linspace(0, len(frame) - 1, cap, dtype=int)
    return frame.iloc[positions]


@dataclass
class _LinearState:
    features: list[str]
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float


class LogisticRidge:
    """Small deterministic logistic baseline with L2 regularisation."""

    def __init__(self, alpha: float = 10.0, max_iter: int = 30) -> None:
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.state: _LinearState | None = None

    def fit(self, frame: pd.DataFrame, features: list[str], target: str) -> "LogisticRidge":
        x = _matrix(frame, features)
        y = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(y)
        if keep.sum() < 2 or np.unique(y[keep]).size < 2:
            raise ValueError("logistic training requires both target classes")
        x, y = x[keep], y[keep]
        median = np.nanmedian(x, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        x = np.where(np.isfinite(x), x, median)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        x = (x - mean) / scale
        coef = np.zeros(x.shape[1], dtype=float)
        prevalence = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        intercept = float(np.log(prevalence / (1 - prevalence)))
        identity = np.eye(x.shape[1])
        for _ in range(self.max_iter):
            logits = np.clip(x @ coef + intercept, -35.0, 35.0)
            probability = 1.0 / (1.0 + np.exp(-logits))
            weight = np.clip(probability * (1 - probability), 1e-6, None)
            gradient_coef = x.T @ (probability - y) + self.alpha * coef
            gradient_intercept = float((probability - y).sum())
            hessian_coef = (x.T * weight) @ x + self.alpha * identity
            hessian_cross = x.T @ weight
            hessian = np.vstack(
                [
                    np.hstack([hessian_coef, hessian_cross[:, None]]),
                    np.hstack([hessian_cross, np.asarray([weight.sum()])])[None, :],
                ]
            )
            step = np.linalg.solve(hessian, np.r_[gradient_coef, gradient_intercept])
            coef -= step[:-1]
            intercept -= float(step[-1])
            if np.max(np.abs(step)) < 1e-7:
                break
        self.state = _LinearState(features, median, mean, scale, coef, intercept)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("model is not fitted")
        x = _matrix(frame, self.state.features)
        x = np.where(np.isfinite(x), x, self.state.median)
        x = (x - self.state.mean) / self.state.scale
        logits = np.clip(x @ self.state.coef + self.state.intercept, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def save(self, path: Path) -> None:
        if self.state is None:
            raise RuntimeError("model is not fitted")
        payload = {
            "kind": "logistic_ridge",
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            **{key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in vars(self.state).items()},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LogisticRidge":
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = cls(payload["alpha"], payload["max_iter"])
        model.state = _LinearState(
            payload["features"],
            np.asarray(payload["median"]),
            np.asarray(payload["mean"]),
            np.asarray(payload["scale"]),
            np.asarray(payload["coef"]),
            float(payload["intercept"]),
        )
        return model


class RidgeReturn:
    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = float(alpha)
        self.state: _LinearState | None = None

    def fit(self, frame: pd.DataFrame, features: list[str], target: str) -> "RidgeReturn":
        x = _matrix(frame, features)
        y = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(y)
        if keep.sum() < 2:
            raise ValueError("ridge training has insufficient finite targets")
        x, y = x[keep], y[keep]
        median = np.nanmedian(x, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        x = np.where(np.isfinite(x), x, median)
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        x = (x - mean) / scale
        intercept = float(y.mean())
        coef = np.linalg.solve(x.T @ x + self.alpha * np.eye(x.shape[1]), x.T @ (y - intercept))
        self.state = _LinearState(features, median, mean, scale, coef, intercept)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("model is not fitted")
        x = _matrix(frame, self.state.features)
        x = np.where(np.isfinite(x), x, self.state.median)
        return ((x - self.state.mean) / self.state.scale) @ self.state.coef + self.state.intercept

    def save(self, path: Path) -> None:
        if self.state is None:
            raise RuntimeError("model is not fitted")
        payload = {
            "kind": "ridge_return",
            "alpha": self.alpha,
            **{key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in vars(self.state).items()},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RidgeReturn":
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = cls(payload["alpha"])
        model.state = _LinearState(
            payload["features"], np.asarray(payload["median"]), np.asarray(payload["mean"]),
            np.asarray(payload["scale"]), np.asarray(payload["coef"]), float(payload["intercept"]),
        )
        return model


def _lightgbm() -> Any:
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - environment check
        raise RuntimeError("V30 requires the optional 'ml' dependency: pip install -e .[ml]") from exc
    return lgb


class LightGBMDirection:
    def __init__(self, params: dict[str, Any] | None = None, rounds: int = 120) -> None:
        self.params = params or {
            "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.04,
            "num_leaves": 15, "max_depth": 5, "min_data_in_leaf": 200,
            "feature_fraction": 0.8, "lambda_l1": 1.0, "lambda_l2": 5.0,
            "verbosity": -1, "seed": 42, "num_threads": 0,
        }
        self.rounds = int(rounds)
        self.features: list[str] = []
        self.booster: Any = None

    def fit(self, frame: pd.DataFrame, features: list[str], target: str) -> "LightGBMDirection":
        lgb = _lightgbm()
        self.features = list(features)
        x = _matrix(frame, self.features)
        y = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(y)
        if np.unique(y[keep]).size < 2:
            raise ValueError("LightGBM direction training requires both classes")
        self.booster = lgb.train(self.params, lgb.Dataset(x[keep], label=y[keep]), num_boost_round=self.rounds)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        return np.clip(np.asarray(self.booster.predict(_matrix(frame, self.features))), 1e-7, 1 - 1e-7)

    def save(self, path: Path) -> None:
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        self.booster.save_model(str(path))
        path.with_suffix(path.suffix + ".meta.json").write_text(
            json.dumps({"kind": "lightgbm_direction", "features": self.features}, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "LightGBMDirection":
        lgb = _lightgbm()
        model = cls()
        model.booster = lgb.Booster(model_file=str(path))
        model.features = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())["features"]
        return model


class LightGBMReturn(LightGBMDirection):
    def __init__(self, params: dict[str, Any] | None = None, rounds: int = 120) -> None:
        super().__init__(params or {
            "objective": "regression_l1", "metric": "l1", "learning_rate": 0.04,
            "num_leaves": 15, "max_depth": 5, "min_data_in_leaf": 200,
            "feature_fraction": 0.8, "lambda_l1": 1.0, "lambda_l2": 5.0,
            "verbosity": -1, "seed": 42, "num_threads": 0,
        }, rounds)

    def fit(self, frame: pd.DataFrame, features: list[str], target: str) -> "LightGBMReturn":
        lgb = _lightgbm()
        self.features = list(features)
        x = _matrix(frame, self.features)
        y = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(y)
        if keep.sum() < 2:
            raise ValueError("LightGBM return training has insufficient targets")
        self.booster = lgb.train(self.params, lgb.Dataset(x[keep], label=y[keep]), num_boost_round=self.rounds)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        return np.asarray(self.booster.predict(_matrix(frame, self.features)), dtype=float)

    def save(self, path: Path) -> None:
        super().save(path)
        metadata = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())
        metadata["kind"] = "lightgbm_return"
        path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
