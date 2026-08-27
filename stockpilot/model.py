from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class RankingModel(Protocol):
    def fit(
        self,
        x: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        group_sizes: np.ndarray | None = None,
    ) -> RankingModel: ...

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray: ...

    def feature_weights(self, names: list[str]) -> dict[str, float]: ...


@dataclass
class RidgeRanker:
    alpha: float = 10.0
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    coef_: np.ndarray | None = None

    def fit(
        self,
        x: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        group_sizes: np.ndarray | None = None,
    ) -> RidgeRanker:
        features = np.asarray(x, dtype=float)
        target = np.asarray(y, dtype=float)
        valid = np.isfinite(features).all(axis=1) & np.isfinite(target)
        features, target = features[valid], target[valid]
        if len(target) < features.shape[1] * 3:
            raise ValueError("训练样本不足")
        self.mean_ = features.mean(axis=0)
        self.scale_ = features.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        z = (features - self.mean_) / self.scale_
        design = np.column_stack([np.ones(len(z)), z])
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0
        self.coef_ = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        return self

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("模型尚未训练")
        features = np.asarray(x, dtype=float)
        z = (features - self.mean_) / self.scale_
        return np.column_stack([np.ones(len(z)), z]) @ self.coef_

    def feature_weights(self, names: list[str]) -> dict[str, float]:
        if self.coef_ is None:
            return {}
        return {name: float(value) for name, value in zip(names, self.coef_[1:])}


@dataclass
class FeatureRanker:
    feature: str
    direction: float = 1.0

    def fit(
        self,
        x: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        group_sizes: np.ndarray | None = None,
    ) -> FeatureRanker:
        return self

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not isinstance(x, pd.DataFrame):
            raise TypeError("规则模型需要带列名的DataFrame")
        return x[self.feature].to_numpy(dtype=float) * self.direction

    def feature_weights(self, names: list[str]) -> dict[str, float]:
        return {name: self.direction if name == self.feature else 0.0 for name in names}


@dataclass
class LightGBMRanker:
    model_: object | None = None

    def fit(
        self,
        x: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        group_sizes: np.ndarray | None = None,
    ) -> LightGBMRanker:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError("LightGBM未安装，请运行 pip install -e .[ml]") from exc
        features = np.asarray(x, dtype=float)
        target = np.asarray(y, dtype=float)
        valid = np.isfinite(features).all(axis=1) & np.isfinite(target)
        feature_names = list(x.columns) if isinstance(x, pd.DataFrame) else "auto"
        objective = "regression_l1"
        metric = "l1"
        train_target = target[valid]
        train_groups = None
        if group_sizes is not None:
            if not valid.all():
                raise ValueError("分组排序训练不允许包含非有限样本")
            relevance = np.empty(len(target), dtype=int)
            offset = 0
            for size in group_sizes:
                values = pd.Series(target[offset : offset + size])
                percentiles = values.rank(pct=True, method="first").to_numpy(dtype=float)
                relevance[offset : offset + size] = np.ceil(percentiles * 5).clip(1, 5) - 1
                offset += size
            if offset != len(target):
                raise ValueError("group_sizes 与训练样本数不一致")
            train_target = relevance
            train_groups = group_sizes
            objective = "lambdarank"
            metric = "ndcg"
        train_data = lgb.Dataset(
            features[valid],
            label=train_target,
            group=train_groups,
            feature_name=feature_names,
            free_raw_data=True,
        )
        self.model_ = lgb.train(
            {
                "objective": objective,
                "metric": metric,
                "learning_rate": 0.04,
                "num_leaves": 15,
                "max_depth": 5,
                "min_data_in_leaf": 200,
                "feature_fraction": 0.8,
                "lambda_l1": 1.0,
                "lambda_l2": 5.0,
                "seed": 42,
                "num_threads": 4,
                "verbosity": -1,
            },
            train_data,
            num_boost_round=120,
        )
        return self

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("模型尚未训练")
        return np.asarray(self.model_.predict(x), dtype=float)

    def feature_weights(self, names: list[str]) -> dict[str, float]:
        if self.model_ is None:
            return {}
        importance = np.asarray(self.model_.feature_importance(), dtype=float)
        total = importance.sum()
        if total > 0:
            importance /= total
        return {name: float(value) for name, value in zip(names, importance)}


def create_model(name: str, ridge_alpha: float = 10.0) -> RankingModel:
    models: dict[str, RankingModel] = {
        "ridge": RidgeRanker(ridge_alpha),
        "lightgbm": LightGBMRanker(),
        "momentum_20": FeatureRanker("ret_20_rank", 1.0),
        "momentum_60": FeatureRanker("momentum_60_rank", 1.0),
        "mean_reversion_5": FeatureRanker("ret_5_rank", -1.0),
        "low_volatility": FeatureRanker("volatility_20_rank", -1.0),
    }
    try:
        return models[name]
    except KeyError as exc:
        raise ValueError(f"未知模型 {name}，可选: {', '.join(models)}") from exc
