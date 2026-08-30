from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from research_v4.config import V4Settings
from research_v4.stability import learn_factor_specs
from research_v5.models import fit_v5_models
from research_v6.config import V6Settings
from research_v6.model import score_v6

from .config import ChallengerSettings


@dataclass
class TrainOnlyPreprocessor:
    lower_: pd.Series | None = None
    upper_: pd.Series | None = None
    median_: pd.Series | None = None
    mean_: pd.Series | None = None
    scale_: pd.Series | None = None
    fit_row_ids_: frozenset[str] = frozenset()

    def fit(self, frame: pd.DataFrame, features: tuple[str, ...]) -> "TrainOnlyPreprocessor":
        values = frame[list(features)].replace([np.inf, -np.inf], np.nan).astype(float)
        self.lower_ = values.quantile(0.01)
        self.upper_ = values.quantile(0.99)
        clipped = values.clip(self.lower_, self.upper_, axis=1)
        self.median_ = clipped.median()
        filled = clipped.fillna(self.median_)
        self.mean_ = filled.mean()
        self.scale_ = filled.std(ddof=0).replace(0, 1.0).fillna(1.0)
        ids = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d") + ":" + frame["symbol"]
        self.fit_row_ids_ = frozenset(ids)
        return self

    def transform(self, frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
        if any(value is None for value in (self.lower_, self.upper_, self.median_, self.mean_, self.scale_)):
            raise RuntimeError("preprocessor is not fitted")
        values = frame[list(features)].replace([np.inf, -np.inf], np.nan).astype(float)
        clipped = values.clip(self.lower_, self.upper_, axis=1)
        filled = clipped.fillna(self.median_)
        return ((filled - self.mean_) / self.scale_).to_numpy(dtype=float)


def deterministic_full_date_sample(frame: pd.DataFrame, row_cap: int) -> pd.DataFrame:
    ordered = frame.sort_values(["date", "symbol"])
    if len(ordered) <= row_cap:
        return ordered
    counts = ordered.groupby("date", sort=True).size()
    target_dates = max(1, int(row_cap / counts.mean()))
    positions = np.linspace(0, len(counts) - 1, target_dates, dtype=int)
    selected_dates = set(counts.index[positions])
    return ordered[ordered["date"].isin(selected_dates)].sort_values(["date", "symbol"])


@dataclass
class RidgeModel:
    alpha: float
    coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeModel":
        design = np.column_stack([np.ones(len(x)), x])
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("ridge is not fitted")
        return np.column_stack([np.ones(len(x)), x]) @ self.coef_

    def signature(self) -> str:
        if self.coef_ is None:
            raise RuntimeError("ridge is not fitted")
        return hashlib.sha256(self.coef_.tobytes()).hexdigest()


@dataclass
class LightGBMModel:
    objective: str
    rounds: int
    seed: int
    booster_: object | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> "LightGBMModel":
        import lightgbm as lgb

        target = np.asarray(y, dtype=float)
        metric = "l1"
        params = {
            "objective": self.objective,
            "metric": metric,
            "learning_rate": 0.04,
            "num_leaves": 15,
            "max_depth": 5,
            "min_data_in_leaf": 200,
            "feature_fraction": 0.8,
            "lambda_l1": 1.0,
            "lambda_l2": 5.0,
            "seed": self.seed,
            "feature_fraction_seed": self.seed,
            "data_random_seed": self.seed,
            "num_threads": 4,
            "verbosity": -1,
            "deterministic": True,
            "force_col_wise": True,
        }
        train_groups = None
        if self.objective == "lambdarank":
            if groups is None or int(np.sum(groups)) != len(target):
                raise ValueError("LambdaRank requires complete date groups")
            target = np.minimum(np.floor(np.clip(target, 0, 0.999999) * 5), 4).astype(int)
            train_groups = groups
            params["metric"] = "ndcg"
            params["ndcg_eval_at"] = [10, 20, 30]
        dataset = lgb.Dataset(x, label=target, group=train_groups, free_raw_data=True)
        self.booster_ = lgb.train(params, dataset, num_boost_round=self.rounds)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.booster_ is None:
            raise RuntimeError("LightGBM is not fitted")
        return np.asarray(self.booster_.predict(x), dtype=float)

    def importance(self, features: tuple[str, ...]) -> dict[str, float]:
        if self.booster_ is None:
            return {}
        values = np.asarray(self.booster_.feature_importance(importance_type="gain"), dtype=float)
        total = values.sum()
        if total > 0:
            values /= total
        return {name: float(value) for name, value in zip(features, values)}

    def signature(self) -> str:
        if self.booster_ is None:
            raise RuntimeError("LightGBM is not fitted")
        payload = self.booster_.model_to_string().encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def fit_candidate_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
    target: str,
    settings: ChallengerSettings,
) -> tuple[dict[str, np.ndarray], list[dict], TrainOnlyPreprocessor, dict[str, object]]:
    numeric_target = pd.to_numeric(train[target], errors="coerce")
    finite_target = numeric_target.notna() & np.isfinite(numeric_target)
    clean_train = train.loc[finite_target].copy()
    if clean_train.empty:
        raise RuntimeError(f"no finite mature training target: {target}")
    sampled = deterministic_full_date_sample(clean_train, settings.training_row_cap)
    processor = TrainOnlyPreprocessor().fit(sampled, features)
    x_train = processor.transform(sampled, features)
    x_test = processor.transform(test, features)
    y_train = pd.to_numeric(sampled[target], errors="raise").to_numpy(dtype=float)
    groups = sampled.groupby("date", sort=True).size().to_numpy(dtype=int)
    models = {
        "ridge": RidgeModel(settings.ridge_alpha).fit(x_train, y_train),
        "lightgbm_regression": LightGBMModel(
            "regression_l1", settings.lightgbm_rounds, settings.random_seed
        ).fit(x_train, y_train),
        "lightgbm_lambdarank": LightGBMModel(
            "lambdarank", settings.lightgbm_rounds, settings.random_seed
        ).fit(x_train, y_train, groups),
    }
    predictions = {name: model.predict(x_test) for name, model in models.items()}
    rows = []
    for name, model in models.items():
        importance = model.importance(features) if isinstance(model, LightGBMModel) else {}
        for feature in features:
            rows.append(
                {
                    "model": name,
                    "feature": feature,
                    "gain_importance": importance.get(feature, np.nan),
                    "model_signature": model.signature(),
                    "training_rows": int(len(sampled)),
                }
            )
    return predictions, rows, processor, models


def v6_oos_scores(
    dataset: pd.DataFrame, test: pd.DataFrame, test_year: int
) -> pd.Series:
    v5_models = fit_v5_models(dataset, test_year)
    v4_specs, _ = learn_factor_specs(dataset, test_year, V4Settings())
    pieces = []
    for _, current in test.groupby("date", sort=True):
        scored = score_v6(current, v5_models, v4_specs, V6Settings())
        pieces.append(scored["score"])
    result = pd.concat(pieces).sort_index()
    if not result.index.equals(test.index.sort_values()):
        result = result.reindex(test.index)
    if result.isna().any():
        raise RuntimeError("V6 scoring produced missing values")
    return result
