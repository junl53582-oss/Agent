from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research_v10.features import V10_FEATURES
from research_v10.model import V10Models, fit_v10_models, score_v10
from research_v10.research_config import V10Settings
from research_v12.model import mature_embargoed_training
from research_v4.stability import FactorSpec
from research_v5.models import V5Models

from .config import V13Settings


def group_percentile(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby(["date", "broad_sector"])["v12_net_marginal_target"].rank(
        pct=True, method="first"
    )


@dataclass
class TwoStageModel:
    classifier_: object | None = None
    magnitude_: object | None = None

    def fit(self, frame: pd.DataFrame, settings: V13Settings) -> "TwoStageModel":
        import lightgbm as lgb

        ordered = frame.sort_values(["date", "broad_sector", "symbol"]).copy()
        percentile = group_percentile(ordered)
        binary = (percentile > settings.classifier_top_quantile).astype(int)
        classifier_data = lgb.Dataset(
            ordered[V10_FEATURES].to_numpy(dtype=float),
            label=binary.to_numpy(dtype=int),
            feature_name=V10_FEATURES,
            free_raw_data=True,
        )
        self.classifier_ = lgb.train(
            {
                "objective": "binary", "metric": "binary_logloss",
                "learning_rate": 0.025, "num_leaves": 15, "max_depth": 5,
                "min_data_in_leaf": 200, "feature_fraction": 0.8,
                "lambda_l1": 1.0, "lambda_l2": 10.0,
                "scale_pos_weight": 4.0, "seed": 44,
                "num_threads": 4, "verbosity": -1,
            },
            classifier_data,
            num_boost_round=160,
        )
        magnitude_rows = percentile > settings.magnitude_training_quantile
        magnitude_data = lgb.Dataset(
            ordered.loc[magnitude_rows, V10_FEATURES].to_numpy(dtype=float),
            label=ordered.loc[magnitude_rows, "v12_net_marginal_target"].to_numpy(dtype=float),
            feature_name=V10_FEATURES,
            free_raw_data=True,
        )
        self.magnitude_ = lgb.train(
            {
                "objective": "regression_l1", "metric": "l1",
                "learning_rate": 0.025, "num_leaves": 15, "max_depth": 5,
                "min_data_in_leaf": 200, "feature_fraction": 0.8,
                "lambda_l1": 1.0, "lambda_l2": 10.0,
                "seed": 45, "num_threads": 4, "verbosity": -1,
            },
            magnitude_data,
            num_boost_round=160,
        )
        return self

    def predict_components(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.classifier_ is None or self.magnitude_ is None:
            raise RuntimeError("V13两阶段模型尚未训练")
        features = frame[V10_FEATURES]
        probability = np.asarray(self.classifier_.predict(features), dtype=float)
        magnitude = np.asarray(self.magnitude_.predict(features), dtype=float)
        return probability, magnitude

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        probability, magnitude = self.predict_components(frame)
        return probability * np.clip(magnitude, 0.0, 0.10)


@dataclass
class V13Models:
    two_stage: TwoStageModel
    v10: V10Models
    global_gate: bool
    technology_gate: bool
    validation_diagnostics: dict[int, dict[str, float | int | bool]]
    global_lower_bound: float
    technology_lower_bound: float
    training_rows: int
    training_end: pd.Timestamp


def _date_payoffs(
    validation: pd.DataFrame, prediction: np.ndarray, settings: V13Settings
) -> tuple[list[float], list[float], list[float]]:
    scored = validation.copy()
    scored["prediction"] = prediction
    portfolio, technology, precision = [], [], []
    for _, group in scored.groupby("date"):
        eligible = group[group["eligible"].fillna(False) & group["v12_net_marginal_target"].notna()]
        parts = []
        for _, sector in eligible.groupby("broad_sector"):
            weight = float(pd.to_numeric(sector["benchmark_weight"], errors="coerce").clip(lower=0).sum())
            count = max(1, int(round(settings.active_top_n * weight)))
            parts.append(sector.nlargest(min(count, len(sector)), "prediction"))
        selected = pd.concat(parts).nlargest(settings.active_top_n, "prediction") if parts else eligible.iloc[0:0]
        if selected.empty:
            continue
        portfolio.append(float(selected["v12_net_marginal_target"].mean()))
        truth = set(eligible.nlargest(settings.active_top_n, "v12_net_marginal_target")["symbol"])
        precision.append(len(set(selected["symbol"]) & truth) / settings.active_top_n)
        tech = eligible[eligible["broad_sector"] == "technology"]
        if len(tech) >= settings.technology_top_n * 2:
            technology.append(float(tech.nlargest(settings.technology_top_n, "prediction")["v12_net_marginal_target"].mean()))
    return portfolio, technology, precision


def confidence_lower_bound(values: list[float], z: float) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 2:
        return float("nan")
    return float(array.mean() - z * array.std(ddof=1) / np.sqrt(len(array)))


def _nested_validation(dataset: pd.DataFrame, test_year: int, settings: V13Settings):
    earliest = test_year - settings.training_window_years
    diagnostics = {}
    pooled, pooled_tech = [], []
    for year in range(test_year - settings.validation_years, test_year):
        train = mature_embargoed_training(dataset, year, earliest, settings.embargo_calendar_days)
        validation = dataset[
            dataset["eligible"].fillna(False)
            & (pd.to_datetime(dataset["date"]).dt.year == year)
            & dataset["v12_net_marginal_target"].notna()
        ].copy()
        dates = validation["date"].drop_duplicates().sort_values().iloc[:: settings.rebalance_every]
        validation = validation[validation["date"].isin(dates)].copy()
        if train.empty or validation.empty:
            diagnostics[year] = {"portfolio_mean": float("nan"), "technology_mean": float("nan"), "top30_precision": float("nan"), "periods": 0, "technology_sample_valid": False}
            continue
        model = TwoStageModel().fit(train, settings)
        payoffs, tech_payoffs, precisions = _date_payoffs(validation, model.predict(validation), settings)
        pooled.extend(payoffs)
        pooled_tech.extend(tech_payoffs)
        tech_train = train[train["broad_sector"] == "technology"]
        enough = len(tech_train) >= settings.minimum_technology_rows and tech_train["date"].nunique() >= settings.minimum_technology_dates
        diagnostics[year] = {
            "portfolio_mean": float(np.mean(payoffs)) if payoffs else float("nan"),
            "technology_mean": float(np.mean(tech_payoffs)) if tech_payoffs else float("nan"),
            "top30_precision": float(np.mean(precisions)) if precisions else float("nan"),
            "periods": len(payoffs), "technology_sample_valid": enough,
        }
    global_lower = confidence_lower_bound(pooled, settings.confidence_z)
    tech_lower = confidence_lower_bound(pooled_tech, settings.confidence_z)
    year_floor = all(np.isfinite(float(v["portfolio_mean"])) and float(v["portfolio_mean"]) >= settings.validation_year_floor for v in diagnostics.values())
    tech_floor = all(bool(v["technology_sample_valid"]) and np.isfinite(float(v["technology_mean"])) and float(v["technology_mean"]) >= settings.validation_year_floor for v in diagnostics.values())
    return diagnostics, bool(np.isfinite(global_lower) and global_lower > 0 and year_floor), bool(np.isfinite(tech_lower) and tech_lower > 0 and tech_floor), global_lower, tech_lower


def fit_v13_models(dataset: pd.DataFrame, test_year: int, settings: V13Settings | None = None) -> V13Models:
    settings = settings or V13Settings()
    earliest = test_year - settings.training_window_years
    train = mature_embargoed_training(dataset, test_year, earliest, settings.embargo_calendar_days)
    if train.empty:
        raise RuntimeError(f"{test_year}没有足够的V13成熟训练数据")
    diagnostics, global_gate, tech_gate, global_lower, tech_lower = _nested_validation(dataset, test_year, settings)
    return V13Models(
        two_stage=TwoStageModel().fit(train, settings),
        v10=fit_v10_models(dataset, test_year, V10Settings()),
        global_gate=global_gate, technology_gate=tech_gate,
        validation_diagnostics=diagnostics,
        global_lower_bound=global_lower, technology_lower_bound=tech_lower,
        training_rows=len(train), training_end=pd.to_datetime(train["label_end_date_20"]).max(),
    )


def score_v13(current: pd.DataFrame, models: V13Models, v5_models: V5Models, v4_specs: list[FactorSpec], settings: V13Settings | None = None) -> pd.DataFrame:
    settings = settings or V13Settings()
    scored = score_v10(current, models.v10, v5_models, v4_specs, V10Settings())
    probability, magnitude = models.two_stage.predict_components(scored)
    scored["top_probability"] = probability
    scored["conditional_magnitude"] = magnitude
    raw = pd.Series(probability * np.clip(magnitude, 0, 0.10), index=scored.index)
    scored["two_stage_score"] = raw.groupby([scored["date"], scored["broad_sector"]]).rank(pct=True, method="average").sub(0.5).fillna(0)
    scored["v13_score"] = settings.two_stage_share * scored["two_stage_score"] + settings.v10_global_share * scored["global_model_score"]
    scored["score"] = scored["v13_score"]
    return scored

