from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research_v10.features import V10_FEATURES
from research_v10.model import V10Models, fit_v10_models, score_v10
from research_v10.research_config import V10Settings
from research_v12.model import mature_embargoed_training
from research_v13.config import V13Settings
from research_v13.model import TwoStageModel, confidence_lower_bound, group_percentile
from research_v4.stability import FactorSpec
from research_v5.models import V5Models

from .config import V14Settings
from .features import ANNOUNCEMENT_FEATURES


def stable_announcement_features(
    train: pd.DataFrame, settings: V14Settings
) -> tuple[list[str], dict[str, dict]]:
    available_features = [
        feature for feature in ANNOUNCEMENT_FEATURES if feature in train.columns
    ]
    daily_rows = []
    for date, group in train.groupby("date", sort=False):
        valid = group["v12_net_marginal_target"].notna()
        sample = group.loc[valid, ["v12_net_marginal_target", *available_features]]
        if len(sample) < 20:
            continue
        target = sample["v12_net_marginal_target"]
        varying = [
            feature
            for feature in available_features
            if sample[feature].nunique(dropna=True) > 1
        ]
        values = sample[varying].corrwith(target, method="spearman")
        daily_rows.append({"date": date, **values.to_dict()})
    daily = pd.DataFrame(daily_rows)
    diagnostics, selected = {}, []
    if daily.empty:
        return selected, diagnostics
    daily["year"] = pd.to_datetime(daily["date"]).dt.year
    event_years_available = int(
        train.assign(_year=pd.to_datetime(train["date"]).dt.year)
        .loc[
            train[available_features].abs().gt(1e-12).any(axis=1), "_year"
        ]
        .nunique()
    )
    required_years = min(settings.stable_min_years, max(1, event_years_available))
    annual = (
        daily.groupby("year").mean(numeric_only=True)
        .reindex(columns=ANNOUNCEMENT_FEATURES)
    )
    for feature in ANNOUNCEMENT_FEATURES:
        values = annual[feature].dropna()
        if len(values) < required_years:
            consistency, median_abs, direction = 0.0, 0.0, 0
        else:
            positive = float((values > 0).mean())
            negative = float((values < 0).mean())
            consistency = max(positive, negative)
            direction = 1 if positive >= negative else -1
            median_abs = float(values.abs().median())
        passed = (
            len(values) >= required_years
            and consistency >= settings.stable_sign_consistency
            and median_abs >= settings.stable_median_abs_ic
        )
        diagnostics[feature] = {
            "years": int(len(values)),
            "event_years_available": event_years_available,
            "required_years": required_years,
            "sign_consistency": consistency,
            "median_abs_ic": median_abs,
            "direction": direction,
            "selected": passed,
        }
        if passed:
            selected.append(feature)
    return selected, diagnostics


@dataclass
class EventTwoStageModel:
    features: list[str]
    classifier_: object | None = None
    magnitude_: object | None = None

    def fit(self, frame: pd.DataFrame, settings: V14Settings) -> "EventTwoStageModel":
        import lightgbm as lgb

        ordered = frame.sort_values(["date", "broad_sector", "symbol"]).copy()
        percentile = group_percentile(ordered)
        binary = (percentile > settings.classifier_top_quantile).astype(int)
        classifier_data = lgb.Dataset(
            ordered[self.features].to_numpy(dtype=float),
            label=binary.to_numpy(dtype=int),
            feature_name=self.features,
            free_raw_data=True,
        )
        self.classifier_ = lgb.train(
            {
                "objective": "binary", "metric": "binary_logloss",
                "learning_rate": 0.025, "num_leaves": 15, "max_depth": 5,
                "min_data_in_leaf": 200, "feature_fraction": 0.8,
                "lambda_l1": 1.0, "lambda_l2": 10.0, "scale_pos_weight": 4.0,
                "seed": 46, "num_threads": 4, "verbosity": -1,
            },
            classifier_data,
            num_boost_round=160,
        )
        magnitude_rows = percentile > settings.magnitude_training_quantile
        magnitude_data = lgb.Dataset(
            ordered.loc[magnitude_rows, self.features].to_numpy(dtype=float),
            label=ordered.loc[magnitude_rows, "v12_net_marginal_target"].to_numpy(dtype=float),
            feature_name=self.features,
            free_raw_data=True,
        )
        self.magnitude_ = lgb.train(
            {
                "objective": "regression_l1", "metric": "l1",
                "learning_rate": 0.025, "num_leaves": 15, "max_depth": 5,
                "min_data_in_leaf": 200, "feature_fraction": 0.8,
                "lambda_l1": 1.0, "lambda_l2": 10.0,
                "seed": 47, "num_threads": 4, "verbosity": -1,
            },
            magnitude_data,
            num_boost_round=160,
        )
        return self

    def predict_components(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.classifier_ is None or self.magnitude_ is None:
            raise RuntimeError("V14公告增强两阶段模型尚未训练")
        probability = np.asarray(self.classifier_.predict(frame[self.features]), dtype=float)
        magnitude = np.asarray(self.magnitude_.predict(frame[self.features]), dtype=float)
        return probability, magnitude

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        probability, magnitude = self.predict_components(frame)
        return probability * np.clip(magnitude, 0.0, 0.10)


def _date_payoffs(validation, prediction, settings):
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


@dataclass
class V14Models:
    event_model: EventTwoStageModel
    baseline_model: TwoStageModel
    v10: V10Models
    selected_features: list[str]
    stability_diagnostics: dict[str, dict]
    global_gate: bool
    technology_gate: bool
    validation_diagnostics: dict[int, dict]
    payoff_lower_bound: float
    incremental_lower_bound: float
    technology_lower_bound: float


def _nested_validation(dataset, test_year, settings):
    earliest = test_year - settings.training_window_years
    diagnostics = {}
    pooled, incremental, tech_pool, tech_incremental = [], [], [], []
    selected_counts = []
    for year in range(test_year - settings.validation_years, test_year):
        train = mature_embargoed_training(dataset, year, earliest, settings.embargo_calendar_days)
        validation = dataset[
            dataset["eligible"].fillna(False)
            & (pd.to_datetime(dataset["date"]).dt.year == year)
            & dataset["v12_net_marginal_target"].notna()
        ].copy()
        dates = validation["date"].drop_duplicates().sort_values().iloc[:: settings.rebalance_every]
        validation = validation[validation["date"].isin(dates)].copy()
        selected, _ = stable_announcement_features(train, settings)
        selected_counts.append(len(selected))
        event_model = EventTwoStageModel([*V10_FEATURES, *selected]).fit(train, settings)
        baseline = TwoStageModel().fit(train, V13Settings())
        event_payoff, event_tech, precision = _date_payoffs(validation, event_model.predict(validation), settings)
        base_payoff, base_tech, _ = _date_payoffs(validation, baseline.predict(validation), settings)
        count = min(len(event_payoff), len(base_payoff))
        tech_count = min(len(event_tech), len(base_tech))
        pooled.extend(event_payoff[:count])
        incremental.extend(np.asarray(event_payoff[:count]) - np.asarray(base_payoff[:count]))
        tech_pool.extend(event_tech[:tech_count])
        tech_incremental.extend(np.asarray(event_tech[:tech_count]) - np.asarray(base_tech[:tech_count]))
        tech_train = train[train["broad_sector"] == "technology"]
        diagnostics[year] = {
            "event_payoff_mean": float(np.mean(event_payoff)) if event_payoff else float("nan"),
            "baseline_payoff_mean": float(np.mean(base_payoff)) if base_payoff else float("nan"),
            "incremental_mean": float(np.mean(incremental[-count:])) if count else float("nan"),
            "technology_payoff_mean": float(np.mean(event_tech)) if event_tech else float("nan"),
            "top30_precision": float(np.mean(precision)) if precision else float("nan"),
            "selected_announcement_features": len(selected),
            "technology_sample_valid": len(tech_train) >= settings.minimum_technology_rows and tech_train["date"].nunique() >= settings.minimum_technology_dates,
        }
    payoff_lower = confidence_lower_bound(pooled, settings.confidence_z)
    incremental_lower = confidence_lower_bound(incremental, settings.confidence_z)
    tech_lower = confidence_lower_bound(tech_pool, settings.confidence_z)
    tech_incremental_lower = confidence_lower_bound(tech_incremental, settings.confidence_z)
    floor = all(np.isfinite(v["event_payoff_mean"]) and v["event_payoff_mean"] >= settings.validation_year_floor for v in diagnostics.values())
    tech_floor = all(v["technology_sample_valid"] and np.isfinite(v["technology_payoff_mean"]) and v["technology_payoff_mean"] >= settings.validation_year_floor for v in diagnostics.values())
    new_data_present = all(count > 0 for count in selected_counts)
    global_gate = bool(new_data_present and np.isfinite(payoff_lower) and payoff_lower > 0 and np.isfinite(incremental_lower) and incremental_lower > 0 and floor)
    technology_gate = bool(new_data_present and np.isfinite(tech_lower) and tech_lower > 0 and np.isfinite(tech_incremental_lower) and tech_incremental_lower > 0 and tech_floor)
    return diagnostics, global_gate, technology_gate, payoff_lower, incremental_lower, tech_lower


def fit_v14_models(dataset, test_year, settings=None):
    settings = settings or V14Settings()
    earliest = test_year - settings.training_window_years
    train = mature_embargoed_training(dataset, test_year, earliest, settings.embargo_calendar_days)
    selected, stability = stable_announcement_features(train, settings)
    diagnostics, global_gate, tech_gate, payoff_lower, incremental_lower, tech_lower = _nested_validation(dataset, test_year, settings)
    return V14Models(
        event_model=EventTwoStageModel([*V10_FEATURES, *selected]).fit(train, settings),
        baseline_model=TwoStageModel().fit(train, V13Settings()),
        v10=fit_v10_models(dataset, test_year, V10Settings()),
        selected_features=selected, stability_diagnostics=stability,
        global_gate=global_gate, technology_gate=tech_gate,
        validation_diagnostics=diagnostics, payoff_lower_bound=payoff_lower,
        incremental_lower_bound=incremental_lower, technology_lower_bound=tech_lower,
    )


def _sector_rank(frame, values):
    return pd.Series(values, index=frame.index).groupby([frame["date"], frame["broad_sector"]]).rank(pct=True, method="average").sub(0.5).fillna(0)


def score_v14(current, models, v5_models: V5Models, v4_specs: list[FactorSpec], settings=None):
    settings = settings or V14Settings()
    scored = score_v10(current, models.v10, v5_models, v4_specs, V10Settings())
    event_probability, event_magnitude = models.event_model.predict_components(scored)
    baseline_probability, baseline_magnitude = models.baseline_model.predict_components(scored)
    scored["event_probability"] = event_probability
    scored["event_conditional_magnitude"] = event_magnitude
    scored["v13_baseline_score"] = _sector_rank(scored, baseline_probability * np.clip(baseline_magnitude, 0, 0.10))
    scored["v14_event_score"] = _sector_rank(scored, event_probability * np.clip(event_magnitude, 0, 0.10))
    scored["v13_comparable_score"] = (
        settings.event_model_share * scored["v13_baseline_score"]
        + settings.v10_global_share * scored["global_model_score"]
    )
    scored["v14_score"] = settings.event_model_share * scored["v14_event_score"] + settings.v10_global_share * scored["global_model_score"]
    scored["score"] = scored["v14_score"]
    return scored
