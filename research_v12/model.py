from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research_v10.features import V10_FEATURES
from research_v10.model import V10Models, fit_v10_models, score_v10
from research_v10.research_config import V10Settings
from research_v4.stability import FactorSpec
from research_v5.models import V5Models

from .config import V12Settings


def mature_embargoed_training(
    dataset: pd.DataFrame, cutoff_year: int, earliest_year: int, embargo_days: int
) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff_year, 1, 1)
    embargo = cutoff - pd.Timedelta(days=embargo_days)
    dates = pd.to_datetime(dataset["date"])
    return dataset[
        dataset["eligible"].fillna(False)
        & dataset["v12_net_marginal_target"].notna()
        & (pd.to_datetime(dataset["label_end_date_20"]) < embargo)
        & (dates < cutoff)
        & (dates.dt.year >= earliest_year)
    ].sort_values(["date", "broad_sector", "symbol"])


def portfolio_relevance(frame: pd.DataFrame) -> pd.Series:
    percentile = frame.groupby(["date", "broad_sector"])["v12_net_marginal_target"].rank(
        pct=True, method="first"
    )
    relevance = pd.Series(0, index=frame.index, dtype=int)
    relevance.loc[percentile > 0.40] = 1
    relevance.loc[percentile > 0.60] = 2
    relevance.loc[percentile > 0.80] = 3
    relevance.loc[percentile > 0.90] = 4
    return relevance


@dataclass
class PortfolioRanker:
    model_: object | None = None

    def fit(self, frame: pd.DataFrame) -> "PortfolioRanker":
        import lightgbm as lgb

        ordered = frame.sort_values(["date", "broad_sector", "symbol"]).copy()
        relevance = portfolio_relevance(ordered)
        groups = ordered.groupby(["date", "broad_sector"], sort=True).size().to_numpy(dtype=int)
        if int(groups.sum()) != len(ordered) or relevance.isna().any():
            raise ValueError("V12行业分组排序输入无效")
        train = lgb.Dataset(
            ordered[V10_FEATURES].to_numpy(dtype=float),
            label=relevance.to_numpy(dtype=int),
            group=groups,
            feature_name=V10_FEATURES,
            free_raw_data=True,
        )
        self.model_ = lgb.train(
            {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [5, 10],
                "label_gain": [0, 1, 3, 7, 15],
                "learning_rate": 0.025,
                "num_leaves": 15,
                "max_depth": 5,
                "min_data_in_leaf": 200,
                "feature_fraction": 0.8,
                "lambda_l1": 1.0,
                "lambda_l2": 10.0,
                "seed": 43,
                "num_threads": 4,
                "verbosity": -1,
            },
            train,
            num_boost_round=160,
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("V12组合排序模型尚未训练")
        return np.asarray(self.model_.predict(frame[V10_FEATURES]), dtype=float)


@dataclass
class V12Models:
    portfolio_ranker: PortfolioRanker
    v10: V10Models
    global_gate: bool
    technology_gate: bool
    validation_diagnostics: dict[int, dict[str, float | bool]]
    training_rows: int
    training_end: pd.Timestamp


def _centered_sector_rank(frame: pd.DataFrame, values: np.ndarray) -> pd.Series:
    raw = pd.Series(values, index=frame.index)
    return raw.groupby([frame["date"], frame["broad_sector"]]).rank(
        pct=True, method="average"
    ).sub(0.5).fillna(0)


def _validation_payoff(
    validation: pd.DataFrame, prediction: np.ndarray, settings: V12Settings
) -> tuple[float, float, float]:
    scored = validation.copy()
    scored["prediction"] = prediction
    portfolio_payoffs: list[float] = []
    tech_payoffs: list[float] = []
    precisions: list[float] = []
    for _, group in scored.groupby("date"):
        eligible = group[group["eligible"].fillna(False) & group["v12_net_marginal_target"].notna()]
        selected_parts = []
        for _, sector in eligible.groupby("broad_sector"):
            sector_weight = float(pd.to_numeric(sector["benchmark_weight"], errors="coerce").clip(lower=0).sum())
            count = max(1, int(round(settings.active_top_n * sector_weight)))
            selected_parts.append(sector.nlargest(min(count, len(sector)), "prediction"))
        selected = pd.concat(selected_parts).nlargest(settings.active_top_n, "prediction") if selected_parts else eligible.iloc[0:0]
        if selected.empty:
            continue
        portfolio_payoffs.append(float(selected["v12_net_marginal_target"].mean()))
        truth = set(eligible.nlargest(settings.active_top_n, "v12_net_marginal_target")["symbol"])
        precisions.append(len(set(selected["symbol"]) & truth) / settings.active_top_n)
        technology = eligible[eligible["broad_sector"] == "technology"]
        if len(technology) >= settings.technology_top_n * 2:
            tech_selected = technology.nlargest(settings.technology_top_n, "prediction")
            tech_payoffs.append(float(tech_selected["v12_net_marginal_target"].mean()))
    return (
        float(np.mean(portfolio_payoffs)) if portfolio_payoffs else float("nan"),
        float(np.mean(tech_payoffs)) if tech_payoffs else float("nan"),
        float(np.mean(precisions)) if precisions else float("nan"),
    )


def _nested_validation(
    dataset: pd.DataFrame, test_year: int, settings: V12Settings
) -> tuple[dict[int, dict[str, float | bool]], bool, bool]:
    earliest = test_year - settings.training_window_years
    diagnostics: dict[int, dict[str, float | bool]] = {}
    for year in range(test_year - settings.validation_years, test_year):
        train = mature_embargoed_training(
            dataset, year, earliest, settings.embargo_calendar_days
        )
        validation = dataset[
            dataset["eligible"].fillna(False)
            & (pd.to_datetime(dataset["date"]).dt.year == year)
            & dataset["v12_net_marginal_target"].notna()
        ].copy()
        dates = validation["date"].drop_duplicates().sort_values().iloc[:: settings.rebalance_every]
        validation = validation[validation["date"].isin(dates)].copy()
        if train.empty or validation.empty:
            diagnostics[year] = {
                "portfolio_net_payoff": float("nan"),
                "technology_net_payoff": float("nan"),
                "top30_precision": float("nan"),
                "technology_sample_valid": False,
            }
            continue
        model = PortfolioRanker().fit(train)
        portfolio_payoff, tech_payoff, precision = _validation_payoff(
            validation, model.predict(validation), settings
        )
        tech_train = train[train["broad_sector"] == "technology"]
        enough_tech = (
            len(tech_train) >= settings.minimum_technology_rows
            and tech_train["date"].nunique() >= settings.minimum_technology_dates
        )
        diagnostics[year] = {
            "portfolio_net_payoff": portfolio_payoff,
            "technology_net_payoff": tech_payoff,
            "top30_precision": precision,
            "technology_sample_valid": enough_tech,
        }
    global_gate = len(diagnostics) == settings.validation_years and all(
        np.isfinite(float(value["portfolio_net_payoff"]))
        and float(value["portfolio_net_payoff"]) > 0
        for value in diagnostics.values()
    )
    technology_gate = len(diagnostics) == settings.validation_years and all(
        bool(value["technology_sample_valid"])
        and np.isfinite(float(value["technology_net_payoff"]))
        and float(value["technology_net_payoff"]) > 0
        for value in diagnostics.values()
    )
    return diagnostics, global_gate, technology_gate


def fit_v12_models(
    dataset: pd.DataFrame, test_year: int, settings: V12Settings | None = None
) -> V12Models:
    settings = settings or V12Settings()
    earliest = test_year - settings.training_window_years
    train = mature_embargoed_training(
        dataset, test_year, earliest, settings.embargo_calendar_days
    )
    if train.empty:
        raise RuntimeError(f"{test_year}没有足够的V12成熟训练数据")
    diagnostics, global_gate, technology_gate = _nested_validation(
        dataset, test_year, settings
    )
    return V12Models(
        portfolio_ranker=PortfolioRanker().fit(train),
        v10=fit_v10_models(dataset, test_year, V10Settings()),
        global_gate=global_gate,
        technology_gate=technology_gate,
        validation_diagnostics=diagnostics,
        training_rows=len(train),
        training_end=pd.to_datetime(train["label_end_date_20"]).max(),
    )


def score_v12(
    current: pd.DataFrame,
    models: V12Models,
    v5_models: V5Models,
    v4_specs: list[FactorSpec],
    settings: V12Settings | None = None,
) -> pd.DataFrame:
    settings = settings or V12Settings()
    scored = score_v10(current, models.v10, v5_models, v4_specs, V10Settings())
    scored["portfolio_rank_score"] = _centered_sector_rank(
        scored, models.portfolio_ranker.predict(scored)
    )
    scored["v12_score"] = (
        settings.portfolio_rank_share * scored["portfolio_rank_score"]
        + settings.v10_global_share * scored["global_model_score"]
    )
    scored["score"] = scored["v12_score"]
    return scored

