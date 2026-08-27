from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research_v10.features import V10_FEATURES
from research_v10.model import V10Models, fit_v10_models, mature_training, score_v10
from research_v10.research_config import V10Settings
from research_v4.stability import FactorSpec
from research_v5.models import V5Models

from .config import V11Settings


@dataclass
class TailRanker:
    model_: object | None = None

    def fit(self, frame: pd.DataFrame) -> "TailRanker":
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("V11需要LightGBM") from exc
        ordered = frame.sort_values(["date", "symbol"]).copy()
        relevance = tail_relevance(ordered)
        groups = ordered.groupby("date", sort=True).size().to_numpy(dtype=int)
        if len(ordered) != int(groups.sum()) or relevance.isna().any():
            raise ValueError("V11分组排序输入无效")
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
                "ndcg_eval_at": [5, 10, 30],
                "label_gain": [0, 1, 3, 7, 15],
                "learning_rate": 0.03,
                "num_leaves": 15,
                "max_depth": 5,
                "min_data_in_leaf": 200,
                "feature_fraction": 0.8,
                "lambda_l1": 1.0,
                "lambda_l2": 8.0,
                "seed": 42,
                "num_threads": 4,
                "verbosity": -1,
            },
            train,
            num_boost_round=140,
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("V11尾部模型尚未训练")
        return np.asarray(self.model_.predict(frame[V10_FEATURES]), dtype=float)


@dataclass
class V11Models:
    tail: TailRanker
    v10: V10Models
    global_gate: bool
    technology_gate: bool
    validation_diagnostics: dict[int, dict[str, float | bool]]
    training_rows: int
    training_end: pd.Timestamp


def tail_relevance(frame: pd.DataFrame) -> pd.Series:
    """Map each day's residual-return ranks to a top-heavy 0..4 label."""
    percentile = frame.groupby("date")["v10_target_20"].rank(
        pct=True, method="first"
    )
    relevance = pd.Series(0, index=frame.index, dtype=int)
    relevance.loc[percentile > 0.40] = 1
    relevance.loc[percentile > 0.60] = 2
    relevance.loc[percentile > 0.80] = 3
    relevance.loc[percentile > 0.90] = 4
    return relevance


def _centered_rank(values: np.ndarray, index: pd.Index) -> pd.Series:
    return pd.Series(values, index=index).rank(pct=True, method="average").sub(0.5).fillna(0)


def _validation_top_tail(
    validation: pd.DataFrame,
    prediction: np.ndarray,
    top_n: int,
    technology_only: bool = False,
) -> tuple[float, float]:
    scored = validation.copy()
    scored["tail_prediction"] = prediction
    excesses: list[float] = []
    precisions: list[float] = []
    for _, group in scored.groupby("date"):
        eligible = group[group["eligible"].fillna(False) & group["future_return_20"].notna()]
        if technology_only:
            eligible = eligible[eligible["broad_sector"] == "technology"]
        if len(eligible) < max(10, top_n * 2):
            continue
        count = min(top_n, len(eligible))
        selected = eligible.nlargest(count, "tail_prediction")
        truth = set(eligible.nlargest(count, "v10_target_20")["symbol"])
        if technology_only:
            weights = pd.to_numeric(eligible["benchmark_weight"], errors="coerce").clip(lower=0)
            benchmark = float(np.average(eligible["future_return_20"], weights=weights)) if weights.sum() > 0 else float(eligible["future_return_20"].mean())
        else:
            weights = pd.to_numeric(eligible["benchmark_weight"], errors="coerce").clip(lower=0)
            benchmark = float(np.average(eligible["future_return_20"], weights=weights)) if weights.sum() > 0 else float(eligible["future_return_20"].mean())
        excesses.append(float(selected["future_return_20"].mean() - benchmark))
        precisions.append(len(set(selected["symbol"]) & truth) / count)
    return (
        float(np.mean(excesses)) if excesses else float("nan"),
        float(np.mean(precisions)) if precisions else float("nan"),
    )


def _nested_tail_validation(
    dataset: pd.DataFrame, test_year: int, settings: V11Settings
) -> tuple[dict[int, dict[str, float | bool]], bool, bool]:
    earliest = test_year - settings.training_window_years
    diagnostics: dict[int, dict[str, float | bool]] = {}
    for year in range(test_year - settings.validation_years, test_year):
        train = mature_training(dataset, year, "v10_target_20", "label_end_date_20", earliest)
        validation = dataset[
            dataset["eligible"].fillna(False)
            & (pd.to_datetime(dataset["date"]).dt.year == year)
            & dataset["future_return_20"].notna()
        ].copy()
        validation_dates = (
            validation["date"].drop_duplicates().sort_values().iloc[:: settings.rebalance_every]
        )
        validation = validation[validation["date"].isin(validation_dates)].copy()
        if train.empty or validation.empty:
            diagnostics[year] = {
                "top30_excess": float("nan"),
                "top30_precision": float("nan"),
                "technology_top_excess": float("nan"),
                "technology_top_precision": float("nan"),
                "technology_sample_valid": False,
            }
            continue
        model = TailRanker().fit(train)
        prediction = model.predict(validation)
        top_excess, top_precision = _validation_top_tail(
            validation, prediction, settings.active_top_n
        )
        tech_rows = train[train["broad_sector"] == "technology"]
        tech_valid = (
            len(tech_rows) >= settings.minimum_technology_rows
            and tech_rows["date"].nunique() >= settings.minimum_technology_dates
        )
        tech_excess, tech_precision = _validation_top_tail(
            validation, prediction, settings.technology_top_n, technology_only=True
        )
        diagnostics[year] = {
            "top30_excess": top_excess,
            "top30_precision": top_precision,
            "technology_top_excess": tech_excess,
            "technology_top_precision": tech_precision,
            "technology_sample_valid": tech_valid,
        }
    global_gate = len(diagnostics) == settings.validation_years and all(
        np.isfinite(float(value["top30_excess"])) and float(value["top30_excess"]) > 0
        for value in diagnostics.values()
    )
    technology_gate = len(diagnostics) == settings.validation_years and all(
        bool(value["technology_sample_valid"])
        and np.isfinite(float(value["technology_top_excess"]))
        and float(value["technology_top_excess"]) > 0
        for value in diagnostics.values()
    )
    return diagnostics, global_gate, technology_gate


def fit_v11_models(
    dataset: pd.DataFrame, test_year: int, settings: V11Settings | None = None
) -> V11Models:
    settings = settings or V11Settings()
    earliest = test_year - settings.training_window_years
    train = mature_training(
        dataset, test_year, "v10_target_20", "label_end_date_20", earliest
    )
    if train.empty:
        raise RuntimeError(f"{test_year}没有足够的V11成熟训练数据")
    diagnostics, global_gate, technology_gate = _nested_tail_validation(
        dataset, test_year, settings
    )
    return V11Models(
        tail=TailRanker().fit(train),
        v10=fit_v10_models(dataset, test_year, V10Settings()),
        global_gate=global_gate,
        technology_gate=technology_gate,
        validation_diagnostics=diagnostics,
        training_rows=len(train),
        training_end=pd.to_datetime(train["label_end_date_20"]).max(),
    )


def score_v11(
    current: pd.DataFrame,
    models: V11Models,
    v5_models: V5Models,
    v4_specs: list[FactorSpec],
    settings: V11Settings | None = None,
) -> pd.DataFrame:
    settings = settings or V11Settings()
    scored = score_v10(current, models.v10, v5_models, v4_specs, V10Settings())
    scored["tail_score"] = _centered_rank(models.tail.predict(scored), scored.index)
    scored["v11_score"] = (
        settings.tail_share * scored["tail_score"]
        + settings.v10_global_share * scored["global_model_score"]
    )
    scored["global_gate"] = models.global_gate
    scored["technology_gate"] = models.technology_gate
    scored["score"] = scored["v11_score"]
    return scored
