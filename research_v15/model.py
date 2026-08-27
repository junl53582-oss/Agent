from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research_v10.model import V10Models, fit_v10_models, score_v10
from research_v10.research_config import V10Settings
from research_v12.model import mature_embargoed_training
from research_v13.config import V13Settings
from research_v13.model import TwoStageModel, confidence_lower_bound
from research_v14.model import _date_payoffs
from research_v4.stability import FactorSpec
from research_v4.config import V4Settings
from research_v4.stability import learn_factor_specs
from research_v5.models import V5Models, fit_v5_models

from .config import V15Settings
from .text_model import EventTextCorpus, MultiHorizonTextModel


def _sector_rank(frame: pd.DataFrame, values) -> pd.Series:
    raw = pd.Series(np.asarray(values, dtype=float), index=frame.index)
    ranked = raw.groupby([frame["date"], frame["broad_sector"]]).rank(
        pct=True, method="average"
    )
    return ranked.groupby([frame["date"], frame["broad_sector"]]).transform(
        lambda group: group - group.mean()
    ).fillna(0.0)


def _validation_predictions(
    validation: pd.DataFrame,
    baseline: TwoStageModel,
    text_model: MultiHorizonTextModel,
    settings: V15Settings,
    baseline_parts: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    combined = pd.Series(0.0, index=validation.index)
    base = pd.Series(0.0, index=validation.index)
    for _, current in validation.groupby("date", sort=False):
        scored = _score_components(current, baseline, text_model, baseline_parts, settings)
        combined.loc[scored.index] = scored["v15_score"]
        base.loc[scored.index] = scored["v13_comparable_score"]
    return combined.reindex(validation.index).to_numpy(), base.reindex(validation.index).to_numpy()


def baseline_training_view(dataset, year, settings):
    boundary = pd.Timestamp(year, 1, 1) - pd.Timedelta(days=settings.embargo_calendar_days)
    mask = pd.to_datetime(dataset["date"]).lt(boundary)
    for column in ("label_end_date_5", "label_end_date_20"):
        mask &= pd.to_datetime(dataset[column]).lt(boundary)
    return dataset.loc[mask].copy()


def baseline_parts_for_year(dataset, year, settings, cache):
    if year not in cache:
        history = baseline_training_view(dataset, year, settings)
        print(f"V15 baseline fitting cutoff={year}", flush=True)
        v5 = fit_v5_models(history, year)
        v4, _ = learn_factor_specs(history, year, V4Settings())
        v10 = fit_v10_models(history, year, V10Settings())
        cache[year] = (v10, v5, v4)
    return cache[year]


def validation_view(dataset, validation_year, test_year, settings):
    boundary = pd.Timestamp(test_year, 1, 1) - pd.Timedelta(days=settings.embargo_calendar_days)
    return dataset[
        dataset["in_universe"].eq(True)
        & pd.to_datetime(dataset["date"]).dt.year.eq(validation_year)
        & np.isfinite(dataset["future_return_20"])
        & pd.to_datetime(dataset["label_end_date_20"]).lt(boundary)
    ].copy()


def raw_year_gate(available_years, mature_years, cap):
    required = min(cap, max(1, len(available_years)))
    return required, len(set(available_years) & set(mature_years)) >= required


@dataclass
class V15Models:
    text_model: MultiHorizonTextModel
    baseline_model: TwoStageModel
    v10: V10Models
    global_gate: bool
    technology_gate: bool
    validation_diagnostics: dict[int, dict]
    payoff_lower_bound: float
    incremental_lower_bound: float
    technology_lower_bound: float
    technology_incremental_lower_bound: float
    training_events: int
    raw_event_years: list[int]


def _nested_validation(
    dataset: pd.DataFrame,
    corpus: EventTextCorpus,
    test_year: int,
    settings: V15Settings,
    baseline_cache: dict,
):
    earliest = test_year - settings.training_window_years
    diagnostics = {}
    pooled, incremental, tech_pool, tech_incremental = [], [], [], []
    for year in range(test_year - settings.validation_years, test_year):
        train = mature_embargoed_training(
            dataset, year, earliest, settings.embargo_calendar_days
        )
        validation = validation_view(dataset, year, test_year, settings)
        dates = validation["date"].drop_duplicates().sort_values().iloc[:: settings.rebalance_every]
        validation = validation[validation["date"].isin(dates)].copy()
        baseline = TwoStageModel().fit(train, V13Settings())
        text_model = MultiHorizonTextModel.fit(corpus, year, earliest, settings)
        parts = baseline_parts_for_year(dataset, year, settings, baseline_cache)
        prediction, baseline_prediction = _validation_predictions(
            validation, baseline, text_model, settings, parts
        )
        text_payoff, text_tech, precision = _date_payoffs(validation, prediction, settings)
        base_payoff, base_tech, _ = _date_payoffs(
            validation, baseline_prediction, settings
        )
        count = min(len(text_payoff), len(base_payoff))
        tech_count = min(len(text_tech), len(base_tech))
        year_incremental = (
            np.asarray(text_payoff[:count]) - np.asarray(base_payoff[:count])
            if count else np.asarray([])
        )
        year_tech_incremental = (
            np.asarray(text_tech[:tech_count]) - np.asarray(base_tech[:tech_count])
            if tech_count else np.asarray([])
        )
        pooled.extend(text_payoff[:count])
        incremental.extend(year_incremental.tolist())
        tech_pool.extend(text_tech[:tech_count])
        tech_incremental.extend(year_tech_incremental.tolist())
        tech_train = train[train["broad_sector"] == "technology"]
        raw_years = text_model.event_years
        required_years, year_gate = raw_year_gate(
            text_model.available_event_years, raw_years, settings.minimum_event_years_cap
        )
        diagnostics[year] = {
            "text_payoff_mean": float(np.mean(text_payoff)) if text_payoff else float("nan"),
            "baseline_payoff_mean": float(np.mean(base_payoff)) if base_payoff else float("nan"),
            "incremental_mean": float(year_incremental.mean()) if count else float("nan"),
            "technology_payoff_mean": float(np.mean(text_tech)) if text_tech else float("nan"),
            "technology_incremental_mean": float(year_tech_incremental.mean()) if tech_count else float("nan"),
            "top30_precision": float(np.mean(precision)) if precision else float("nan"),
            "training_events": text_model.training_events,
            "raw_event_years": len(raw_years),
            "available_raw_event_years": len(text_model.available_event_years),
            "raw_event_year_values": ",".join(map(str, raw_years)),
            "validation_label_end_max": str(pd.to_datetime(validation["label_end_date_20"]).max()),
            "validation_cutoff_exclusive": str(pd.Timestamp(test_year, 1, 1) - pd.Timedelta(days=settings.embargo_calendar_days)),
            "required_raw_event_years": required_years,
            "raw_event_year_gate": year_gate,
            "technology_sample_valid": len(tech_train) >= settings.minimum_technology_rows and tech_train["date"].nunique() >= settings.minimum_technology_dates,
        }
    payoff_lower = confidence_lower_bound(pooled, settings.confidence_z)
    incremental_lower = confidence_lower_bound(incremental, settings.confidence_z)
    tech_lower = confidence_lower_bound(tech_pool, settings.confidence_z)
    tech_incremental_lower = confidence_lower_bound(tech_incremental, settings.confidence_z)
    floor = all(
        np.isfinite(values["text_payoff_mean"])
        and values["text_payoff_mean"] >= settings.validation_year_floor
        and values["raw_event_year_gate"]
        for values in diagnostics.values()
    )
    tech_floor = all(
        values["technology_sample_valid"]
        and np.isfinite(values["technology_payoff_mean"])
        and values["technology_payoff_mean"] >= settings.validation_year_floor
        and values["raw_event_year_gate"]
        for values in diagnostics.values()
    )
    global_gate = bool(
        np.isfinite(payoff_lower) and payoff_lower > 0
        and np.isfinite(incremental_lower) and incremental_lower > 0
        and floor
    )
    technology_gate = bool(
        np.isfinite(tech_lower) and tech_lower > 0
        and np.isfinite(tech_incremental_lower) and tech_incremental_lower > 0
        and tech_floor
    )
    return (
        diagnostics, global_gate, technology_gate, payoff_lower,
        incremental_lower, tech_lower, tech_incremental_lower,
    )


def fit_v15_models(
    dataset: pd.DataFrame,
    corpus: EventTextCorpus,
    test_year: int,
    settings: V15Settings | None = None,
    baseline_cache: dict | None = None,
) -> V15Models:
    settings = settings or V15Settings()
    earliest = test_year - settings.training_window_years
    train = mature_embargoed_training(
        dataset, test_year, earliest, settings.embargo_calendar_days
    )
    baseline_cache = {} if baseline_cache is None else baseline_cache
    diagnostics = _nested_validation(dataset, corpus, test_year, settings, baseline_cache)
    text_model = MultiHorizonTextModel.fit(corpus, test_year, earliest, settings)
    return V15Models(
        text_model=text_model,
        baseline_model=TwoStageModel().fit(train, V13Settings()),
        v10=baseline_parts_for_year(dataset, test_year, settings, baseline_cache)[0],
        global_gate=diagnostics[1],
        technology_gate=diagnostics[2],
        validation_diagnostics=diagnostics[0],
        payoff_lower_bound=diagnostics[3],
        incremental_lower_bound=diagnostics[4],
        technology_lower_bound=diagnostics[5],
        technology_incremental_lower_bound=diagnostics[6],
        training_events=text_model.training_events,
        raw_event_years=text_model.event_years,
    )


def score_v15(
    current: pd.DataFrame,
    models: V15Models,
    v5_models: V5Models,
    v4_specs: list[FactorSpec],
    settings: V15Settings | None = None,
) -> pd.DataFrame:
    settings = settings or V15Settings()
    return _score_components(
        current, models.baseline_model, models.text_model,
        (models.v10, v5_models, v4_specs), settings,
    )


def _score_components(current, baseline_model, text_model, baseline_parts, settings):
    v10, v5_models, v4_specs = baseline_parts
    scored = score_v10(current, v10, v5_models, v4_specs, V10Settings())
    baseline_probability, baseline_magnitude = baseline_model.predict_components(scored)
    baseline_raw = baseline_probability * np.clip(baseline_magnitude, 0.0, 0.10)
    scored["v13_baseline_score"] = _sector_rank(scored, baseline_raw)
    v13_settings = V13Settings()
    scored["v13_comparable_score"] = (
        v13_settings.two_stage_share * scored["v13_baseline_score"]
        + v13_settings.v10_global_share * scored["global_model_score"]
    )
    recent = text_model.recent_scores(scored, settings).set_index("symbol")
    scored["text_event_score"] = scored["symbol"].map(recent["text_score"]).fillna(0.0)
    scored["recent_text_events"] = scored["symbol"].map(recent["text_events"]).fillna(0).astype(int)
    scored["v15_score"] = (
        settings.baseline_share * scored["v13_comparable_score"]
        + settings.text_share * scored["text_event_score"]
    )
    scored["score"] = scored["v15_score"]
    return scored
