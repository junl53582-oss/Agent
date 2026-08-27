from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import V4Settings
from .features import FACTOR_COLUMNS


@dataclass(frozen=True)
class FactorSpec:
    factor: str
    selected: bool
    direction: int
    weight: float
    mean_rank_ic: float
    direction_consistency: float
    training_years: tuple[int, ...]
    ic_days: int


def _daily_rank_ic(data: pd.DataFrame, factor: str) -> pd.DataFrame:
    rows = []
    for date, group in data.groupby("date", sort=True):
        valid = group[[factor, "label_5"]].dropna()
        if len(valid) < 20 or valid[factor].nunique() < 2:
            continue
        value = valid[factor].corr(valid["label_5"], method="spearman")
        if pd.notna(value):
            rows.append({"date": date, "rank_ic": float(value)})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["year"] = pd.to_datetime(result["date"]).dt.year
    return result


def learn_factor_specs(
    dataset: pd.DataFrame, test_year: int, settings: V4Settings | None = None
) -> tuple[list[FactorSpec], pd.DataFrame]:
    settings = settings or V4Settings()
    cutoff = pd.Timestamp(test_year, 1, 1)
    earliest_year = test_year - settings.training_year_window
    train = dataset[
        dataset["eligible"]
        & dataset["label_5"].notna()
        & (pd.to_datetime(dataset["label_end_date_5"]) < cutoff)
        & (pd.to_datetime(dataset["date"]) < cutoff)
        & (pd.to_datetime(dataset["date"]).dt.year >= earliest_year)
    ]
    specs: list[FactorSpec] = []
    diagnostics = []
    for factor in FACTOR_COLUMNS:
        daily = _daily_rank_ic(train, factor)
        annual = (
            daily.groupby("year")["rank_ic"].agg(["mean", "count"]).reset_index()
            if not daily.empty
            else pd.DataFrame(columns=["year", "mean", "count"])
        )
        valid_annual = annual[annual["count"] >= settings.minimum_ic_days_per_year]
        mean_ic = float(daily["rank_ic"].mean()) if not daily.empty else 0.0
        direction = int(np.sign(mean_ic))
        consistency = (
            float((np.sign(valid_annual["mean"]) == direction).mean())
            if direction and not valid_annual.empty
            else 0.0
        )
        selected = (
            len(valid_annual) >= settings.minimum_training_years
            and abs(mean_ic) >= settings.minimum_absolute_mean_rank_ic
            and consistency >= settings.minimum_direction_consistency
        )
        spec = FactorSpec(
            factor=factor,
            selected=selected,
            direction=direction if selected else 0,
            weight=abs(mean_ic) if selected else 0.0,
            mean_rank_ic=mean_ic,
            direction_consistency=consistency,
            training_years=tuple(int(year) for year in valid_annual["year"]),
            ic_days=len(daily),
        )
        specs.append(spec)
        for row in annual.itertuples(index=False):
            diagnostics.append(
                {
                    "test_year": test_year,
                    "factor": factor,
                    "training_year": int(row.year),
                    "annual_rank_ic": float(row.mean),
                    "ic_days": int(row.count),
                    "valid_year": int(row.count) >= settings.minimum_ic_days_per_year,
                }
            )
    total = sum(spec.weight for spec in specs)
    if total > 0:
        specs = [
            FactorSpec(**{**spec.__dict__, "weight": spec.weight / total}) for spec in specs
        ]
    return specs, pd.DataFrame(diagnostics)


def score_with_specs(data: pd.DataFrame, specs: list[FactorSpec]) -> pd.Series:
    score = pd.Series(0.0, index=data.index)
    for spec in specs:
        if spec.selected:
            score += spec.weight * spec.direction * data[spec.factor]
    return score
