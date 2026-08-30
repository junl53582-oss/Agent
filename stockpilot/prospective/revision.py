from __future__ import annotations

import numpy as np
import pandas as pd


REVISION_COLUMNS = [
    "expectation_level",
    "expectation_revision_abs",
    "expectation_revision_pct",
    "expectation_revision_rank",
    "positive_revision_flag",
    "negative_revision_flag",
    "revision_direction",
    "revision_persistence",
    "relative_revision_vs_industry",
]


def no_revision_panel(current: pd.DataFrame) -> pd.DataFrame:
    output = current.copy()
    output["expectation_level"] = pd.to_numeric(output["forecast_eps_1"], errors="coerce")
    for column in REVISION_COLUMNS[1:]:
        output[column] = pd.NA
    output["revision_available"] = False
    output["revision_status"] = "INSUFFICIENT_PROSPECTIVE_SNAPSHOTS"
    return output


def build_revision_panel(
    previous: pd.DataFrame | None,
    current: pd.DataFrame,
    *,
    previous_observed_at: str | pd.Timestamp | None,
    current_observed_at: str | pd.Timestamp,
    earlier_revision: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if previous is None or previous.empty or previous_observed_at is None:
        return no_revision_panel(current)
    prior_time = pd.Timestamp(previous_observed_at)
    current_time = pd.Timestamp(current_observed_at)
    if prior_time.tzinfo is None or current_time.tzinfo is None:
        raise ValueError("revision observation timestamps must be timezone-aware")
    if current_time <= prior_time:
        raise ValueError("revision requires t1.observed_at > t0.observed_at")
    required = {"symbol", "forecast_year_1", "forecast_eps_1", "industry"}
    for name, frame in (("previous", previous), ("current", current)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} revision snapshot missing {sorted(missing)}")
        if frame["symbol"].duplicated().any():
            raise ValueError(f"{name} revision snapshot has duplicate symbols")
    prior = previous[["symbol", "forecast_year_1", "forecast_eps_1"]].rename(
        columns={"forecast_year_1": "prior_forecast_year_1", "forecast_eps_1": "prior_expectation_level"}
    )
    output = current.merge(prior, on="symbol", how="left", validate="one_to_one")
    output["expectation_level"] = pd.to_numeric(output["forecast_eps_1"], errors="coerce")
    output["prior_expectation_level"] = pd.to_numeric(output["prior_expectation_level"], errors="coerce")
    same_year = output["forecast_year_1"].astype("string").eq(
        output["prior_forecast_year_1"].astype("string")
    )
    comparable = same_year & output["expectation_level"].notna() & output["prior_expectation_level"].notna()
    output["expectation_revision_abs"] = (
        output["expectation_level"] - output["prior_expectation_level"]
    ).where(comparable)
    nonzero_prior = comparable & output["prior_expectation_level"].ne(0)
    output["expectation_revision_pct"] = (
        output["expectation_revision_abs"] / output["prior_expectation_level"].abs()
    ).where(nonzero_prior)
    output["expectation_revision_rank"] = output["expectation_revision_pct"].rank(
        pct=True, method="average", na_option="keep"
    )
    revision = output["expectation_revision_abs"]
    output["positive_revision_flag"] = revision.gt(0).where(revision.notna()).astype("boolean")
    output["negative_revision_flag"] = revision.lt(0).where(revision.notna()).astype("boolean")
    output["revision_direction"] = pd.Series(
        np.select([revision.gt(0), revision.lt(0)], [1, -1], default=0), index=output.index
    ).where(revision.notna()).astype("Int64")
    industry_mean = output.groupby("industry", dropna=False)["expectation_revision_pct"].transform("mean")
    output["relative_revision_vs_industry"] = output["expectation_revision_pct"] - industry_mean
    output["revision_persistence"] = pd.NA
    if earlier_revision is not None and not earlier_revision.empty:
        prior_direction = earlier_revision[["symbol", "revision_direction"]].rename(
            columns={"revision_direction": "prior_revision_direction"}
        )
        output = output.merge(prior_direction, on="symbol", how="left", validate="one_to_one")
        output["revision_persistence"] = (
            output["revision_direction"].eq(output["prior_revision_direction"])
            & output["revision_direction"].ne(0)
        ).where(output["revision_direction"].notna() & output["prior_revision_direction"].notna()).astype("boolean")
    output["revision_available"] = output["expectation_revision_abs"].notna()
    output["revision_status"] = np.where(output["revision_available"], "AVAILABLE", "NOT_COMPARABLE")
    output["previous_observed_at"] = prior_time.isoformat()
    output["current_observed_at"] = current_time.isoformat()
    return output


def build_industry_revision(panel: pd.DataFrame, previous_industry: pd.DataFrame | None = None) -> pd.DataFrame:
    required = {"industry", "symbol", "expectation_level", "expectation_revision_pct"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"industry revision input missing {sorted(missing)}")
    grouped = panel.groupby("industry", dropna=False)
    output = grouped.agg(
        industry_expectation_level=("expectation_level", "median"),
        industry_revision_mean=("expectation_revision_pct", "mean"),
        industry_revision_median=("expectation_revision_pct", "median"),
        industry_revision_dispersion=("expectation_revision_pct", "std"),
        industry_revision_breadth=("expectation_revision_pct", "count"),
        industry_members=("symbol", "nunique"),
    ).reset_index()
    output["industry_positive_revision_ratio"] = grouped["expectation_revision_pct"].apply(
        lambda value: value.dropna().gt(0).mean() if value.notna().any() else np.nan
    ).to_numpy()
    output["industry_negative_revision_ratio"] = grouped["expectation_revision_pct"].apply(
        lambda value: value.dropna().lt(0).mean() if value.notna().any() else np.nan
    ).to_numpy()
    output["industry_revision_rank"] = output["industry_revision_mean"].rank(
        pct=True, method="average", na_option="keep"
    )
    output["industry_revision_acceleration"] = pd.NA
    if previous_industry is not None and not previous_industry.empty:
        prior = previous_industry[["industry", "industry_revision_mean"]].rename(
            columns={"industry_revision_mean": "prior_industry_revision_mean"}
        )
        output = output.merge(prior, on="industry", how="left", validate="one_to_one")
        output["industry_revision_acceleration"] = (
            output["industry_revision_mean"] - output["prior_industry_revision_mean"]
        )
    return output
