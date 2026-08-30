from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SnapshotProof:
    observation_id: str
    observed_at: str
    snapshot_hash: str
    source_hash: str

    @property
    def timestamp(self) -> pd.Timestamp:
        value = pd.Timestamp(self.observed_at)
        if value.tzinfo is None:
            raise ValueError("snapshot proof timestamp must be timezone-aware")
        return value

    def validate(self) -> None:
        if not self.observation_id:
            raise ValueError("snapshot proof observation_id is required")
        if len(self.snapshot_hash) != 64 or len(self.source_hash) != 64:
            raise ValueError("snapshot proof requires SHA-256 snapshot/source hashes")
        self.timestamp


def _prove_order(*proofs: SnapshotProof) -> None:
    for proof in proofs:
        proof.validate()
    times = [proof.timestamp for proof in proofs]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("revision proof requires strict T-2 < T-1 < T observation order")


def no_revision_panel(current: pd.DataFrame, current_proof: SnapshotProof) -> pd.DataFrame:
    current_proof.validate()
    output = current.copy()
    output["expectation_level"] = pd.to_numeric(output["forecast_eps_1"], errors="coerce")
    for name in (
        "expectation_revision_abs", "expectation_revision_pct", "expectation_revision_rank",
        "positive_revision_flag", "negative_revision_flag", "revision_direction",
        "revision_persistence", "relative_revision_vs_industry",
    ):
        output[name] = pd.NA
    output["revision_available"] = False
    output["revision_status"] = "INSUFFICIENT_PROSPECTIVE_SNAPSHOTS"
    output["current_observation_id"] = current_proof.observation_id
    output["current_observed_at"] = current_proof.timestamp.isoformat()
    output["current_snapshot_hash"] = current_proof.snapshot_hash
    output["current_source_hash"] = current_proof.source_hash
    return output


def build_revision_panel(
    previous: pd.DataFrame | None,
    current: pd.DataFrame,
    *,
    previous_proof: SnapshotProof | None,
    current_proof: SnapshotProof,
    earlier_revision: pd.DataFrame | None = None,
    earlier_proof: SnapshotProof | None = None,
) -> pd.DataFrame:
    if previous is None or previous.empty or previous_proof is None:
        if earlier_revision is not None or earlier_proof is not None:
            raise ValueError("earlier revision cannot exist without the previous snapshot")
        return no_revision_panel(current, current_proof)
    _prove_order(previous_proof, current_proof)
    if (earlier_revision is None) != (earlier_proof is None):
        raise ValueError("earlier revision data and proof must be supplied together")
    if earlier_proof is not None:
        _prove_order(earlier_proof, previous_proof, current_proof)
    required = {"symbol", "forecast_year_1", "forecast_eps_1", "industry"}
    for name, frame in (("previous", previous), ("current", current)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} revision snapshot missing {sorted(missing)}")
        if frame["symbol"].duplicated().any():
            raise ValueError(f"{name} revision snapshot has duplicate symbols")
    prior = previous[["symbol", "forecast_year_1", "forecast_eps_1"]].rename(
        columns={
            "forecast_year_1": "prior_forecast_year_1",
            "forecast_eps_1": "prior_expectation_level",
        }
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
    nonzero = comparable & output["prior_expectation_level"].ne(0)
    output["expectation_revision_pct"] = (
        output["expectation_revision_abs"] / output["prior_expectation_level"].abs()
    ).where(nonzero)
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
    if earlier_revision is not None:
        required_earlier = {"symbol", "revision_direction"}
        if required_earlier - set(earlier_revision.columns) or earlier_revision["symbol"].duplicated().any():
            raise ValueError("earlier revision proof/data schema is invalid")
        prior_direction = earlier_revision[["symbol", "revision_direction"]].rename(
            columns={"revision_direction": "prior_revision_direction"}
        )
        output = output.merge(prior_direction, on="symbol", how="left", validate="one_to_one")
        available = output["revision_direction"].notna() & output["prior_revision_direction"].notna()
        output["revision_persistence"] = (
            output["revision_direction"].eq(output["prior_revision_direction"])
            & output["revision_direction"].ne(0)
        ).where(available).astype("boolean")
        output["earlier_observation_id"] = earlier_proof.observation_id
        output["earlier_observed_at"] = earlier_proof.timestamp.isoformat()
        output["earlier_snapshot_hash"] = earlier_proof.snapshot_hash
        output["earlier_source_hash"] = earlier_proof.source_hash
    output["revision_available"] = output["expectation_revision_abs"].notna()
    output["revision_status"] = np.where(output["revision_available"], "AVAILABLE", "NOT_COMPARABLE")
    for prefix, proof in (("previous", previous_proof), ("current", current_proof)):
        output[f"{prefix}_observation_id"] = proof.observation_id
        output[f"{prefix}_observed_at"] = proof.timestamp.isoformat()
        output[f"{prefix}_snapshot_hash"] = proof.snapshot_hash
        output[f"{prefix}_source_hash"] = proof.source_hash
    return output


def build_industry_revision(
    panel: pd.DataFrame,
    *,
    current_proof: SnapshotProof,
    previous_industry: pd.DataFrame | None = None,
    previous_proof: SnapshotProof | None = None,
) -> pd.DataFrame:
    current_proof.validate()
    if (previous_industry is None) != (previous_proof is None):
        raise ValueError("previous industry data and proof must be supplied together")
    if previous_proof is not None:
        _prove_order(previous_proof, current_proof)
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
        lambda values: values.dropna().gt(0).mean() if values.notna().any() else np.nan
    ).to_numpy()
    output["industry_negative_revision_ratio"] = grouped["expectation_revision_pct"].apply(
        lambda values: values.dropna().lt(0).mean() if values.notna().any() else np.nan
    ).to_numpy()
    output["industry_revision_rank"] = output["industry_revision_mean"].rank(
        pct=True, method="average", na_option="keep"
    )
    output["industry_revision_acceleration"] = pd.NA
    if previous_industry is not None:
        prior = previous_industry[["industry", "industry_revision_mean"]].rename(
            columns={"industry_revision_mean": "prior_industry_revision_mean"}
        )
        output = output.merge(prior, on="industry", how="left", validate="one_to_one")
        output["industry_revision_acceleration"] = (
            output["industry_revision_mean"] - output["prior_industry_revision_mean"]
        )
        output["previous_industry_observation_id"] = previous_proof.observation_id
        output["previous_industry_observed_at"] = previous_proof.timestamp.isoformat()
        output["previous_industry_snapshot_hash"] = previous_proof.snapshot_hash
        output["previous_industry_source_hash"] = previous_proof.source_hash
    output["current_observation_id"] = current_proof.observation_id
    output["current_observed_at"] = current_proof.timestamp.isoformat()
    output["current_snapshot_hash"] = current_proof.snapshot_hash
    output["current_source_hash"] = current_proof.source_hash
    return output
