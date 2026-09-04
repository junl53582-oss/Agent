from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
SYMBOL_PATTERN = re.compile(r"\d{6}")
ANNOUNCEMENT_FIELDS = (
    "symbol",
    "announcement_id",
    "published_at_source",
    "effective_trading_date",
    "document_sha256",
    "text_sha256",
    "revision_of_announcement_id",
    "source_uri",
)
ESTIMATE_FIELDS = (
    "symbol",
    "estimate_id",
    "institution_id",
    "published_at",
    "forecast_period",
    "metric",
    "estimate_value",
    "currency",
    "revision_status",
    "supersedes_estimate_id",
    "raw_record_sha256",
)
ACTUAL_FIELDS = (
    "symbol",
    "actual_id",
    "report_period",
    "metric",
    "actual_value",
    "published_at",
    "revision_status",
    "supersedes_actual_id",
    "raw_record_sha256",
)


def _timestamps(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(frame[column], format="mixed", errors="coerce", utc=True)


def _hashes_valid(values: pd.Series) -> bool:
    return bool(values.fillna("").astype(str).str.lower().map(lambda x: bool(HASH_PATTERN.fullmatch(x))).all())


def _symbols_valid(values: pd.Series) -> bool:
    return bool(values.fillna("").astype(str).map(lambda x: bool(SYMBOL_PATTERN.fullmatch(x))).all())


def _coverage(frame: pd.DataFrame, timestamps: pd.Series) -> dict[str, int]:
    valid = timestamps.dropna()
    return {
        "rows": len(frame),
        "symbols": int(frame["symbol"].nunique()) if "symbol" in frame else 0,
        "years": int(valid.dt.year.nunique()) if not valid.empty else 0,
        "distinct_months": int(valid.dt.strftime("%Y-%m").nunique()) if not valid.empty else 0,
    }


def _missing(frame: pd.DataFrame, required: tuple[str, ...]) -> list[str]:
    return sorted(set(required) - set(frame.columns))


def validate_announcement_documents(frame: pd.DataFrame, protocol: dict[str, Any]) -> dict[str, Any]:
    missing = _missing(frame, ANNOUNCEMENT_FIELDS)
    if missing:
        return {"passed": False, "missing_columns": missing, "coverage": {}}
    published = _timestamps(frame, "published_at_source")
    effective = pd.to_datetime(frame["effective_trading_date"], errors="coerce")
    coverage = _coverage(frame, published)
    requirements = protocol["required_imports"]["announcement_documents"]
    checks = {
        "symbols_valid": _symbols_valid(frame["symbol"]),
        "announcement_ids_unique": not frame["announcement_id"].astype(str).duplicated().any(),
        "published_timestamps_valid": not published.isna().any(),
        "effective_dates_valid": not effective.isna().any(),
        "not_same_day_when_publication_time_is_date_only": bool(
            (
                effective.dt.normalize()
                > published.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).dt.normalize()
            ).all()
        ),
        "document_hashes_valid": _hashes_valid(frame["document_sha256"]),
        "text_hashes_valid": _hashes_valid(frame["text_sha256"]),
        "source_uri_present": bool(frame["source_uri"].fillna("").astype(str).str.startswith("https://").all()),
        "document_coverage": coverage["rows"] >= requirements["minimum_documents"],
        "symbol_coverage": coverage["symbols"] >= requirements["minimum_symbols"],
        "year_coverage": coverage["years"] >= requirements["minimum_years"],
    }
    return {"passed": all(checks.values()), "missing_columns": [], "coverage": coverage, "checks": checks}


def validate_analyst_estimates(frame: pd.DataFrame, protocol: dict[str, Any]) -> dict[str, Any]:
    missing = _missing(frame, ESTIMATE_FIELDS)
    if missing:
        return {"passed": False, "missing_columns": missing, "coverage": {}}
    published = _timestamps(frame, "published_at")
    coverage = _coverage(frame, published)
    requirements = protocol["required_imports"]["analyst_estimates"]
    values = pd.to_numeric(frame["estimate_value"], errors="coerce")
    states = set(frame["revision_status"].dropna().astype(str).str.upper())
    revised = frame["revision_status"].astype(str).str.upper().eq("REVISED")
    checks = {
        "symbols_valid": _symbols_valid(frame["symbol"]),
        "estimate_ids_unique": not frame["estimate_id"].astype(str).duplicated().any(),
        "institution_ids_present": bool(frame["institution_id"].fillna("").astype(str).str.len().gt(0).all()),
        "published_timestamps_valid": not published.isna().any(),
        "forecast_periods_valid": not pd.to_datetime(frame["forecast_period"], errors="coerce").isna().any(),
        "eps_metric_present": frame["metric"].astype(str).str.upper().eq("EPS").any(),
        "values_numeric": bool(values.notna().all() and np.isfinite(values).all()),
        "revision_states_valid": states <= {"ORIGINAL", "REVISED", "WITHDRAWN"},
        "revisions_linked": bool(frame.loc[revised, "supersedes_estimate_id"].notna().all()),
        "raw_hashes_valid": _hashes_valid(frame["raw_record_sha256"]),
        "symbol_coverage": coverage["symbols"] >= requirements["minimum_symbols"],
        "year_coverage": coverage["years"] >= requirements["minimum_years"],
        "month_coverage": coverage["distinct_months"] >= requirements["minimum_distinct_months"],
    }
    return {"passed": all(checks.values()), "missing_columns": [], "coverage": coverage, "checks": checks}


def validate_actual_versions(frame: pd.DataFrame, protocol: dict[str, Any]) -> dict[str, Any]:
    missing = _missing(frame, ACTUAL_FIELDS)
    if missing:
        return {"passed": False, "missing_columns": missing, "coverage": {}}
    published = _timestamps(frame, "published_at")
    coverage = _coverage(frame, published)
    requirements = protocol["required_imports"]["actual_versions"]
    values = pd.to_numeric(frame["actual_value"], errors="coerce")
    states = set(frame["revision_status"].dropna().astype(str).str.upper())
    revised = frame["revision_status"].astype(str).str.upper().eq("REVISED")
    checks = {
        "symbols_valid": _symbols_valid(frame["symbol"]),
        "actual_ids_unique": not frame["actual_id"].astype(str).duplicated().any(),
        "report_periods_valid": not pd.to_datetime(frame["report_period"], errors="coerce").isna().any(),
        "published_timestamps_valid": not published.isna().any(),
        "values_numeric": bool(values.notna().all() and np.isfinite(values).all()),
        "revision_states_valid": states <= {"PRELIMINARY", "ORIGINAL", "REVISED"},
        "revisions_linked": bool(frame.loc[revised, "supersedes_actual_id"].notna().all()),
        "raw_hashes_valid": _hashes_valid(frame["raw_record_sha256"]),
        "symbol_coverage": coverage["symbols"] >= requirements["minimum_symbols"],
        "year_coverage": coverage["years"] >= requirements["minimum_years"],
    }
    return {"passed": all(checks.values()), "missing_columns": [], "coverage": coverage, "checks": checks}


def build_earnings_surprise(estimates: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Build PIT surprise using each institution's latest estimate strictly before each actual."""
    estimate = estimates.copy()
    actual = actuals.copy()
    estimate["published_at"] = _timestamps(estimate, "published_at")
    actual["published_at"] = _timestamps(actual, "published_at")
    estimate["estimate_value"] = pd.to_numeric(estimate["estimate_value"], errors="raise")
    actual["actual_value"] = pd.to_numeric(actual["actual_value"], errors="raise")
    output: list[dict[str, Any]] = []
    for row in actual.sort_values("published_at").itertuples(index=False):
        eligible = estimate[
            estimate["symbol"].astype(str).eq(str(row.symbol))
            & estimate["forecast_period"].astype(str).eq(str(row.report_period))
            & estimate["metric"].astype(str).str.upper().eq(str(row.metric).upper())
            & estimate["published_at"].lt(row.published_at)
            & ~estimate["revision_status"].astype(str).str.upper().eq("WITHDRAWN")
        ].sort_values(["institution_id", "published_at", "estimate_id"])
        latest = eligible.groupby("institution_id", as_index=False).tail(1)
        if latest.empty:
            continue
        consensus = float(latest["estimate_value"].mean())
        actual_value = float(row.actual_value)
        output.append(
            {
                "symbol": str(row.symbol),
                "actual_id": str(row.actual_id),
                "report_period": str(row.report_period),
                "metric": str(row.metric),
                "actual_published_at": row.published_at.isoformat(),
                "last_estimate_published_at": latest["published_at"].max().isoformat(),
                "estimate_count": len(latest),
                "consensus": consensus,
                "actual_value": actual_value,
                "surprise": actual_value - consensus,
                "scaled_surprise": (actual_value - consensus) / (abs(consensus) + 0.05),
                "strictly_pre_release": bool(latest["published_at"].lt(row.published_at).all()),
            }
        )
    return pd.DataFrame(output)
