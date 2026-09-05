from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_v10.features import V10_FEATURES

FORBIDDEN_COLUMN_TOKENS = (
    "future_return",
    "forward_return",
    "raw_label",
    "neutral_label",
    "label_",
    "target_",
    "v10_target",
    "v9_target",
)
SAFE_GEN2_META = ("date", "symbol", "broad_sector")
EXPECTED_JQ_FEATURES = 71


@dataclass(frozen=True)
class AuditSettings:
    jq_feature_store: Path
    gen2_panel: Path
    protocol_path: Path
    artifact_dir: Path
    redundancy_threshold: float = 0.85
    partial_redundancy_threshold: float = 0.65
    minimum_daily_cross_section: int = 10
    minimum_correlation_observations: int = 100
    continuous_minimum_dates: int = 120
    sparse_asof_minimum_dates: int = 30
    event_minimum_observations: int = 100
    shortlist_target: int = 20
    shortlist_maximum: int = 25
    shortlist_event_reserve: int = 4
    shortlist_snapshot_reserve: int = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): clean(inner) for key, inner in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(inner) for inner in item]
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating, float)):
            return None if not math.isfinite(float(item)) else float(item)
        if isinstance(item, (np.bool_, bool)):
            return bool(item)
        if isinstance(item, (pd.Timestamp, datetime)):
            return item.isoformat()
        return item

    return (json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    payload = _json_bytes(value)
    _write_bytes(path, payload)
    _write_bytes(path.with_name(f"{path.name}.sha256"), f"{hashlib.sha256(payload).hexdigest()}\n".encode())


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    _write_bytes(path, payload)
    _write_bytes(path.with_name(f"{path.name}.sha256"), f"{hashlib.sha256(payload).hexdigest()}\n".encode())


def _assert_no_label_columns(columns: list[str] | tuple[str, ...]) -> None:
    bad = [name for name in columns if any(token in name.lower() for token in FORBIDDEN_COLUMN_TOKENS)]
    if bad:
        raise RuntimeError(f"RETURN_LABEL_ACCESS_FORBIDDEN:{','.join(sorted(bad))}")


def _verify_hash(path: Path, expected: str, identity: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"INPUT_HASH_MISMATCH:{identity}:{expected}:{actual}")


def _load_protocol(settings: AuditSettings) -> dict[str, Any]:
    protocol = json.loads(settings.protocol_path.read_text(encoding="utf-8"))
    if protocol["protocol"] != "PREDICTION_V2_JQDATA_FEATURE_OVERLAP_AND_INFORMATION_AUDIT_V1":
        raise RuntimeError("PROTOCOL_ID_MISMATCH")
    frozen = protocol["frozen_rules"]
    expected = {
        "redundancy_threshold": settings.redundancy_threshold,
        "partial_redundancy_threshold": settings.partial_redundancy_threshold,
        "minimum_daily_cross_section": settings.minimum_daily_cross_section,
        "minimum_correlation_observations": settings.minimum_correlation_observations,
        "continuous_minimum_dates": settings.continuous_minimum_dates,
        "sparse_asof_minimum_dates": settings.sparse_asof_minimum_dates,
        "event_minimum_observations": settings.event_minimum_observations,
        "shortlist_target": settings.shortlist_target,
        "shortlist_maximum": settings.shortlist_maximum,
        "shortlist_event_reserve": settings.shortlist_event_reserve,
        "shortlist_snapshot_reserve": settings.shortlist_snapshot_reserve,
    }
    if frozen != expected:
        raise RuntimeError("FROZEN_RULES_MISMATCH")
    return protocol


def load_gen2_features_only(path: Path, expected_sha256: str) -> pd.DataFrame:
    """Read an explicit safe projection. Return-label values are never loaded."""
    _verify_hash(path, expected_sha256, "GEN2_PANEL")
    requested = [*SAFE_GEN2_META, *V10_FEATURES]
    _assert_no_label_columns(requested)
    frame = pd.read_parquet(path, columns=requested)
    if list(frame.columns) != requested:
        raise RuntimeError("GEN2_SAFE_PROJECTION_MISMATCH")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    if frame.duplicated(["date", "symbol"]).any():
        raise RuntimeError("GEN2_DUPLICATE_KEY")
    return frame


def load_jq_feature_store(
    directory: Path,
    expected_wide_sha256: str,
    expected_long_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide_path = directory / "features_wide.csv"
    long_path = directory / "features_long.csv"
    _verify_hash(wide_path, expected_wide_sha256, "JQDATA_FEATURES_WIDE")
    _verify_hash(long_path, expected_long_sha256, "JQDATA_FEATURES_LONG")
    wide = pd.read_csv(wide_path, dtype={"symbol": str})
    features = [name for name in wide.columns if name not in {"date", "symbol"}]
    _assert_no_label_columns(features)
    if len(features) != EXPECTED_JQ_FEATURES or any(not name.startswith("jq_") for name in features):
        raise RuntimeError(f"JQ_FEATURE_SCHEMA_MISMATCH:{len(features)}")
    wide["date"] = pd.to_datetime(wide["date"], errors="raise").dt.normalize()
    wide["symbol"] = wide["symbol"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    if wide.duplicated(["date", "symbol"]).any():
        raise RuntimeError("JQDATA_DUPLICATE_KEY")
    lineage_columns = [
        "feature_name",
        "source_dataset",
        "raw_sha256",
        "available_at",
        "feature_asof_date",
        "pit_status",
    ]
    _assert_no_label_columns(lineage_columns)
    long = pd.read_csv(long_path, usecols=lineage_columns)
    if set(long["feature_name"].unique()) != set(features):
        raise RuntimeError("JQDATA_LINEAGE_FEATURE_SET_MISMATCH")
    available = pd.to_datetime(long["available_at"], errors="coerce")
    asof = pd.to_datetime(long["feature_asof_date"], errors="coerce")
    if available.isna().any() or asof.isna().any() or (available < asof).any():
        raise RuntimeError("JQDATA_PIT_LINEAGE_INVALID")
    return wide.sort_values(["date", "symbol"]), long


def _family(feature: str) -> str:
    prefixes = (
        "jq_company_forecast_",
        "jq_earnings_event_",
        "jq_industry_",
        "jq_valuation_",
        "jq_hkhold_",
        "jq_quality_",
        "jq_growth_",
        "jq_risk_",
        "jq_momentum_",
        "jq_emotion_",
        "jq_financial_",
    )
    for prefix in prefixes:
        if feature.startswith(prefix):
            return prefix[3:-1]
    raise RuntimeError(f"UNKNOWN_JQ_FEATURE_FAMILY:{feature}")


def _role(family: str) -> str:
    if family == "industry":
        return "CONTROL_ONLY"
    if family in {"earnings_event", "financial"}:
        return "RESEARCH_ONLY_REVISION_RISK"
    if family == "company_forecast":
        return "EVENT_SPARSE"
    if family == "hkhold":
        return "SNAPSHOT_SPARSE"
    if family == "valuation":
        return "SPARSE_ASOF"
    return "CONTINUOUS"


def _lineage_map(long: pd.DataFrame) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for feature, part in long.groupby("feature_name", sort=True):
        datasets = sorted(part["source_dataset"].dropna().astype(str).unique())
        hashes = sorted(part["raw_sha256"].dropna().astype(str).unique())
        statuses = sorted(part["pit_status"].dropna().astype(str).unique())
        if len(datasets) != 1 or len(hashes) != 1:
            raise RuntimeError(f"AMBIGUOUS_FEATURE_LINEAGE:{feature}")
        result[str(feature)] = {
            "source_dataset": datasets[0],
            "raw_sha256": hashes[0],
            "pit_statuses": "|".join(statuses),
            "pit_admissible": all("PIT_SAFE" in value and "RESTATEMENT_RISK" not in value for value in statuses),
        }
    return result


def _robust_outlier_fraction(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    median = float(clean.median())
    mad = float((clean - median).abs().median())
    if mad <= 0:
        return 0.0
    robust_z = (clean - median).abs() / (1.4826 * mad)
    return float((robust_z > 10).mean())


def _temporal_metrics(wide: pd.DataFrame, feature: str) -> dict[str, Any]:
    values = pd.to_numeric(wide[feature], errors="coerce")
    active = wide.loc[values.notna(), ["date", "symbol"]].copy()
    observations = int(values.notna().sum())
    if active.empty:
        return {
            "observations": 0,
            "coverage": 0.0,
            "active_dates": 0,
            "date_min": None,
            "date_max": None,
            "median_symbols_per_active_date": 0.0,
            "monthly_coverage_cv": None,
            "maximum_monthly_median_shift_iqr": None,
            "robust_outlier_fraction": 0.0,
            "unique_values": 0,
        }
    per_date = active.groupby("date").size()
    month = wide["date"].dt.to_period("M")
    monthly_coverage = values.notna().groupby(month).mean()
    clean = values.dropna()
    iqr = float(clean.quantile(0.75) - clean.quantile(0.25))
    monthly_median = values.groupby(month).median().dropna()
    shift = float((monthly_median - float(clean.median())).abs().max() / iqr) if iqr > 0 else 0.0
    mean_coverage = float(monthly_coverage.mean())
    coverage_cv = (
        float(monthly_coverage.std(ddof=0) / mean_coverage)
        if len(monthly_coverage) > 1 and mean_coverage > 0
        else 0.0
    )
    return {
        "observations": observations,
        "coverage": observations / len(wide),
        "active_dates": int(active["date"].nunique()),
        "date_min": str(active["date"].min().date()),
        "date_max": str(active["date"].max().date()),
        "median_symbols_per_active_date": float(per_date.median()),
        "monthly_coverage_cv": coverage_cv,
        "maximum_monthly_median_shift_iqr": shift,
        "robust_outlier_fraction": _robust_outlier_fraction(values),
        "unique_values": int(clean.nunique()),
    }


def _correlations(
    joined: pd.DataFrame,
    jq_feature: str,
    minimum_cross_section: int,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    residual_pieces: list[pd.DataFrame] = []
    daily_rows: list[dict[str, Any]] = []
    columns = [jq_feature, *V10_FEATURES]
    for _, part in joined[["date", "broad_sector", *columns]].groupby("date", sort=True):
        part = part.dropna(subset=[jq_feature])
        if len(part) < minimum_cross_section or part[jq_feature].nunique() < 2:
            continue
        ranked = part[columns].rank(method="average", pct=True)
        varying = [
            feature
            for feature in V10_FEATURES
            if ranked[feature].notna().sum() >= 2 and ranked[feature].nunique() >= 2
        ]
        corr = pd.Series(index=V10_FEATURES, dtype=float)
        if varying:
            corr.loc[varying] = ranked[varying].corrwith(ranked[jq_feature])
        sector = part["broad_sector"].fillna("UNKNOWN")
        residual = ranked - ranked.groupby(sector.to_numpy()).transform("mean")
        residual_corr = pd.Series(index=V10_FEATURES, dtype=float)
        residual_varying = [
            feature
            for feature in varying
            if residual[feature].notna().sum() >= 2
            and residual[feature].nunique() >= 2
            and residual[jq_feature].nunique() >= 2
        ]
        if residual_varying:
            residual_corr.loc[residual_varying] = residual[residual_varying].corrwith(
                residual[jq_feature]
            )
        ranked["__jq__"] = ranked[jq_feature]
        pieces.append(ranked.drop(columns=jq_feature))
        residual["__jq__"] = residual[jq_feature]
        residual_pieces.append(residual.drop(columns=jq_feature))
        for feature in V10_FEATURES:
            daily_rows.append(
                {
                    "gen2_feature": feature,
                    "daily_rank_corr": corr.get(feature),
                    "daily_sector_conditioned_rank_corr": residual_corr.get(feature),
                }
            )
    columns_out = [
        "jq_feature",
        "gen2_feature",
        "correlation_dates",
        "pooled_date_rank_corr",
        "median_daily_rank_corr",
        "p90_abs_daily_rank_corr",
        "conditioned_pooled_date_rank_corr",
        "conditioned_median_daily_rank_corr",
    ]
    if not pieces:
        return pd.DataFrame(columns=columns_out)
    pooled = pd.concat(pieces, ignore_index=True)
    residual_pooled = pd.concat(residual_pieces, ignore_index=True)
    pooled_corr = pd.Series(index=V10_FEATURES, dtype=float)
    pooled_varying = [
        feature
        for feature in V10_FEATURES
        if pooled[feature].notna().sum() >= 2 and pooled[feature].nunique() >= 2
    ]
    if pooled_varying:
        pooled_corr.loc[pooled_varying] = pooled[pooled_varying].corrwith(pooled["__jq__"])
    residual_corr = pd.Series(index=V10_FEATURES, dtype=float)
    residual_varying = [
        feature
        for feature in V10_FEATURES
        if residual_pooled[feature].notna().sum() >= 2
        and residual_pooled[feature].nunique() >= 2
        and residual_pooled["__jq__"].nunique() >= 2
    ]
    if residual_varying:
        residual_corr.loc[residual_varying] = residual_pooled[residual_varying].corrwith(
            residual_pooled["__jq__"]
        )
    daily = pd.DataFrame(daily_rows)
    rows = []
    for feature in V10_FEATURES:
        part = daily[daily["gen2_feature"] == feature]
        unconditioned = pd.to_numeric(part["daily_rank_corr"], errors="coerce").dropna()
        conditioned = pd.to_numeric(
            part["daily_sector_conditioned_rank_corr"], errors="coerce"
        ).dropna()
        rows.append(
            {
                "jq_feature": jq_feature,
                "gen2_feature": feature,
                "correlation_dates": len(unconditioned),
                "pooled_date_rank_corr": pooled_corr.get(feature),
                "median_daily_rank_corr": unconditioned.median() if not unconditioned.empty else None,
                "p90_abs_daily_rank_corr": unconditioned.abs().quantile(0.9) if not unconditioned.empty else None,
                "conditioned_pooled_date_rank_corr": residual_corr.get(feature),
                "conditioned_median_daily_rank_corr": (
                    conditioned.median() if not conditioned.empty else None
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns_out)


def _maximum_correlation(correlations: pd.DataFrame, column: str) -> tuple[str | None, float | None]:
    if correlations.empty:
        return None, None
    values = pd.to_numeric(correlations[column], errors="coerce").abs().dropna()
    if values.empty:
        return None, None
    index = values.idxmax()
    return str(correlations.loc[index, "gen2_feature"]), float(values.loc[index])


def _temporal_ready(role: str, metrics: dict[str, Any], settings: AuditSettings) -> bool:
    if role == "CONTINUOUS":
        return (
            metrics["active_dates"] >= settings.continuous_minimum_dates
            and metrics["median_symbols_per_active_date"] >= settings.minimum_daily_cross_section
            and metrics["unique_values"] > 2
        )
    if role == "SPARSE_ASOF":
        return (
            metrics["active_dates"] >= settings.sparse_asof_minimum_dates
            and metrics["median_symbols_per_active_date"] >= settings.minimum_daily_cross_section
            and metrics["unique_values"] > 2
        )
    if role == "EVENT_SPARSE":
        return metrics["observations"] >= settings.event_minimum_observations
    if role == "SNAPSHOT_SPARSE":
        return metrics["active_dates"] >= 20
    return role == "CONTROL_ONLY"


def _novelty_class(maximum: float | None, settings: AuditSettings) -> str:
    if maximum is None:
        return "NOT_ESTIMABLE"
    if maximum >= settings.redundancy_threshold:
        return "HIGH_REDUNDANCY"
    if maximum >= settings.partial_redundancy_threshold:
        return "PARTIAL_REDUNDANCY"
    return "LOW_REDUNDANCY"


def _selection_status(row: pd.Series) -> str:
    if not bool(row["pit_admissible"]):
        return "EXCLUDE_PIT_RISK"
    role = row["role"]
    if role == "CONTROL_ONLY":
        return "CONTROL_ONLY"
    if role == "RESEARCH_ONLY_REVISION_RISK":
        return "EXCLUDE_REVISION_RISK"
    if role == "EVENT_SPARSE" and bool(row["temporal_ready"]):
        return "KEEP_ACCUMULATING_EVENT"
    if role == "SNAPSHOT_SPARSE":
        return "KEEP_ACCUMULATING_SNAPSHOT"
    if not bool(row["temporal_ready"]):
        return "PAUSE_INSUFFICIENT_STABILITY"
    if row["novelty_class"] == "HIGH_REDUNDANCY":
        return "PAUSE_HIGH_REDUNDANCY"
    return "KEEP_FOR_RESIDUAL_AUDIT"


def _selection_score(row: pd.Series) -> float:
    role = row["role"]
    if role == "CONTINUOUS":
        date_score = min(float(row["active_dates"]) / 120.0, 1.0)
    elif role == "SPARSE_ASOF":
        date_score = min(float(row["active_dates"]) / 30.0, 1.0)
    else:
        date_score = min(float(row["observations"]) / 300.0, 1.0)
    cross_section_score = min(float(row["median_symbols_per_active_date"]) / 50.0, 1.0)
    maximum = row["maximum_abs_gen2_rank_corr"]
    novelty = 0.5 if pd.isna(maximum) else max(0.0, 1.0 - float(maximum))
    return 0.45 * date_score + 0.30 * cross_section_score + 0.25 * novelty


def _shortlist(diagnostics: pd.DataFrame, settings: AuditSettings) -> pd.DataFrame:
    eligible_statuses = {
        "KEEP_FOR_RESIDUAL_AUDIT",
        "KEEP_ACCUMULATING_EVENT",
        "KEEP_ACCUMULATING_SNAPSHOT",
    }
    candidates = diagnostics[diagnostics["selection_status"].isin(eligible_statuses)].copy()
    candidates["selection_score"] = candidates.apply(_selection_score, axis=1)
    candidates = candidates.sort_values(
        ["selection_score", "feature"], ascending=[False, True]
    )
    event = candidates[candidates["selection_status"] == "KEEP_ACCUMULATING_EVENT"].head(
        settings.shortlist_event_reserve
    )
    snapshot = candidates[
        candidates["selection_status"] == "KEEP_ACCUMULATING_SNAPSHOT"
    ].head(settings.shortlist_snapshot_reserve)
    reserved = pd.concat([event, snapshot], ignore_index=False)
    remaining_slots = settings.shortlist_target - len(reserved)
    residual = candidates[
        candidates["selection_status"] == "KEEP_FOR_RESIDUAL_AUDIT"
    ].head(remaining_slots)
    selected = pd.concat([residual, reserved], ignore_index=False)
    if len(selected) < settings.shortlist_target:
        selected_ids = set(selected["feature"])
        fill = candidates[~candidates["feature"].isin(selected_ids)].head(
            settings.shortlist_target - len(selected)
        )
        selected = pd.concat([selected, fill], ignore_index=False)
    selected = selected.sort_values(
        ["selection_status", "selection_score", "feature"],
        ascending=[True, False, True],
    ).copy()
    selected.insert(0, "shortlist_rank", range(1, len(selected) + 1))
    selected["predictive_alpha_claim"] = False
    selected["admission"] = "COLLECTION_SHORTLIST_RESEARCH_ONLY"
    return selected


def _quota_recommendations(diagnostics: pd.DataFrame) -> dict[str, Any]:
    kept = diagnostics[diagnostics["selection_status"].str.startswith("KEEP_")]
    factor_names = sorted(
        kept.loc[kept["source_dataset"] == "FACTOR_LIBRARY", "feature"].astype(str)
    )
    valuation_names = sorted(
        kept.loc[kept["source_dataset"] == "VALUATION", "feature"].astype(str)
    )
    return {
        "provider_queries_this_audit": 0,
        "continue": {
            "FACTOR_LIBRARY": {
                "action": "REDUCE_TO_COLLECTION_SHORTLIST",
                "features": factor_names,
            },
            "VALUATION": {
                "action": "WEEKLY_INCREMENTAL_ONLY",
                "features": valuation_names,
            },
            "STK_FIN_FORCAST": "INCREMENTAL_EVENTS_ONLY",
            "STK_HK_HOLD_INFO": "INCREMENTAL_SNAPSHOTS_ONLY",
            "GET_HISTORY_INDUSTRY": "INCREMENTAL_CHANGES_ONLY",
            "STK_REPORT_DISCLOSURE": "INCREMENTAL_PUBLICATION_METADATA_ONLY",
        },
        "pause_or_stop": {
            "MONEYFLOW_HISTORY_DAILY": "STOP_RETRY_UNTIL_ENTITLEMENT_CHANGES",
            "FINANCE_BALANCE_SHEET": "PAUSE_UNTIL_RESTATEMENT_REPLAY_IS_PROVED",
            "FINANCE_INCOME_STATEMENT": "PAUSE_UNTIL_RESTATEMENT_REPLAY_IS_PROVED",
            "FINANCE_CASHFLOW_STATEMENT": "PAUSE_UNTIL_RESTATEMENT_REPLAY_IS_PROVED",
            "STK_PERFORMANCE_LETTERS": "PAUSE_FEATURE_ADMISSION_UNTIL_REVISION_LINEAGE_IS_PROVED",
            "INDUSTRY_CATALOG": "STATIC_CACHE_REFRESH_ONLY_ON_PROVIDER_VERSION_CHANGE",
        },
    }


def run(settings: AuditSettings) -> dict[str, Any]:
    protocol = _load_protocol(settings)
    inputs = protocol["inputs"]
    wide, long = load_jq_feature_store(
        settings.jq_feature_store,
        inputs["jq_features_wide_sha256"],
        inputs["jq_features_long_sha256"],
    )
    gen2 = load_gen2_features_only(settings.gen2_panel, inputs["gen2_panel_sha256"])
    lineage = _lineage_map(long)
    joined = wide.merge(gen2, on=["date", "symbol"], how="inner", validate="one_to_one")
    if joined.empty:
        raise RuntimeError("NO_JQDATA_GEN2_KEY_OVERLAP")
    features = [name for name in wide.columns if name not in {"date", "symbol"}]
    correlation_parts = []
    diagnostic_rows = []
    for feature in features:
        metrics = _temporal_metrics(wide, feature)
        role = _role(_family(feature))
        correlations = _correlations(joined, feature, settings.minimum_daily_cross_section)
        if not correlations.empty:
            correlations["sufficient_observations"] = (
                correlations["correlation_dates"] * settings.minimum_daily_cross_section
                >= settings.minimum_correlation_observations
            )
            correlation_parts.append(correlations)
        usable = correlations[
            correlations["correlation_dates"] * settings.minimum_daily_cross_section
            >= settings.minimum_correlation_observations
        ]
        best, maximum = _maximum_correlation(usable, "pooled_date_rank_corr")
        conditioned_best, conditioned_maximum = _maximum_correlation(
            usable, "conditioned_pooled_date_rank_corr"
        )
        effective_maximum = max(
            value for value in (maximum, conditioned_maximum) if value is not None
        ) if maximum is not None or conditioned_maximum is not None else None
        row = {
            "feature": feature,
            "family": _family(feature),
            "role": role,
            **lineage[feature],
            **metrics,
            "overlap_observations": int(pd.to_numeric(joined[feature], errors="coerce").notna().sum()),
            "most_correlated_gen2_feature": best,
            "maximum_abs_unconditioned_gen2_rank_corr": maximum,
            "most_correlated_conditioned_gen2_feature": conditioned_best,
            "maximum_abs_conditioned_gen2_rank_corr": conditioned_maximum,
            "maximum_abs_gen2_rank_corr": effective_maximum,
            "novelty_class": _novelty_class(effective_maximum, settings),
            "temporal_ready": _temporal_ready(role, metrics, settings),
        }
        diagnostic_rows.append(row)
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(["family", "feature"])
    diagnostics["selection_status"] = diagnostics.apply(_selection_status, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        correlations = (
            pd.concat(correlation_parts, ignore_index=True)
            if correlation_parts
            else pd.DataFrame()
        )
    shortlist = _shortlist(diagnostics, settings)
    if len(shortlist) > settings.shortlist_maximum:
        raise RuntimeError("SHORTLIST_CAP_EXCEEDED")
    quota = _quota_recommendations(diagnostics)
    status_counts = diagnostics["selection_status"].value_counts().sort_index().to_dict()
    summary = {
        "audit": "PREDICTION_V2_JQDATA_FEATURE_OVERLAP_AND_INFORMATION_AUDIT",
        "status": "PREDICTION_V2_JQDATA_INFORMATION_AUDIT_COMPLETE",
        "research_status": "RESEARCH_ONLY",
        "inputs": {
            "jq_features": len(features),
            "jq_rows": len(wide),
            "jq_symbols": int(wide["symbol"].nunique()),
            "jq_date_min": str(wide["date"].min().date()),
            "jq_date_max": str(wide["date"].max().date()),
            "gen2_features": len(V10_FEATURES),
            "gen2_rows_safe_projection": len(gen2),
            "gen2_symbols": int(gen2["symbol"].nunique()),
            "gen2_date_min": str(gen2["date"].min().date()),
            "gen2_date_max": str(gen2["date"].max().date()),
            "overlap_rows": len(joined),
            "overlap_symbols": int(joined["symbol"].nunique()),
            "overlap_dates": int(joined["date"].nunique()),
            "overlap_date_min": str(joined["date"].min().date()),
            "overlap_date_max": str(joined["date"].max().date()),
        },
        "classification_counts": status_counts,
        "collection_shortlist_count": len(shortlist),
        "collection_shortlist": shortlist["feature"].tolist(),
        "high_redundancy": diagnostics.loc[
            diagnostics["novelty_class"] == "HIGH_REDUNDANCY", "feature"
        ].tolist(),
        "insufficient_stability": diagnostics.loc[
            ~diagnostics["temporal_ready"], "feature"
        ].tolist(),
        "integrity": {
            "return_label_columns_requested": [],
            "return_label_values_read": False,
            "rank_ic": "NOT_COMPUTED",
            "model_training": False,
            "predictive_alpha_claim": False,
            "provider_queries": 0,
            "gen2_modified": 0,
            "contracts_007_012_modified": 0,
            "daily_pit_modified": 0,
            "daily_prediction_modified": 0,
            "production_modified": 0,
        },
        "interpretation": (
            "Novelty means low contemporaneous structural correlation with the frozen Gen2 features; "
            "without return labels it is not evidence of predictive alpha."
        ),
        "next_stage": "JQDATA_FORWARD_ONLY_RESIDUAL_CHALLENGER_RESEARCH_ONLY",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    artifact_dir = settings.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(artifact_dir / "feature_diagnostics.csv", diagnostics)
    _write_csv(artifact_dir / "gen2_overlap_matrix.csv", correlations)
    _write_csv(artifact_dir / "collection_shortlist.csv", shortlist)
    _write_json(artifact_dir / "quota_recommendations.json", quota)
    _write_json(artifact_dir / "audit_summary.json", summary)
    report = _report(summary, diagnostics, shortlist, quota)
    _write_bytes(artifact_dir / "PREDICTION_V2_JQDATA_FEATURE_OVERLAP_AND_INFORMATION_AUDIT_REPORT.md", report.encode("utf-8"))
    _write_bytes(
        artifact_dir / "PREDICTION_V2_JQDATA_FEATURE_OVERLAP_AND_INFORMATION_AUDIT_REPORT.md.sha256",
        f"{hashlib.sha256(report.encode('utf-8')).hexdigest()}\n".encode(),
    )
    manifest = {
        path.name: sha256_file(path)
        for path in sorted(artifact_dir.iterdir())
        if path.is_file() and not path.name.startswith("artifact_manifest")
    }
    _write_json(artifact_dir / "artifact_manifest.json", manifest)
    return summary


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_None_"
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame[columns].iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6f}" if math.isfinite(value) else "")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, rule, *rows])


def _report(
    summary: dict[str, Any],
    diagnostics: pd.DataFrame,
    shortlist: pd.DataFrame,
    quota: dict[str, Any],
) -> str:
    inputs = summary["inputs"]
    shortlist_view = shortlist[
        [
            "shortlist_rank",
            "feature",
            "family",
            "selection_status",
            "active_dates",
            "maximum_abs_gen2_rank_corr",
            "novelty_class",
        ]
    ]
    redundant = diagnostics[diagnostics["novelty_class"] == "HIGH_REDUNDANCY"][
        ["feature", "most_correlated_gen2_feature", "maximum_abs_gen2_rank_corr"]
    ]
    stop_lines = "\n".join(
        f"- `{dataset}`: {action}" for dataset, action in quota["pause_or_stop"].items()
    )
    return f"""# PREDICTION_V2_JQDATA_FEATURE_OVERLAP_AND_INFORMATION_AUDIT_REPORT

## Outcome

Status: `PREDICTION_V2_JQDATA_INFORMATION_AUDIT_COMPLETE`

Research status: `RESEARCH_ONLY`

This audit identifies structural overlap, coverage and collection readiness only. It did not read return-label values, compute RankIC, train a model or claim predictive alpha.

## Inputs

- JQData features: {inputs['jq_features']}
- JQData rows / symbols: {inputs['jq_rows']} / {inputs['jq_symbols']}
- JQData date range: {inputs['jq_date_min']} to {inputs['jq_date_max']}
- Frozen Gen2 features: {inputs['gen2_features']}
- Safe-projection Gen2 rows / symbols: {inputs['gen2_rows_safe_projection']} / {inputs['gen2_symbols']}
- Exact overlap: {inputs['overlap_rows']} rows, {inputs['overlap_symbols']} symbols, {inputs['overlap_dates']} dates
- Overlap range: {inputs['overlap_date_min']} to {inputs['overlap_date_max']}

## Collection Shortlist

The {len(shortlist)} entries below are worth continued collection or bounded residual research. They are not promoted signals.

{_markdown_table(shortlist_view, list(shortlist_view.columns))}

## High Structural Redundancy

Threshold: absolute contemporaneous date-ranked correlation at least 0.85, including sector-conditioned comparison.

{_markdown_table(redundant, list(redundant.columns))}

## Classification Counts

```json
{json.dumps(summary['classification_counts'], ensure_ascii=False, indent=2, sort_keys=True)}
```

## Quota Actions

Provider queries in this audit: 0.

Pause or stop consuming quota:

{stop_lines}

The factor library should be reduced to the recorded collection shortlist. Valuation stays weekly; company forecasts, HK holdings and disclosure metadata stay incremental-only.

## Integrity

- Return labels read: FALSE
- RankIC: NOT_COMPUTED
- Model training: FALSE
- Predictive alpha claim: FALSE
- Gen2 modified: 0
- 007-012 modified: 0
- DAILY PIT modified: 0
- daily prediction modified: 0
- Production modified: 0

## Next Stage

`JQDATA_FORWARD_ONLY_RESIDUAL_CHALLENGER_RESEARCH_ONLY`

That stage must remain research-only until enough mature, cross-regime forward observations exist. Frozen Gen2 remains the production champion.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jq-feature-store", type=Path, required=True)
    parser.add_argument("--gen2-panel", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("artifacts/prediction_v2/jqdata_feature_overlap_audit/protocol.json"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/prediction_v2/jqdata_feature_overlap_audit"),
    )
    args = parser.parse_args()
    settings = AuditSettings(
        jq_feature_store=args.jq_feature_store,
        gen2_panel=args.gen2_panel,
        protocol_path=args.protocol,
        artifact_dir=args.artifact_dir,
    )
    print(json.dumps(run(settings), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
