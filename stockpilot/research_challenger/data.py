from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ChallengerSettings, FORBIDDEN_FEATURE_TOKENS


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset_manifest(settings: ChallengerSettings) -> dict:
    if not settings.dataset_path.exists() or not settings.dataset_manifest_path.exists():
        raise RuntimeError("REAL_DATA_NOT_AVAILABLE")
    manifest = json.loads(settings.dataset_manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for name, expected in manifest.get("source_hashes", {}).items():
        path = Path(name)
        actual = sha256(path) if path.exists() else "MISSING"
        if actual != expected:
            mismatches.append({"path": name, "expected": expected, "actual": actual})
    if mismatches:
        raise RuntimeError(f"V31_INPUT_HASH_MISMATCH: {mismatches}")
    return {
        "dataset_sha256": sha256(settings.dataset_path),
        "manifest_sha256": sha256(settings.dataset_manifest_path),
        "source_hashes": manifest["source_hashes"],
        "manifest_rows": int(manifest["rows"]),
    }


def assert_feature_columns_safe(columns: list[str] | tuple[str, ...]) -> None:
    forbidden = [
        name
        for name in columns
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"future or label columns cannot enter features: {forbidden}")


def _rank_by_date(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
    return values.groupby(pd.to_datetime(frame["date"])).rank(pct=True, method="average")


def add_research_targets(frame: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    data = frame.copy()
    for horizon in horizons:
        returns = pd.to_numeric(data[f"future_return_{horizon}d"], errors="coerce")
        data[f"return_rank_{horizon}d"] = _rank_by_date(data, returns)
        industry_mean = returns.groupby(
            [pd.to_datetime(data["date"]), data["industry"].fillna("UNKNOWN")]
        ).transform("mean")
        data[f"industry_alpha_{horizon}d"] = returns - industry_mean
        data[f"industry_alpha_rank_{horizon}d"] = _rank_by_date(
            data, data[f"industry_alpha_{horizon}d"]
        )
    return data


def load_research_dataset(settings: ChallengerSettings) -> tuple[pd.DataFrame, dict]:
    evidence = verify_dataset_manifest(settings)
    assert_feature_columns_safe(settings.factor_columns)
    data = pd.read_parquet(settings.dataset_path)
    if len(data) != evidence["manifest_rows"]:
        raise RuntimeError("V31_DATASET_ROW_COUNT_MISMATCH")
    data["date"] = pd.to_datetime(data["date"])
    data["symbol"] = data["symbol"].astype(str).str.zfill(6)
    if data.duplicated(["date", "symbol"]).any():
        raise RuntimeError("V31_DUPLICATE_DATE_SYMBOL")
    required = {
        "eligible",
        "in_universe",
        "membership_snapshot_date",
        "available_date",
        "industry_effective_date",
        "industry",
        "broad_sector",
        "benchmark_weight",
        "regime",
        "volatility_20",
        "amount",
        *settings.factor_columns,
    }
    for horizon in settings.horizons:
        required.update({f"future_return_{horizon}d", f"label_end_date_{horizon}d"})
    missing = sorted(required.difference(data.columns))
    if missing:
        raise RuntimeError(f"V31_DATASET_SCHEMA_MISSING: {missing}")
    decision_dates = pd.to_datetime(data["date"])
    membership = pd.to_datetime(data["membership_snapshot_date"], errors="coerce")
    fundamentals = pd.to_datetime(data["available_date"], errors="coerce")
    industries = pd.to_datetime(data["industry_effective_date"], errors="coerce")
    checks = {
        "membership_pit": bool((membership.isna() | membership.le(decision_dates)).all()),
        "fundamentals_pit": bool((fundamentals.isna() | fundamentals.le(decision_dates)).all()),
        "industry_pit": bool((industries.isna() | industries.le(decision_dates)).all()),
        "no_prospective_paths": all(
            "prospective" not in name.lower() for name in evidence["source_hashes"]
        ),
        "finite_features": bool(
            np.isfinite(data[list(settings.factor_columns)].to_numpy(dtype=float)).all()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"V31_PIT_AUDIT_FAILED: {checks}")
    eligible = data["eligible"].fillna(False) & data["in_universe"].fillna(False)
    data = data.loc[eligible].copy()
    data = add_research_targets(data, settings.horizons)
    evidence.update(
        {
            "rows": int(len(data)),
            "symbols": int(data["symbol"].nunique()),
            "date_min": str(data["date"].min().date()),
            "date_max": str(data["date"].max().date()),
            "pit_checks": checks,
            "benchmark_target_status": "DISABLED_BENCHMARK_EVIDENCE_UNAPPROVED",
            "prospective_rows_used": 0,
        }
    )
    return data.sort_values(["date", "symbol"]).reset_index(drop=True), evidence


def source_column(feature: str) -> str | None:
    explicit = {
        "quality": "quality_score",
        "growth": "growth_score",
        "momentum": "ret_20",
        "short_reversal": "ret_5",
        "volume_attention": "volume_ratio_20",
        "low_volatility": "volatility_20",
        "liquidity": "amount",
        "industry_momentum": "industry_momentum",
        "fundamental_coverage": "fundamental_coverage",
        "fundamental_freshness_rank": "fundamental_age_days",
        "benchmark_weight_rank": "benchmark_weight",
    }
    if feature in explicit:
        return explicit[feature]
    if feature.startswith("technology_"):
        return "industry"
    base = feature.removesuffix("_rank")
    return base


def factor_group(feature: str) -> str:
    if feature.startswith("technology_") or feature == "industry_momentum":
        return "industry_technology"
    if any(token in feature for token in ("roe", "roic", "margin", "turnover", "cash", "debt", "cycle", "staff", "fcff", "growth", "book_to_price", "earnings_yield", "quality", "fundamental")):
        return "fundamental"
    if any(token in feature for token in ("volatility", "drawdown", "low_volatility")):
        return "risk"
    if any(token in feature for token in ("volume", "liquidity", "amount")):
        return "liquidity"
    return "price_behavior"


def factor_lookback(feature: str) -> str:
    for window in (250, 120, 60, 20, 14, 10, 5, 1):
        if str(window) in feature:
            return f"{window} trading days"
    if factor_group(feature) == "fundamental":
        return "latest PIT filing and prior PIT filing for change features"
    return "composite/current PIT cross-section"


def factor_inventory(data: pd.DataFrame, settings: ChallengerSettings) -> pd.DataFrame:
    rows = []
    for feature in settings.factor_columns:
        source = source_column(feature)
        raw_coverage = (
            float(pd.to_numeric(data[source], errors="coerce").notna().mean())
            if source in data.columns and source != "industry"
            else float(data[source].notna().mean()) if source in data.columns else float("nan")
        )
        rows.append(
            {
                "factor_name": feature,
                "factor_group": factor_group(feature),
                "source": source or "derived",
                "lookback": factor_lookback(feature),
                "pit_available": True,
                "missing_ratio_after_existing_v10_processing": float(data[feature].isna().mean()),
                "raw_source_coverage": raw_coverage,
                "coverage": float(np.isfinite(data[feature].to_numpy(dtype=float)).mean()),
            }
        )
    return pd.DataFrame(rows)
