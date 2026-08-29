from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from research_v10.features import build_v10_dataset
from research_v10.fundamentals import attach_extended_fundamentals_asof, load_extended_fundamentals
from research_v9.data import attach_industry_asof, attach_membership_weight, load_industry_history
from stockpilot.data import load_panel
from stockpilot.membership import attach_point_in_time_membership, load_membership_history

from .config import PredictionSettings
from .labels import add_prediction_labels


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_prediction_dataset(settings: PredictionSettings | None = None) -> pd.DataFrame:
    settings = settings or PredictionSettings()
    required = (
        settings.market_path, settings.membership_path, settings.fundamental_path,
        settings.industry_path,
    )
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise RuntimeError("REAL_DATA_NOT_AVAILABLE: " + ", ".join(missing))
    cache_dir = settings.artifact_dir / "cache"
    cache_path = cache_dir / "eligible_panel.parquet"
    cache_manifest_path = cache_dir / "manifest.json"
    source_hashes = {str(path): _hash(Path(path)) for path in required}
    if cache_path.exists() and cache_manifest_path.exists():
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if cache_manifest.get("source_hashes") == source_hashes:
            return pd.read_parquet(cache_path)
    membership = load_membership_history(settings.membership_path)
    panel = attach_point_in_time_membership(load_panel(settings.market_path), membership)
    panel = attach_membership_weight(panel, membership)
    panel = attach_extended_fundamentals_asof(
        panel, load_extended_fundamentals(settings.fundamental_path)
    )
    panel = attach_industry_asof(panel, load_industry_history(settings.industry_path))
    dataset = build_v10_dataset(panel)
    # V30 predicts only the actual point-in-time index universe. Filtering before
    # label construction avoids copying millions of out-of-universe rows while
    # preserving the complete market trading calendar.
    dataset = dataset[dataset["eligible"].fillna(False)].copy()
    dataset = add_prediction_labels(dataset, settings.horizons, settings.direction_thresholds)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(cache_path, index=False, compression="zstd")
    cache_manifest_path.write_text(
        json.dumps({"source_hashes": source_hashes, "rows": len(dataset)}, indent=2), encoding="utf-8"
    )
    return dataset


def pit_data_audit(frame: pd.DataFrame) -> dict:
    dates = pd.to_datetime(frame["date"])
    checks = {
        "unique_date_symbol": not frame.duplicated(["date", "symbol"]).any(),
        "membership_not_future": bool(
            (~frame["eligible"].fillna(False) | pd.to_datetime(frame["membership_snapshot_date"]).le(dates)).all()
        ),
        "fundamentals_not_future": bool(
            (frame["available_date"].isna() | pd.to_datetime(frame["available_date"]).le(dates)).all()
        ),
        "industry_not_future": bool(
            (frame["industry_effective_date"].isna() | pd.to_datetime(frame["industry_effective_date"]).le(dates)).all()
        ),
        "real_history_span": dates.min() <= pd.Timestamp("2015-01-31") and dates.max() >= pd.Timestamp("2025-12-31"),
        "minimum_symbols": frame["symbol"].nunique() >= 650,
    }
    return {
        "rows": int(len(frame)),
        "eligible_rows": int(frame["eligible"].fillna(False).sum()),
        "symbols": int(frame["symbol"].nunique()),
        "date_min": str(dates.min().date()),
        "date_max": str(dates.max().date()),
        "checks": checks,
        "passed": all(checks.values()),
    }
