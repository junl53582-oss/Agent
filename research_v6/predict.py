from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research_v3.fundamentals import attach_fundamentals_asof, load_fundamentals
from research_v4.config import V4Settings
from research_v4.lock import verify_plan_lock
from research_v4.predict import _write_immutable_snapshot
from research_v4.stability import learn_factor_specs
from research_v5.features import build_v5_dataset
from research_v5.models import fit_v5_models
from stockpilot.exposure import attach_exposures_asof, load_exposures
from stockpilot.membership import attach_point_in_time_membership, load_membership_history
from stockpilot.shadow import load_shadow_panel

from .config import PLAN_LOCK_SHA256, V6Settings
from .model import score_v6, select_sector_balanced


def update_latest_prediction(
    baseline_path: str | Path = "data/market_history.csv",
    bar_dir: str | Path = "data/shadow/bars",
    membership_path: str | Path = "data/universes/000300/history.csv",
    exposure_path: str | Path = "data/exposures.csv",
    shadow_exposure_dir: str | Path = "data/shadow/exposures",
    fundamental_path: str | Path = "data/fundamentals_pit.csv",
    output_dir: str | Path = "artifacts/research_v6/live",
    settings: V6Settings | None = None,
) -> dict:
    settings = settings or V6Settings()
    verify_plan_lock(settings.plan_lock_path, PLAN_LOCK_SHA256)
    panel = load_shadow_panel(baseline_path, bar_dir)
    panel = attach_point_in_time_membership(panel, load_membership_history(membership_path))
    exposure_pieces = [load_exposures(exposure_path)]
    exposure_pieces.extend(
        load_exposures(path) for path in sorted(Path(shadow_exposure_dir).glob("*.csv"))
    )
    exposures = pd.concat(exposure_pieces, ignore_index=True)
    exposures = exposures.sort_values(["date", "symbol"]).drop_duplicates(
        ["date", "symbol"], keep="last"
    )
    panel = attach_exposures_asof(panel, exposures)
    panel = attach_fundamentals_asof(panel, load_fundamentals(fundamental_path))
    dataset = build_v5_dataset(panel)
    latest_date = pd.to_datetime(dataset["date"]).max()
    year = int(latest_date.year)
    v5_models = fit_v5_models(dataset, year)
    v4_specs, _ = learn_factor_specs(dataset, year, V4Settings())
    current = dataset[
        (pd.to_datetime(dataset["date"]) == latest_date) & dataset["eligible"]
    ].copy()
    scored = score_v6(current, v5_models, v4_specs, settings)
    scored["pred_rank"] = scored["score"].rank(ascending=False, method="first").astype(int)
    selected = select_sector_balanced(scored, settings)
    scored["selected"] = scored.index.isin(selected.index)
    weights = selected.set_index("symbol")["weight"]
    scored["weight"] = scored["symbol"].map(weights).fillna(0.0)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "research_v6_sector_balanced_ensemble",
        "protocol_status": "retrospective_research",
        "execution_authorized": False,
        "plan_lock_sha256": PLAN_LOCK_SHA256,
        "training_cutoff": f"{year - 1}-12-31",
    }
    for key, value in metadata.items():
        scored[key] = value
    columns = [
        "date", "symbol", "close", "broad_sector", "regime", "score", "pred_rank",
        "selected", "weight", "fundamental", "behavior", "risk", "liquidity_component",
        "global_model", "industry_expert", "v4_rank", *metadata,
    ]
    predictions = scored[columns].sort_values("pred_rank").reset_index(drop=True)
    signals = predictions[predictions["selected"]].copy()
    signals["rank"] = signals["pred_rank"]
    signal_columns = ["date", "rank", "symbol", "close", "broad_sector", "regime", "score", "weight", *metadata]
    signals = signals[signal_columns].reset_index(drop=True)
    target = Path(output_dir)
    date_text = str(latest_date.date())
    signal_path = target / "signals" / f"{date_text}.csv"
    prediction_path = target / "predictions" / f"{date_text}.csv"
    wrote_signal = _write_immutable_snapshot(signals, signal_path)
    wrote_prediction = _write_immutable_snapshot(predictions, prediction_path)
    report = {
        "latest_prediction_date": date_text,
        "model": metadata["model"],
        "prediction_count": len(predictions),
        "signal_count": len(signals),
        "signal_path": str(signal_path),
        "prediction_path": str(prediction_path),
        "sector_counts": signals["broad_sector"].value_counts().to_dict(),
        "execution_authorized": False,
        "replacement_approved": True,
        "already_recorded": not wrote_signal and not wrote_prediction,
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / "latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
