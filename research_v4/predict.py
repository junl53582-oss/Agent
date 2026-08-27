from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research_v3.fundamentals import attach_fundamentals_asof, load_fundamentals
from stockpilot.exposure import attach_exposures_asof, load_exposures
from stockpilot.membership import attach_point_in_time_membership, load_membership_history
from stockpilot.portfolio import portfolio_weights, select_with_buffer_and_cap
from stockpilot.shadow import load_shadow_panel

from .config import PLAN_LOCK_SHA256, V4Settings
from .features import FACTOR_COLUMNS, build_v4_dataset
from .lock import verify_plan_lock
from .stability import FactorSpec, learn_factor_specs, score_with_specs


def build_latest_snapshot(
    dataset: pd.DataFrame,
    specs: list[FactorSpec],
    settings: V4Settings | None = None,
    previous_symbols: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = settings or V4Settings()
    latest_date = pd.to_datetime(dataset["date"]).max()
    current = dataset[
        (pd.to_datetime(dataset["date"]) == latest_date) & dataset["eligible"]
    ].copy()
    if current.empty:
        raise RuntimeError("最新交易日没有可预测股票")
    if not any(spec.selected for spec in specs):
        raise RuntimeError("训练期没有稳定因子，V4按协议应持有现金，不生成候选")
    current["score"] = score_with_specs(current, specs)
    current["pred_rank"] = current["score"].rank(ascending=False, method="first").astype(int)
    current["selected"] = False
    selected = select_with_buffer_and_cap(
        current,
        settings.top_n,
        previous_symbols or set(),
        settings.hold_buffer,
        settings.industry_cap,
    )
    if len(selected) < settings.min_positions:
        raise RuntimeError(f"V4最新候选仅{len(selected)}只，低于{settings.min_positions}只")
    selected["weight"] = portfolio_weights(selected, "inverse_volatility")
    current.loc[selected.index, "selected"] = True
    weights = selected.set_index("symbol")["weight"]
    current["weight"] = current["symbol"].map(weights).fillna(0.0)
    current["rank"] = current["pred_rank"]
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "research_v4_factor_stability",
        "protocol_status": "retrospective_research",
        "execution_authorized": False,
        "plan_lock_sha256": PLAN_LOCK_SHA256,
        "training_cutoff": f"{latest_date.year - 1}-12-31",
    }
    for key, value in metadata.items():
        current[key] = value
    prediction_columns = [
        "date",
        "symbol",
        "close",
        "score",
        "pred_rank",
        "selected",
        "weight",
        *FACTOR_COLUMNS,
        *metadata,
    ]
    predictions = current[prediction_columns].sort_values("pred_rank").reset_index(drop=True)
    signals = predictions[predictions["selected"]].copy()
    signals["rank"] = signals["pred_rank"]
    signal_columns = [
        "date",
        "rank",
        "symbol",
        "close",
        "score",
        "weight",
        *FACTOR_COLUMNS,
        *metadata,
    ]
    return signals[signal_columns].reset_index(drop=True), predictions


def _stable_content(data: pd.DataFrame) -> pd.DataFrame:
    stable = data.drop(columns=["generated_at_utc"], errors="ignore").copy()
    if "date" in stable:
        stable["date"] = pd.to_datetime(stable["date"]).dt.strftime("%Y-%m-%d")
    return stable.reset_index(drop=True)


def _write_immutable_snapshot(data: pd.DataFrame, path: Path) -> bool:
    if path.exists():
        existing = pd.read_csv(path, dtype={"symbol": str})
        expected = data.copy()
        existing["symbol"] = existing["symbol"].str.zfill(6)
        expected["symbol"] = expected["symbol"].astype(str).str.zfill(6)
        try:
            pd.testing.assert_frame_equal(
                _stable_content(existing),
                _stable_content(expected),
                check_dtype=False,
                check_exact=False,
                rtol=1e-10,
                atol=1e-12,
            )
        except AssertionError as exc:
            raise RuntimeError(f"V4快照已存在且内容不一致，拒绝覆盖: {path}") from exc
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8-sig")
    return True


def update_latest_prediction(
    baseline_path: str | Path = "data/market_history.csv",
    bar_dir: str | Path = "data/shadow/bars",
    membership_path: str | Path = "data/universes/000300/history.csv",
    exposure_path: str | Path = "data/exposures.csv",
    shadow_exposure_dir: str | Path = "data/shadow/exposures",
    fundamental_path: str | Path = "data/fundamentals_pit.csv",
    output_dir: str | Path = "artifacts/research_v4/live",
    settings: V4Settings | None = None,
) -> dict:
    settings = settings or V4Settings()
    verify_plan_lock(settings.plan_lock_path)
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
    dataset = build_v4_dataset(panel)
    latest_date = pd.to_datetime(dataset["date"]).max()
    specs, _ = learn_factor_specs(dataset, int(latest_date.year), settings)
    target = Path(output_dir)
    previous_paths = sorted((target / "signals").glob("*.csv"))
    previous_symbols: set[str] = set()
    if previous_paths and previous_paths[-1].stem < str(latest_date.date()):
        previous = pd.read_csv(previous_paths[-1], dtype={"symbol": str})
        previous_symbols = set(previous["symbol"].str.zfill(6))
    signals, predictions = build_latest_snapshot(dataset, specs, settings, previous_symbols)
    date_text = str(latest_date.date())
    signal_path = target / "signals" / f"{date_text}.csv"
    prediction_path = target / "predictions" / f"{date_text}.csv"
    wrote_signal = _write_immutable_snapshot(signals, signal_path)
    wrote_prediction = _write_immutable_snapshot(predictions, prediction_path)
    selected_specs = [asdict(spec) for spec in specs if spec.selected]
    report = {
        "latest_prediction_date": date_text,
        "model": "research_v4_factor_stability",
        "prediction_count": len(predictions),
        "signal_count": len(signals),
        "signal_path": str(signal_path),
        "prediction_path": str(prediction_path),
        "selected_factors": selected_specs,
        "execution_authorized": False,
        "protocol_status": "retrospective_research",
        "already_recorded": not wrote_signal and not wrote_prediction,
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / "latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return report
