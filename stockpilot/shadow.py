from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .audit import (
    append_audit_record,
    settings_from_addendum,
    verify_audit_chain,
    verify_protocol_addendum,
)
from .backtest import generate_latest_prediction_snapshot
from .config import Settings
from .data import fetch_akshare, load_panel, save_panel, validate_panel
from .exposure import attach_exposures_asof, load_exposures
from .future_test import future_test_status, verify_frozen_inputs
from .membership import attach_point_in_time_membership, load_membership_history


def _shadow_settings(manifest: dict, base: Settings | None = None) -> Settings:
    base = base or Settings.from_env()
    selected = manifest["selected_config"]
    allowed = {"model_name", "top_n", "weighting", "hold_buffer", "industry_cap"}
    return replace(base, **{key: selected[key] for key in allowed if key in selected})


def load_shadow_panel(baseline_path: str | Path, bar_dir: str | Path) -> pd.DataFrame:
    pieces = [load_panel(baseline_path)]
    pieces.extend(load_panel(path) for path in sorted(Path(bar_dir).glob("*.csv")))
    return validate_panel(pd.concat(pieces, ignore_index=True))


def is_signal_due(observed_trading_days: int, rebalance_every: int) -> bool:
    """Emit on observation 1, then once per frozen rebalance interval."""
    if observed_trading_days < 1 or rebalance_every < 1:
        return False
    return (observed_trading_days - 1) % rebalance_every == 0


def update_shadow_observation(
    manifest_path: str | Path,
    end_date: str,
    bar_dir: str | Path = "data/shadow/bars",
    signal_dir: str | Path = "artifacts/future_test/signals",
    provider: str = "tencent",
    workers: int = 4,
    max_exposure_age_days: int = 7,
    shadow_exposure_dir: str | Path = "data/shadow/exposures",
    prediction_dir: str | Path = "artifacts/future_test/predictions",
    addendum_path: str | Path = "artifacts/future_test/protocol.addendum.lock.json",
    audit_chain_path: str | Path = "artifacts/future_test/audit_chain.jsonl",
) -> dict:
    """Append new bars and write one immutable shadow-signal snapshot for the latest date."""
    verify_frozen_inputs(manifest_path)
    addendum_target = Path(addendum_path)
    chain_target = Path(audit_chain_path)
    if addendum_target.exists():
        verify_protocol_addendum(addendum_target)
        protocol = json.loads(addendum_target.read_text(encoding="utf-8"))
        locked_provider = protocol.get("runtime", {}).get("market_data_provider")
        if locked_provider and provider != locked_provider:
            raise RuntimeError(f"行情提供方已冻结为{locked_provider}，拒绝使用{provider}")
    if chain_target.exists():
        verify_audit_chain(chain_target)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    frozen = manifest["frozen_inputs"]
    baseline_path = Path(frozen["market"]["path"])
    membership_path = Path(frozen["membership"]["path"])
    exposure_path = Path(frozen["exposure"]["path"])
    evaluation_start = pd.Timestamp(manifest["evaluation_start"])
    cutoff = pd.Timestamp(manifest["research_cutoff"])
    target_end = pd.Timestamp(end_date).normalize()
    if target_end < evaluation_start:
        raise ValueError("end_date 早于冻结的未来测试起点")

    bars_target = Path(bar_dir)
    bars_target.mkdir(parents=True, exist_ok=True)
    existing_dates: set[pd.Timestamp] = set()
    for path in bars_target.glob("*.csv"):
        dates = pd.read_csv(path, usecols=["date"])["date"]
        existing_dates.update(pd.to_datetime(dates, errors="coerce").dropna().dt.normalize())
    latest_existing = max(existing_dates) if existing_dates else cutoff
    fetch_start = max(latest_existing + pd.Timedelta(days=1), evaluation_start)
    downloaded_dates: list[str] = []
    if fetch_start <= target_end:
        history = load_membership_history(membership_path)
        latest_snapshot = history["snapshot_date"].max()
        active_symbols = history.loc[history["snapshot_date"] == latest_snapshot, "symbol"]
        fresh = fetch_akshare(
            active_symbols,
            str(fetch_start.date()),
            str(target_end.date()),
            cache_dir=bars_target.parent / "cache",
            provider=provider,
            workers=workers,
        )
        fresh = fresh[fresh["date"] >= evaluation_start]
        for date, snapshot in fresh.groupby("date"):
            path = bars_target / f"{pd.Timestamp(date).date()}.csv"
            if path.exists():
                raise FileExistsError(f"影子行情快照已存在，禁止覆盖: {path}")
            save_panel(snapshot, path)
            downloaded_dates.append(str(pd.Timestamp(date).date()))
            if chain_target.exists():
                append_audit_record(chain_target, path, "shadow_bar")

    available_dates = existing_dates | {pd.Timestamp(date) for date in downloaded_dates}
    latest_available = max(available_dates) if available_dates else None
    signal_target = Path(signal_dir)
    prediction_target = Path(prediction_dir)
    if latest_available is not None:
        existing_signal = signal_target / f"{latest_available.date()}.csv"
        existing_prediction = prediction_target / f"{latest_available.date()}.csv"
        if existing_signal.exists() and existing_prediction.exists() and not downloaded_dates:
            existing = pd.read_csv(existing_signal)
            return {
                "downloaded_dates": [],
                "latest_signal_date": str(latest_available.date()),
                "signal_count": len(existing),
                "signal_path": str(existing_signal),
                "execution_authorized": False,
                "already_recorded": True,
                "status": future_test_status(
                    manifest_path, baseline_path, bars_target, signal_target
                ),
            }

    panel = load_shadow_panel(baseline_path, bars_target)
    membership = load_membership_history(membership_path)
    panel = attach_point_in_time_membership(panel, membership)
    exposure_pieces = [load_exposures(exposure_path)]
    exposure_pieces.extend(
        load_exposures(path) for path in sorted(Path(shadow_exposure_dir).glob("*.csv"))
    )
    if chain_target.exists():
        for path in sorted(Path(shadow_exposure_dir).glob("*.csv")):
            append_audit_record(chain_target, path, "shadow_exposure")
    exposure = pd.concat(exposure_pieces, ignore_index=True)
    exposure = exposure.sort_values(["date", "symbol"]).drop_duplicates(
        ["date", "symbol"], keep="last"
    )
    panel = attach_exposures_asof(panel, exposure)
    latest_date = panel["date"].max()
    latest_scope = panel[(panel["date"] == latest_date) & panel["in_universe"]]
    exposure_coverage = float(latest_scope["float_market_cap"].notna().mean())
    exposure_age = float(latest_scope["exposure_age_days"].dropna().max())
    if exposure_coverage < 0.95:
        raise RuntimeError(f"最新暴露覆盖仅{exposure_coverage:.1%}，低于95%门槛")
    if exposure_age > max_exposure_age_days:
        raise RuntimeError(f"最新暴露已陈旧{exposure_age:.0f}天，超过{max_exposure_age_days}天门槛")

    settings = (
        settings_from_addendum(addendum_target)
        if addendum_target.exists()
        else _shadow_settings(manifest)
    )
    observed_days = len(panel.loc[panel["date"] >= evaluation_start, "date"].drop_duplicates())
    if not is_signal_due(observed_days, settings.rebalance_every):
        status = future_test_status(manifest_path, baseline_path, bars_target, signal_target)
        return {
            "downloaded_dates": downloaded_dates,
            "latest_observation": str(pd.Timestamp(latest_date).date()),
            "signal_due": False,
            "next_signal_in_trading_days": settings.rebalance_every
            - ((observed_days - 1) % settings.rebalance_every),
            "latest_exposure_coverage": exposure_coverage,
            "latest_exposure_age_days": exposure_age,
            "execution_authorized": False,
            "status": status,
        }
    prior_signal_paths = [
        path for path in sorted(signal_target.glob("*.csv")) if path.stem < str(latest_date.date())
    ]
    previous_symbols: set[str] = set()
    if prior_signal_paths:
        prior = pd.read_csv(prior_signal_paths[-1], dtype={"symbol": str})
        previous_symbols = set(prior["symbol"].str.zfill(6))
    signals, predictions = generate_latest_prediction_snapshot(panel, settings, previous_symbols)
    signal_date = pd.Timestamp(signals["date"].iloc[0])
    if signal_date < evaluation_start:
        raise RuntimeError("尚无未来测试期行情，不能生成影子信号")
    signals["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    signals["execution_authorized"] = False
    signals["protocol_mode"] = manifest["mode"]
    predictions["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    predictions["execution_authorized"] = False
    predictions["protocol_mode"] = manifest["mode"]
    signal_target.mkdir(parents=True, exist_ok=True)
    prediction_target.mkdir(parents=True, exist_ok=True)
    signal_path = signal_target / f"{signal_date.date()}.csv"
    prediction_path = prediction_target / f"{signal_date.date()}.csv"
    if signal_path.exists():
        existing = pd.read_csv(signal_path, dtype={"symbol": str})
        existing_symbols = existing["symbol"].astype(str).str.zfill(6).tolist()
        if existing_symbols != signals["symbol"].astype(str).str.zfill(6).tolist():
            raise RuntimeError(f"重建信号与已固化快照不一致: {signal_path}")
    else:
        signals.to_csv(signal_path, index=False, encoding="utf-8-sig")
    if prediction_path.exists():
        raise FileExistsError(f"全截面预测快照已存在，禁止覆盖: {prediction_path}")
    predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    if chain_target.exists():
        append_audit_record(chain_target, signal_path, "shadow_signal")
        append_audit_record(chain_target, prediction_path, "prediction_snapshot")

    status = future_test_status(manifest_path, baseline_path, bars_target, signal_target)
    return {
        "downloaded_dates": downloaded_dates,
        "latest_signal_date": str(signal_date.date()),
        "signal_count": len(signals),
        "signal_path": str(signal_path),
        "prediction_count": len(predictions),
        "prediction_path": str(prediction_path),
        "signal_due": True,
        "latest_exposure_coverage": exposure_coverage,
        "latest_exposure_age_days": exposure_age,
        "execution_authorized": False,
        "status": status,
    }
