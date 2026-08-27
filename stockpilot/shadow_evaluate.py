from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import (
    append_audit_record,
    settings_from_addendum,
    verify_audit_chain,
    verify_protocol_addendum,
)
from .config import Settings
from .exposure import attach_exposures_asof, load_exposures
from .features import build_dataset
from .future_test import verify_frozen_inputs
from .membership import attach_point_in_time_membership, load_membership_history
from .portfolio import turnover
from .shadow import _shadow_settings, load_shadow_panel

LEDGER_COLUMNS = [
    "signal_date",
    "entry_date",
    "exit_date",
    "gross_return",
    "net_return",
    "benchmark_return",
    "excess_return",
    "buy_turnover",
    "sell_turnover",
    "transaction_cost",
    "holdings",
    "cash_weight",
    "blocked_entries",
    "rank_ic",
    "exposure_coverage",
    "equity",
    "benchmark",
]


def _json_value(value):
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _period_accounting(
    stored_signals: pd.DataFrame,
    current: pd.DataFrame,
    previous_weights: dict[str, float],
    settings: Settings,
) -> tuple[dict, dict[str, float], list[dict]]:
    """Account one frozen signal snapshot using only subsequently observed prices."""
    selected = stored_signals[["symbol", "weight"]].copy()
    selected["symbol"] = selected["symbol"].astype(str).str.zfill(6)
    selected["planned_weight"] = pd.to_numeric(selected["weight"], errors="coerce").fillna(0)
    columns = [
        "symbol",
        "entry_date",
        "execution_exit_date",
        "entry_tradable",
        "entry_limit_up",
        "entry_open",
        "execution_exit_open",
        "execution_return",
        "future_return",
    ]
    selected = selected.merge(current[columns], on="symbol", how="left", validate="one_to_one")
    continuing = selected["symbol"].isin(previous_weights)
    selected["holding_return"] = selected["execution_exit_open"] / selected["entry_open"] - 1
    selected["realized_return"] = selected["execution_return"].where(
        ~continuing, selected["holding_return"]
    )
    selected["executed"] = (continuing | selected["entry_tradable"].fillna(False)) & selected[
        "realized_return"
    ].notna()
    selected["actual_weight"] = selected["planned_weight"].where(selected["executed"], 0.0)
    current_weights = {
        row.symbol: float(row.actual_weight)
        for row in selected.itertuples()
        if row.actual_weight > 0
    }
    buy_turnover, sell_turnover = turnover(previous_weights, current_weights)
    buy_rate = settings.fee_rate + settings.slippage
    sell_rate = settings.fee_rate + settings.slippage + settings.stamp_duty
    transaction_cost = buy_turnover * buy_rate + sell_turnover * sell_rate
    gross_return = float((selected["actual_weight"] * selected["realized_return"].fillna(0)).sum())
    eligible_returns = pd.to_numeric(current["future_return"], errors="coerce").dropna()
    benchmark_return = float(eligible_returns.mean())
    detail = selected[
        [
            "symbol",
            "planned_weight",
            "actual_weight",
            "executed",
            "entry_tradable",
            "entry_limit_up",
            "realized_return",
        ]
    ].to_dict(orient="records")
    result = {
        "gross_return": gross_return,
        "net_return": gross_return - transaction_cost,
        "benchmark_return": benchmark_return,
        "buy_turnover": buy_turnover,
        "sell_turnover": sell_turnover,
        "transaction_cost": transaction_cost,
        "holdings": len(current_weights),
        "cash_weight": 1 - sum(current_weights.values()),
        "blocked_entries": int((~continuing & ~selected["executed"]).sum()),
        "entry_date": selected["entry_date"].dropna().min(),
        "exit_date": selected["execution_exit_date"].dropna().max(),
    }
    return result, current_weights, detail


def _same_outcome(existing: dict, computed: dict) -> bool:
    keys = set(existing) - {"recorded_at_utc"}
    if keys != set(computed):
        return False
    for key in keys:
        left, right = existing[key], computed[key]
        if isinstance(left, float) or isinstance(right, float):
            if not np.isclose(left, right, rtol=1e-10, atol=1e-12, equal_nan=True):
                return False
        elif isinstance(left, (list, dict)):
            if json.dumps(left, sort_keys=True) != json.dumps(right, sort_keys=True):
                return False
        elif left != right:
            return False
    return True


def evaluate_shadow_outcomes(
    manifest_path: str | Path = "artifacts/future_test/manifest.lock.json",
    bar_dir: str | Path = "data/shadow/bars",
    signal_dir: str | Path = "artifacts/future_test/signals",
    outcome_dir: str | Path = "artifacts/future_test/outcomes",
    ledger_path: str | Path = "artifacts/future_test/ledger.csv",
    summary_path: str | Path = "artifacts/future_test/evaluation.json",
    shadow_exposure_dir: str | Path = "data/shadow/exposures",
    prediction_dir: str | Path = "artifacts/future_test/predictions",
    addendum_path: str | Path = "artifacts/future_test/protocol.addendum.lock.json",
    audit_chain_path: str | Path = "artifacts/future_test/audit_chain.jsonl",
) -> dict:
    """Mature append-only shadow signals and rebuild the derived equity ledger."""
    verify_frozen_inputs(manifest_path)
    addendum_target = Path(addendum_path)
    chain_target = Path(audit_chain_path)
    if addendum_target.exists():
        verify_protocol_addendum(addendum_target)
    if chain_target.exists():
        verify_audit_chain(chain_target)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    frozen = manifest["frozen_inputs"]
    settings = (
        settings_from_addendum(addendum_target)
        if addendum_target.exists()
        else _shadow_settings(manifest)
    )
    panel = load_shadow_panel(frozen["market"]["path"], bar_dir)
    panel = attach_point_in_time_membership(
        panel, load_membership_history(frozen["membership"]["path"])
    )
    exposure_pieces = [load_exposures(frozen["exposure"]["path"])]
    exposure_pieces.extend(
        load_exposures(path) for path in sorted(Path(shadow_exposure_dir).glob("*.csv"))
    )
    exposures = (
        pd.concat(exposure_pieces, ignore_index=True)
        .sort_values(["date", "symbol"])
        .drop_duplicates(["date", "symbol"], keep="last")
    )
    panel = attach_exposures_asof(panel, exposures)
    dataset = build_dataset(panel, settings.horizon, settings.label_mode)
    latest_date = dataset["date"].max()
    observed_dates = (
        dataset.loc[dataset["date"] >= pd.Timestamp(manifest["evaluation_start"]), "date"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    outcomes_target = Path(outcome_dir)
    outcomes_target.mkdir(parents=True, exist_ok=True)
    signal_paths = sorted(Path(signal_dir).glob("*.csv"))
    previous_weights: dict[str, float] = {}
    equity = benchmark = 1.0
    rows: list[dict] = []
    matured_signals: set[str] = set()
    for signal_path in signal_paths:
        signal_date = pd.Timestamp(signal_path.stem)
        position = observed_dates.index(signal_date) if signal_date in observed_dates else -1
        if position < 0 or position + settings.horizon + 1 >= len(observed_dates):
            break
        current = dataset[(dataset["date"] == signal_date) & dataset["eligible"]].copy()
        stored = pd.read_csv(signal_path, dtype={"symbol": str})
        prediction_path = Path(prediction_dir) / signal_path.name
        if not prediction_path.exists():
            raise RuntimeError(f"缺少信号日全截面预测快照: {prediction_path}")
        predictions = pd.read_csv(prediction_path, dtype={"symbol": str})
        predictions["symbol"] = predictions["symbol"].str.zfill(6)
        ranked = (
            predictions[["symbol", "score"]]
            .merge(current[["symbol", "label"]], on="symbol", how="inner")
            .dropna()
        )
        rank_ic = float(ranked["score"].corr(ranked["label"], method="spearman"))
        if pd.isna(rank_ic):
            rank_ic = None
        exposure_coverage = float(current["float_market_cap"].notna().mean())
        selected_current = current[current["symbol"].isin(stored["symbol"].str.zfill(6))]
        normal_mature = selected_current["future_return"].notna()
        unresolved_exit = normal_mature & selected_current["execution_exit_open"].isna()
        if unresolved_exit.any() and position + settings.horizon + 2 >= len(observed_dates):
            break
        period, previous_weights, detail = _period_accounting(
            stored, current, previous_weights, settings
        )
        equity *= 1 + period["net_return"]
        benchmark *= 1 + period["benchmark_return"]
        computed = {
            "signal_date": str(signal_date.date()),
            **{key: _json_value(value) for key, value in period.items()},
            "excess_return": period["net_return"] - period["benchmark_return"],
            "rank_ic": rank_ic,
            "exposure_coverage": exposure_coverage,
            "equity": equity,
            "benchmark": benchmark,
            "positions": [
                {key: _json_value(value) for key, value in row.items()} for row in detail
            ],
            "execution_authorized": False,
        }
        outcome_path = outcomes_target / f"{signal_date.date()}.json"
        if outcome_path.exists():
            existing = json.loads(outcome_path.read_text(encoding="utf-8"))
            if not _same_outcome(existing, computed):
                raise RuntimeError(f"已固化结果与重建值不一致，拒绝覆盖: {outcome_path}")
        else:
            payload = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), **computed}
            outcome_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
        if chain_target.exists():
            append_audit_record(chain_target, outcome_path, "matured_outcome")
        matured_signals.add(str(signal_date.date()))
        rows.append({key: computed[key] for key in LEDGER_COLUMNS})

    ledger = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    ledger_target = Path(ledger_path)
    ledger_target.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(ledger_target, index=False, encoding="utf-8-sig")
    total_return = float(equity - 1)
    benchmark_return = float(benchmark - 1)
    summary = {
        "as_of": str(pd.Timestamp(latest_date).date()),
        "observed_trading_days": len(observed_dates),
        "signal_snapshots": len(signal_paths),
        "matured_periods": len(rows),
        "pending_signals": len(signal_paths) - len(matured_signals),
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "excess_return": total_return - benchmark_return,
        "latest_matured_signal": rows[-1]["signal_date"] if rows else None,
        "ledger_path": str(ledger_target),
        "execution_authorized": False,
        "audit_chain": verify_audit_chain(chain_target, raise_on_error=False)
        if chain_target.exists()
        else None,
    }
    summary_target = Path(summary_path)
    summary_target.parent.mkdir(parents=True, exist_ok=True)
    summary_target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
