from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .audit import append_audit_record, verify_audit_chain, verify_protocol_addendum
from .data import load_panel
from .future_test import verify_frozen_inputs


def _observation_dates(
    market_path: str | Path, shadow_bar_dir: str | Path, evaluation_start: str
) -> list[pd.Timestamp]:
    start = pd.Timestamp(evaluation_start)
    dates = set(load_panel(market_path).loc[lambda data: data["date"] >= start, "date"])
    for path in Path(shadow_bar_dir).glob("*.csv"):
        snapshot = pd.read_csv(path, usecols=["date"])
        dates.update(pd.to_datetime(snapshot["date"], errors="coerce").dropna().dt.normalize())
    return sorted(dates)


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min()) if not equity.empty else 0.0


def adjudicate_future_test(
    manifest_path: str | Path = "artifacts/future_test/manifest.lock.json",
    addendum_path: str | Path = "artifacts/future_test/protocol.addendum.lock.json",
    market_path: str | Path = "data/market_history.csv",
    shadow_bar_dir: str | Path = "data/shadow/bars",
    signal_dir: str | Path = "artifacts/future_test/signals",
    ledger_path: str | Path = "artifacts/future_test/ledger.csv",
    status_path: str | Path = "artifacts/future_test/adjudication_status.json",
    decision_path: str | Path = "artifacts/future_test/decision.lock.json",
    audit_chain_path: str | Path = "artifacts/future_test/audit_chain.jsonl",
) -> dict:
    """Apply the predeclared gates once the first frozen observation window has matured."""
    verify_frozen_inputs(manifest_path)
    verify_protocol_addendum(addendum_path)
    verify_audit_chain(audit_chain_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    dates = _observation_dates(market_path, shadow_bar_dir, manifest["evaluation_start"])
    required = int(manifest["minimum_trading_days"])
    cutoff = dates[required - 1] if len(dates) >= required else None
    signal_paths = sorted(Path(signal_dir).glob("*.csv"))
    window_signals = [
        path for path in signal_paths if cutoff is not None and pd.Timestamp(path.stem) <= cutoff
    ]
    ledger_target = Path(ledger_path)
    ledger = pd.read_csv(ledger_target) if ledger_target.exists() else pd.DataFrame()
    if not ledger.empty:
        ledger["signal_date"] = pd.to_datetime(ledger["signal_date"])
    matured = set(ledger["signal_date"].dt.strftime("%Y-%m-%d")) if not ledger.empty else set()
    pending = [path.stem for path in window_signals if path.stem not in matured]
    if len(dates) < required:
        phase = "collecting"
    elif pending:
        phase = "awaiting_label_maturity"
    else:
        phase = "ready"
    report = {
        "phase": phase,
        "observed_trading_days": len(dates),
        "minimum_trading_days": required,
        "evaluation_cutoff": str(cutoff.date()) if cutoff is not None else None,
        "window_signal_snapshots": len(window_signals),
        "pending_window_signals": pending,
        "ready_for_adjudication": phase == "ready",
        "execution_authorized": False,
    }
    if phase == "ready":
        window = ledger[ledger["signal_date"] <= cutoff].copy()
        total_return = float((1 + window["net_return"]).prod() - 1)
        benchmark_return = float((1 + window["benchmark_return"]).prod() - 1)
        window["year"] = pd.to_datetime(window["exit_date"]).dt.year
        yearly_excess = pd.Series(
            {
                year: (1 + group["net_return"]).prod() - (1 + group["benchmark_return"]).prod()
                for year, group in window.groupby("year")
            }
        )
        metrics = {
            "periods": len(window),
            "total_return": total_return,
            "benchmark_return": benchmark_return,
            "excess_return": total_return - benchmark_return,
            "mean_rank_ic": float(window["rank_ic"].mean()),
            "max_drawdown": _max_drawdown(window["net_return"]),
            "positive_excess_year_ratio": float((yearly_excess > 0).mean()),
            "minimum_exposure_coverage": float(window["exposure_coverage"].min()),
        }
        gates = {
            "excess_return": metrics["excess_return"] > 0,
            "mean_rank_ic": metrics["mean_rank_ic"] > 0,
            "max_drawdown": metrics["max_drawdown"] > -0.20,
            "positive_excess_year_ratio": metrics["positive_excess_year_ratio"] >= 0.50,
            "exposure_coverage": metrics["minimum_exposure_coverage"] >= 0.95,
        }
        report.update(
            {
                "metrics": metrics,
                "gates": gates,
                "passed": all(gates.values()),
                "decision": "paper_trade_candidate" if all(gates.values()) else "research_only",
                "note": "即使通过也只允许进入人工审批的模拟盘候选，不授权自动实盘。",
            }
        )
        decision_target = Path(decision_path)
        locked = {"decided_at_utc": datetime.now(timezone.utc).isoformat(), **report}
        if decision_target.exists():
            existing = json.loads(decision_target.read_text(encoding="utf-8"))
            comparable = {key: value for key, value in existing.items() if key != "decided_at_utc"}
            if comparable != report:
                raise RuntimeError(f"最终裁决锁与重建结果不一致，拒绝覆盖: {decision_target}")
        else:
            decision_target.write_text(
                json.dumps(locked, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        append_audit_record(audit_chain_path, decision_target, "final_decision")
    status_target = Path(status_path)
    status_target.parent.mkdir(parents=True, exist_ok=True)
    status_target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
