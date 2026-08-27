from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stockpilot.data import load_panel
from stockpilot.membership import attach_point_in_time_membership, load_membership_history
from stockpilot.portfolio import turnover
from stockpilot.trading import add_execution_columns

from research_v9.data import attach_membership_weight

from .config import V10AuditSettings


def build_core_panel(market: pd.DataFrame, membership: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    data = attach_point_in_time_membership(market, membership)
    data = attach_membership_weight(data, membership).sort_values(["symbol", "date"])
    grouped = data.groupby("symbol", group_keys=False)
    data["entry_open"] = grouped["open"].shift(-1)
    data["exit_open"] = grouped["open"].shift(-(horizon + 1))
    data["label_end_date"] = grouped["date"].shift(-(horizon + 1))
    data["future_return"] = data["exit_open"] / data["entry_open"] - 1
    return add_execution_columns(data, horizon)


def _period_core(
    current: pd.DataFrame,
    previous: dict[str, float],
    buy_rate: float,
    sell_rate: float,
) -> tuple[dict[str, float], dict]:
    valid = current[
        current["in_universe"].fillna(False)
        & current["benchmark_weight"].gt(0)
        & current["future_return"].notna()
    ].copy()
    total = float(valid["benchmark_weight"].sum())
    desired = (
        (valid.set_index("symbol")["benchmark_weight"] / total).to_dict()
        if total > 0
        else {}
    )
    lookup = valid.set_index("symbol")
    executed: dict[str, float] = {}
    realized: dict[str, float] = {}
    for symbol, weight in desired.items():
        row = lookup.loc[symbol]
        continuing = symbol in previous
        value = (
            float(row["execution_exit_open"] / row["entry_open"] - 1)
            if continuing
            else float(row["execution_return"])
        )
        if (continuing or bool(row["entry_tradable"])) and np.isfinite(value):
            executed[symbol] = float(weight)
            realized[symbol] = value
    benchmark = float((valid["benchmark_weight"] * valid["future_return"]).sum() / total)
    gross = float(sum(weight * realized[symbol] for symbol, weight in executed.items()))
    buys, sells = turnover(previous, executed)
    cost = buys * buy_rate + sells * sell_rate
    return executed, {
        "gross_return": gross,
        "net_return": gross - cost,
        "benchmark_return": benchmark,
        "gross_excess": gross - benchmark,
        "net_excess": gross - cost - benchmark,
        "buy_turnover": buys,
        "sell_turnover": sells,
        "transaction_cost": cost,
        "cash_weight": 1 - sum(executed.values()),
        "target_symbols": len(desired),
        "executed_symbols": len(executed),
        "weight_coverage": sum(executed.values()),
    }


def run_core_audit(
    market_path: str | Path = "data/market_history_v10.csv",
    membership_path: str | Path = "data/universes/000300/history_v9.csv",
    settings: V10AuditSettings | None = None,
) -> dict:
    from .freeze import verify_audit_lock

    settings = settings or V10AuditSettings()
    settings.ensure_dirs()
    lock = verify_audit_lock()
    membership = load_membership_history(membership_path)
    panel = build_core_panel(load_panel(market_path), membership, settings.horizon)
    panel["date"] = pd.to_datetime(panel["date"])
    scope = panel[
        panel["date"].dt.year.isin(settings.test_years)
        & panel["in_universe"].fillna(False)
        & panel["future_return"].notna()
    ]
    dates = scope["date"].drop_duplicates().sort_values().reset_index(drop=True)
    previous: dict[str, float] = {}
    rows = []
    buy_rate = settings.fee_rate + settings.slippage
    sell_rate = settings.fee_rate + settings.slippage + settings.stamp_duty
    for date in dates.iloc[:: settings.rebalance_every]:
        previous, result = _period_core(
            scope[scope["date"] == date], previous, buy_rate, sell_rate
        )
        rows.append({"date": date, "test_year": int(date.year), **result})
    periods = pd.DataFrame(rows)
    annualized_gross_te = float(periods["gross_excess"].std(ddof=1) * np.sqrt(252 / 5))
    annualized_net_te = float(periods["net_excess"].std(ddof=1) * np.sqrt(252 / 5))
    gross_return = float((1 + periods["gross_return"]).prod() - 1)
    net_return = float((1 + periods["net_return"]).prod() - 1)
    benchmark_return = float((1 + periods["benchmark_return"]).prod() - 1)
    metrics = {
        "gross_return": gross_return,
        "net_return": net_return,
        "benchmark_return": benchmark_return,
        "gross_excess_return": gross_return - benchmark_return,
        "net_excess_return": net_return - benchmark_return,
        "annualized_gross_tracking_error": annualized_gross_te,
        "annualized_net_tracking_error": annualized_net_te,
        "average_cash_weight": float(periods["cash_weight"].mean()),
        "minimum_weight_coverage": float(periods["weight_coverage"].min()),
        "average_one_way_turnover": float(
            (periods["buy_turnover"] + periods["sell_turnover"]).mean() / 2
        ),
        "average_transaction_cost": float(periods["transaction_cost"].mean()),
    }
    gate = annualized_net_te <= settings.maximum_annualized_tracking_error
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_core_replication_audit_complete",
        "audit_lock_sha256": lock["lock_sha256"],
        "metrics": metrics,
        "gate": {
            "annualized_net_tracking_error_lte_2pct": gate,
        },
        "passed": gate,
        "decision": "continue_v10" if gate else "stop_and_diagnose_replication",
    }
    periods.to_csv(
        settings.artifact_dir / "core_replication_periods.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (settings.artifact_dir / "core_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
