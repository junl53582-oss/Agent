from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research_v3.fundamentals import attach_fundamentals_asof, load_fundamentals
from research_v4.lock import verify_locked_inputs, verify_plan_lock
from research_v5.backtest import max_drawdown
from research_v5.features import build_v5_dataset
from stockpilot.data import load_panel
from stockpilot.exposure import attach_exposures, load_exposures
from stockpilot.membership import attach_point_in_time_membership, load_membership_history

from .backtest import run_v6_backtest
from .config import PLAN_LOCK_SHA256, V6Settings


def run_research_v6(
    market_path: str | Path = "data/market_history.csv",
    membership_path: str | Path = "data/universes/000300/history.csv",
    exposure_path: str | Path = "data/exposures.csv",
    fundamental_path: str | Path = "data/fundamentals_pit.csv",
    settings: V6Settings | None = None,
    verify_inputs: bool = True,
) -> dict:
    settings = settings or V6Settings()
    settings.ensure_dirs()
    plan = verify_plan_lock(settings.plan_lock_path, PLAN_LOCK_SHA256)
    if verify_inputs:
        verify_locked_inputs(plan)
    panel = load_panel(market_path)
    panel = attach_point_in_time_membership(panel, load_membership_history(membership_path))
    panel = attach_exposures(panel, load_exposures(exposure_path))
    fundamentals = load_fundamentals(fundamental_path)
    panel = attach_fundamentals_asof(panel, fundamentals)
    dataset = build_v5_dataset(panel)
    equity, signals, sector_ics, specs = run_v6_backtest(dataset, settings)
    annual_rows = []
    for year, group in equity.groupby("test_year"):
        strategy = float((1 + group["period_return"]).prod() - 1)
        benchmark = float((1 + group["benchmark_return"]).prod() - 1)
        annual_rows.append({"test_year": int(year), "periods": len(group), "total_return": strategy, "benchmark_return": benchmark, "excess_return": strategy - benchmark, "mean_rank_ic": float(group["rank_ic"].mean()), "max_drawdown": max_drawdown(group["period_return"])})
    annual = pd.DataFrame(annual_rows)
    sector_metrics = sector_ics.groupby("broad_sector")["rank_ic"].agg(["mean", "count"]).reset_index()
    total = float((1 + equity["period_return"]).prod() - 1)
    benchmark = float((1 + equity["benchmark_return"]).prod() - 1)
    metrics = {"periods": len(equity), "total_return": total, "benchmark_return": benchmark, "excess_return": total - benchmark, "mean_rank_ic": float(equity["rank_ic"].mean()), "max_drawdown": max_drawdown(equity["period_return"]), "positive_test_year_ratio": float((annual["excess_return"] > 0).mean()), "nonnegative_broad_sector_ic_ratio": float((sector_metrics["mean"] >= 0).mean()), "average_cash_weight": float(equity["cash_weight"].mean())}
    strict_gates = {"excess_return": metrics["excess_return"] > 0, "mean_rank_ic": metrics["mean_rank_ic"] > 0, "max_drawdown": metrics["max_drawdown"] > -0.20, "positive_test_year_ratio": metrics["positive_test_year_ratio"] >= 0.50, "broad_sector_ic": metrics["nonnegative_broad_sector_ic_ratio"] >= 0.60}
    replacement_gates = {"excess_better_than_v4": metrics["excess_return"] > -0.35492101062415604, "mean_rank_ic": metrics["mean_rank_ic"] > 0, "drawdown_better_than_v4": metrics["max_drawdown"] > -0.30289312680425784, "year_stability_at_least_v4": metrics["positive_test_year_ratio"] >= 1 / 3}
    replacement_approved = all(replacement_gates.values())
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "retrospective_research", "plan_lock_sha256": PLAN_LOCK_SHA256, "settings": {**asdict(settings), "artifact_dir": str(settings.artifact_dir), "plan_lock_path": str(settings.plan_lock_path)}, "metrics": metrics, "strict_gates": strict_gates, "strict_passed": all(strict_gates.values()), "replacement_gates": replacement_gates, "replacement_approved": replacement_approved, "decision": "replace_v4_as_default_research_model" if replacement_approved else "keep_v4", "execution_authorized": False, "warning": "V6替换门槛仅代表相对V4改进，不等于通过实盘门槛。"}
    target = settings.artifact_dir
    equity.to_csv(target / "equity.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(target / "annual_metrics.csv", index=False, encoding="utf-8-sig")
    signals.to_csv(target / "signals.csv", index=False, encoding="utf-8-sig")
    sector_ics.to_csv(target / "sector_ic_daily.csv", index=False, encoding="utf-8-sig")
    sector_metrics.to_csv(target / "sector_metrics.csv", index=False, encoding="utf-8-sig")
    specs.to_csv(target / "v4_stability_specs.csv", index=False, encoding="utf-8-sig")
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report
