from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research_v3.fundamentals import attach_fundamentals_asof, load_fundamentals
from research_v4.lock import verify_locked_inputs, verify_plan_lock
from stockpilot.data import load_panel
from stockpilot.exposure import attach_exposures, load_exposures
from stockpilot.membership import attach_point_in_time_membership, load_membership_history

from .backtest import max_drawdown, run_v5_backtest
from .config import PLAN_LOCK_SHA256, V5Settings
from .features import build_v5_dataset


def run_research_v5(
    market_path: str | Path = "data/market_history.csv",
    membership_path: str | Path = "data/universes/000300/history.csv",
    exposure_path: str | Path = "data/exposures.csv",
    fundamental_path: str | Path = "data/fundamentals_pit.csv",
    settings: V5Settings | None = None,
    verify_inputs: bool = True,
) -> dict:
    settings = settings or V5Settings()
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
    equity, signals, sector_ics, diagnostics = run_v5_backtest(dataset, settings)
    if equity.empty:
        raise RuntimeError("V5没有形成可评估测试折")
    annual_rows = []
    for year, group in equity.groupby("test_year"):
        strategy = float((1 + group["period_return"]).prod() - 1)
        benchmark = float((1 + group["benchmark_return"]).prod() - 1)
        annual_rows.append(
            {"test_year": int(year), "periods": len(group), "total_return": strategy, "benchmark_return": benchmark, "excess_return": strategy - benchmark, "mean_rank_ic": float(group["rank_ic"].mean()), "max_drawdown": max_drawdown(group["period_return"])}
        )
    annual = pd.DataFrame(annual_rows)
    sector_metrics = sector_ics.groupby("broad_sector")["rank_ic"].agg(["mean", "count"]).reset_index()
    sector_nonnegative = float((sector_metrics["mean"] >= 0).mean())
    total = float((1 + equity["period_return"]).prod() - 1)
    benchmark = float((1 + equity["benchmark_return"]).prod() - 1)
    metrics = {
        "periods": len(equity),
        "total_return": total,
        "benchmark_return": benchmark,
        "excess_return": total - benchmark,
        "mean_rank_ic": float(equity["rank_ic"].mean()),
        "max_drawdown": max_drawdown(equity["period_return"]),
        "positive_test_year_ratio": float((annual["excess_return"] > 0).mean()),
        "nonnegative_broad_sector_ic_ratio": sector_nonnegative,
        "average_cash_weight": float(equity["cash_weight"].mean()),
    }
    gates = {
        "excess_return": metrics["excess_return"] > 0,
        "mean_rank_ic": metrics["mean_rank_ic"] > 0,
        "max_drawdown": metrics["max_drawdown"] > -0.20,
        "positive_test_year_ratio": metrics["positive_test_year_ratio"] >= 0.50,
        "broad_sector_ic": metrics["nonnegative_broad_sector_ic_ratio"] >= 0.60,
    }
    passed = all(gates.values())
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "retrospective_research",
        "plan_lock_sha256": PLAN_LOCK_SHA256,
        "settings": {**asdict(settings), "artifact_dir": str(settings.artifact_dir), "plan_lock_path": str(settings.plan_lock_path)},
        "dimensions_implemented": list(plan["dimensions"]),
        "known_missing_dimensions": plan["known_missing_dimensions"],
        "fundamental_pit_violations": int((fundamentals["available_date"] < fundamentals["report_date"]).sum()),
        "metrics": metrics,
        "gates": gates,
        "passed": passed,
        "decision": "candidate_for_new_future_protocol" if passed else "continue_research",
        "warning": "历史区间已被观察；即使通过也只能建立新的未来影子测试，不能授权交易。",
    }
    target = settings.artifact_dir
    equity.to_csv(target / "equity.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(target / "annual_metrics.csv", index=False, encoding="utf-8-sig")
    signals.to_csv(target / "signals.csv", index=False, encoding="utf-8-sig")
    sector_ics.to_csv(target / "sector_ic_daily.csv", index=False, encoding="utf-8-sig")
    sector_metrics.to_csv(target / "sector_metrics.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(target / "model_coefficients.csv", index=False, encoding="utf-8-sig")
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report
