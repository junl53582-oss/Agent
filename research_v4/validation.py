from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research_v3.fundamentals import attach_fundamentals_asof, load_fundamentals
from stockpilot.data import load_panel
from stockpilot.exposure import attach_exposures, load_exposures
from stockpilot.membership import attach_point_in_time_membership, load_membership_history

from .backtest import max_drawdown, run_v4_backtest
from .config import PLAN_LOCK_SHA256, V4Settings
from .features import build_v4_dataset
from .lock import verify_locked_inputs, verify_plan_lock


def _annual_metrics(equity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in equity.groupby("test_year", sort=True):
        strategy = float((1 + group["period_return"]).prod() - 1)
        benchmark = float((1 + group["benchmark_return"]).prod() - 1)
        rows.append(
            {
                "test_year": int(year),
                "periods": len(group),
                "total_return": strategy,
                "benchmark_return": benchmark,
                "excess_return": strategy - benchmark,
                "mean_rank_ic": float(group["rank_ic"].mean()),
                "max_drawdown": max_drawdown(group["period_return"]),
            }
        )
    return pd.DataFrame(rows)


def run_research_v4(
    market_path: str | Path = "data/market_history.csv",
    membership_path: str | Path = "data/universes/000300/history.csv",
    exposure_path: str | Path = "data/exposures.csv",
    fundamental_path: str | Path = "data/fundamentals_pit.csv",
    settings: V4Settings | None = None,
    verify_inputs: bool = True,
) -> dict:
    settings = settings or V4Settings()
    settings.ensure_dirs()
    plan = verify_plan_lock(settings.plan_lock_path)
    if verify_inputs:
        verify_locked_inputs(plan)
    panel = load_panel(market_path)
    panel = attach_point_in_time_membership(panel, load_membership_history(membership_path))
    panel = attach_exposures(panel, load_exposures(exposure_path))
    fundamentals = load_fundamentals(fundamental_path)
    panel = attach_fundamentals_asof(panel, fundamentals)
    dataset = build_v4_dataset(panel)
    equity, specs, diagnostics, signals = run_v4_backtest(dataset, settings)
    if equity.empty:
        raise RuntimeError("V4没有形成可评估的年度测试折")
    annual = _annual_metrics(equity)
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
        "average_cash_weight": float(equity["cash_weight"].mean()),
    }
    gates = {
        "excess_return": metrics["excess_return"] > 0,
        "mean_rank_ic": metrics["mean_rank_ic"] > 0,
        "max_drawdown": metrics["max_drawdown"] > -0.20,
        "positive_test_year_ratio": metrics["positive_test_year_ratio"] >= 0.50,
    }
    passed = all(gates.values())
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "retrospective_research",
        "plan_lock_sha256": PLAN_LOCK_SHA256,
        "settings": {
            **asdict(settings),
            "artifact_dir": str(settings.artifact_dir),
            "plan_lock_path": str(settings.plan_lock_path),
        },
        "fundamental_rows": len(fundamentals),
        "fundamental_symbols": int(fundamentals["symbol"].nunique()),
        "fundamental_pit_violations": int(
            (fundamentals["available_date"] < fundamentals["report_date"]).sum()
        ),
        "test_years": annual["test_year"].tolist(),
        "metrics": metrics,
        "gates": gates,
        "passed": passed,
        "decision": "candidate_for_new_future_protocol" if passed else "continue_research",
        "warning": "历史区间已经被观察；即使通过也必须另开未来影子验证，不能授权交易。",
    }
    target = settings.artifact_dir
    equity.to_csv(target / "equity.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(target / "annual_metrics.csv", index=False, encoding="utf-8-sig")
    specs.to_csv(target / "factor_specs.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(target / "factor_ic_diagnostics.csv", index=False, encoding="utf-8-sig")
    signals.to_csv(target / "signals.csv", index=False, encoding="utf-8-sig")
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return report
