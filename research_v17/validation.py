from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .backtest import MODES, max_drawdown, run_v17_backtest
from .config import V17Settings
from .freeze import verify_research


V16_EQUITY = "artifacts/research_v16/v16_text_ungated_equity.csv"


def _summarize(equity: pd.DataFrame) -> dict:
    held = equity[equity["in_market"]].copy()
    total = float((1 + equity["period_return"]).prod() - 1)
    benchmark = float((1 + equity["benchmark_return"]).prod() - 1)
    win_rate = float((held["period_return"] > 0).mean()) if len(held) else float("nan")
    annual = {}
    for year, group in equity.groupby("test_year"):
        strategy = float((1 + group["period_return"]).prod() - 1)
        bench = float((1 + group["benchmark_return"]).prod() - 1)
        annual[int(year)] = {
            "return": strategy,
            "benchmark": bench,
            "excess": strategy - bench,
            "periods_held": int(group["in_market"].sum()),
            "periods_total": int(len(group)),
        }
    positive_years = sum(1 for value in annual.values() if value["excess"] > 0)
    return {
        "total_return": total,
        "benchmark_return": benchmark,
        "excess_return": total - benchmark,
        "win_rate": win_rate,
        "periods_held": int(equity["in_market"].sum()),
        "periods_total": int(len(equity)),
        "hold_ratio": float(equity["in_market"].mean()),
        "max_drawdown": max_drawdown(equity["period_return"]),
        "positive_excess_years": positive_years,
        "annual": annual,
    }


def run_research_v17(settings: V17Settings | None = None) -> dict:
    settings = settings or V17Settings()
    if asdict(settings) != asdict(V17Settings()):
        raise RuntimeError("V17只能按冻结默认配置运行，不接受事后参数改写")
    settings.ensure_dirs()
    lock = verify_research()
    if (settings.artifact_dir / "report.json").exists():
        raise RuntimeError("V17报告已存在，禁止覆盖")
    with (settings.artifact_dir / "run.started.json").open("x", encoding="utf-8") as handle:
        json.dump({"started_at_utc": datetime.now(timezone.utc).isoformat(), "lock_sha256": lock["lock_sha256"]}, handle, indent=2)

    equity = run_v17_backtest(V16_EQUITY, settings)
    verify_research()

    ablation = {}
    for mode in MODES:
        mode_equity = equity[equity["mode"] == mode].copy()
        metrics = _summarize(mode_equity)
        ablation[mode] = metrics
        mode_equity.to_csv(settings.artifact_dir / f"{mode}_equity.csv", index=False, encoding="utf-8-sig")

    ungated = ablation["v16_ungated"]
    timing = ablation["v17_timing"]

    gates = {
        "timing_win_rate_at_least_plus_10pct": timing["win_rate"] >= ungated["win_rate"] + 0.10,
        "timing_reduces_drawdown": timing["max_drawdown"] >= ungated["max_drawdown"],
        "timing_excess_positive": timing["excess_return"] > 0,
        "timing_not_fewer_positive_years": timing["positive_excess_years"] >= ungated["positive_excess_years"],
        "timing_holds_at_least_30pct": timing["hold_ratio"] >= 0.30,
        "timing_return_positive": timing["total_return"] > 0,
    }
    approved = all(gates.values())

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_v17_timing_validation_complete",
        "plan_lock_sha256": lock["lock_sha256"],
        "environment": lock["environment"],
        "source_equity": V16_EQUITY,
        "ablation": ablation,
        "timing_metrics": timing,
        "gates": gates,
        "approved": approved,
        "decision": "v17_timing_replaces_ungated" if approved else "keep_v6",
        "execution_authorized": False,
        "limitations": [
            "Timing signal is the prior period benchmark return, discovered on the 2020-2025 retrospective window already used by V3-V16.",
            "Only 73 rebalance periods; a 94 percent win rate has a wide confidence interval.",
            "A positive result still requires the future shadow protocol before any live use.",
            "Timing reduces risk by going to cash, not by improving stock selection."
        ],
    }
    (settings.artifact_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
