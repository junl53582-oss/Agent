from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_v3.fundamentals import attach_fundamentals_asof, load_fundamentals
from stockpilot.data import load_panel
from stockpilot.membership import attach_point_in_time_membership, load_membership_history

from .backtest import MODES, max_drawdown, run_v9_backtest
from .config import V9Settings
from .data import (
    attach_industry_asof,
    attach_membership_weight,
    load_industry_history,
)
from .features import build_v9_dataset
from .freeze import LOCK_PATH, verify


V6_RANK_IC = 0.025300983903540037
V8_TURNOVER = 0.4197862975359378
V8_COST = 0.0008795827208884614


def _metrics(
    equity: pd.DataFrame, sector_ics: pd.DataFrame
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    annual_rows = []
    for year, group in equity.groupby("test_year"):
        strategy = float((1 + group["period_return"]).prod() - 1)
        benchmark = float((1 + group["benchmark_return"]).prod() - 1)
        annual_rows.append(
            {
                "test_year": int(year),
                "total_return": strategy,
                "benchmark_return": benchmark,
                "excess_return": strategy - benchmark,
                "mean_rank_ic": float(group["rank_ic"].mean()),
                "max_drawdown": max_drawdown(group["period_return"]),
                "average_one_way_turnover": float(
                    (group["buy_turnover"] + group["sell_turnover"]).mean() / 2
                ),
                "average_transaction_cost": float(group["transaction_cost"].mean()),
            }
        )
    annual = pd.DataFrame(annual_rows)
    sectors = (
        sector_ics.groupby("broad_sector")["rank_ic"].agg(["mean", "count"]).reset_index()
        if not sector_ics.empty
        else pd.DataFrame(columns=["broad_sector", "mean", "count"])
    )
    total = float((1 + equity["period_return"]).prod() - 1)
    benchmark = float((1 + equity["benchmark_return"]).prod() - 1)
    technology = sectors.loc[sectors["broad_sector"] == "technology", "mean"]
    tracking_error = float(
        equity["excess_period_return"].std(ddof=1) * np.sqrt(252 / 5)
    )
    metrics = {
        "total_return": total,
        "benchmark_return": benchmark,
        "excess_return": total - benchmark,
        "positive_excess_years": int((annual["excess_return"] > 0).sum()),
        "mean_rank_ic": float(equity["rank_ic"].mean()),
        "technology_rank_ic": float(technology.iloc[0]) if len(technology) else float("nan"),
        "max_drawdown": max_drawdown(equity["period_return"]),
        "average_one_way_turnover": float(
            (equity["buy_turnover"] + equity["sell_turnover"]).mean() / 2
        ),
        "average_transaction_cost": float(equity["transaction_cost"].mean()),
        "annualized_tracking_error": tracking_error,
    }
    return metrics, annual, sectors


def run_research_v9(
    market_path: str | Path = "data/market_history_v9.csv",
    membership_path: str | Path = "data/universes/000300/history_v9.csv",
    fundamental_path: str | Path = "data/fundamentals_pit_v9.csv",
    industry_path: str | Path = "data/industry_history_v9.csv",
    settings: V9Settings | None = None,
) -> dict:
    settings = settings or V9Settings()
    settings.ensure_dirs()
    lock = verify()
    membership = load_membership_history(membership_path)
    panel = attach_point_in_time_membership(load_panel(market_path), membership)
    panel = attach_membership_weight(panel, membership)
    panel = attach_fundamentals_asof(panel, load_fundamentals(fundamental_path))
    panel = attach_industry_asof(panel, load_industry_history(industry_path))
    dataset = build_v9_dataset(panel)
    equity, signals, sector_ics = run_v9_backtest(dataset, settings)

    ablation = {}
    for mode in MODES:
        mode_equity = equity[equity["mode"] == mode].copy()
        mode_sectors = sector_ics[sector_ics["mode"] == mode].copy()
        metrics, annual, sectors = _metrics(mode_equity, mode_sectors)
        ablation[mode] = metrics
        mode_equity.to_csv(
            settings.artifact_dir / f"{mode}_equity.csv", index=False, encoding="utf-8-sig"
        )
        annual.to_csv(
            settings.artifact_dir / f"{mode}_annual_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        sectors.to_csv(
            settings.artifact_dir / f"{mode}_sector_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        signals[signals["mode"] == mode].to_csv(
            settings.artifact_dir / f"{mode}_signals.csv",
            index=False,
            encoding="utf-8-sig",
        )

    metrics = ablation["v9_full"]
    retrospective_gates = {
        "cumulative_excess_positive": metrics["excess_return"] > 0,
        "at_least_four_of_six_positive_years": metrics["positive_excess_years"] >= 4,
        "rank_ic_at_least_v6": metrics["mean_rank_ic"] >= V6_RANK_IC,
        "technology_ic_nonnegative": bool(
            np.isfinite(metrics["technology_rank_ic"])
            and metrics["technology_rank_ic"] >= 0
        ),
        "max_drawdown_not_below_minus_18pct": metrics["max_drawdown"] >= -0.18,
        "turnover_not_above_v8": metrics["average_one_way_turnover"] <= V8_TURNOVER,
        "cost_not_above_v8": metrics["average_transaction_cost"] <= V8_COST,
    }
    historical_approved = all(retrospective_gates.values())
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "retrospective_frozen_validation_complete",
        "plan_lock": LOCK_PATH,
        "plan_lock_sha256": Path("artifacts/research_v9/plan.lock.sha256")
        .read_text(encoding="utf-8")
        .strip(),
        "environment": lock["environment"],
        "ablation": ablation,
        "metrics": metrics,
        "retrospective_gates": retrospective_gates,
        "retrospective_approved": historical_approved,
        "future_shadow_gate": "pending" if historical_approved else "not_started",
        "replacement_approved": False,
        "decision": "start_v9_shadow_keep_v6" if historical_approved else "keep_v6",
        "production_model": "V6",
        "execution_authorized": False,
    }
    (settings.artifact_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report

