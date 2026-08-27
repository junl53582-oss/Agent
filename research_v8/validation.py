from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research_v3.fundamentals import attach_fundamentals_asof, load_fundamentals
from research_v4.lock import verify_locked_inputs, verify_plan_lock
from stockpilot.data import load_panel
from stockpilot.exposure import attach_exposures, load_exposures
from stockpilot.membership import attach_point_in_time_membership, load_membership_history

from .backtest import max_drawdown, run_v8_backtest
from .config import PLAN_LOCK_SHA256, V8Settings
from .features import build_v8_dataset


def _summarize(equity: pd.DataFrame, sector_ics: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
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
            }
        )
    annual = pd.DataFrame(annual_rows)
    sector_metrics = sector_ics.groupby("broad_sector")["rank_ic"].agg(["mean", "count"]).reset_index()
    total = float((1 + equity["period_return"]).prod() - 1)
    benchmark = float((1 + equity["benchmark_return"]).prod() - 1)
    metrics = {
        "total_return": total,
        "benchmark_return": benchmark,
        "excess_return": total - benchmark,
        "mean_rank_ic": float(equity["rank_ic"].mean()),
        "max_drawdown": max_drawdown(equity["period_return"]),
        "positive_test_year_ratio": float((annual["excess_return"] > 0).mean()),
        "nonnegative_sector_ic_ratio": float((sector_metrics["mean"] >= 0).mean()),
        "average_one_way_turnover": float(
            (equity["buy_turnover"] + equity["sell_turnover"]).mean() / 2
        ),
        "average_transaction_cost": float(equity["transaction_cost"].mean()),
    }
    return metrics, annual, sector_metrics


def run_research_v8(
    market_path: str | Path = "data/market_history.csv",
    membership_path: str | Path = "data/universes/000300/history.csv",
    exposure_path: str | Path = "data/exposures.csv",
    fundamental_path: str | Path = "data/fundamentals_pit.csv",
    settings: V8Settings | None = None,
) -> dict:
    settings = settings or V8Settings()
    settings.ensure_dirs()
    plan = verify_plan_lock(settings.plan_lock_path, PLAN_LOCK_SHA256)
    verify_locked_inputs(plan)
    panel = attach_point_in_time_membership(
        load_panel(market_path), load_membership_history(membership_path)
    )
    panel = attach_exposures(panel, load_exposures(exposure_path))
    panel = attach_fundamentals_asof(panel, load_fundamentals(fundamental_path))
    dataset = build_v8_dataset(panel)
    results = {}
    for mode in ("v6_buffer", "v8_full"):
        equity, signals, sector_ics = run_v8_backtest(dataset, settings, mode)
        metrics, annual, sector_metrics = _summarize(equity, sector_ics)
        results[mode] = metrics
        equity.to_csv(settings.artifact_dir / f"{mode}_equity.csv", index=False, encoding="utf-8-sig")
        annual.to_csv(settings.artifact_dir / f"{mode}_annual_metrics.csv", index=False, encoding="utf-8-sig")
        sector_metrics.to_csv(settings.artifact_dir / f"{mode}_sector_metrics.csv", index=False, encoding="utf-8-sig")
        signals.to_csv(settings.artifact_dir / f"{mode}_signals.csv", index=False, encoding="utf-8-sig")
    metrics = results["v8_full"]
    strict = {
        "excess": metrics["excess_return"] > 0,
        "ic": metrics["mean_rank_ic"] > 0,
        "drawdown": metrics["max_drawdown"] > -0.20,
        "years": metrics["positive_test_year_ratio"] >= 0.50,
        "sectors": metrics["nonnegative_sector_ic_ratio"] >= 0.60,
    }
    replacement = {
        "excess_better_v6": metrics["excess_return"] > -0.026184784718117804,
        "ic_at_least_v6": metrics["mean_rank_ic"] >= 0.025300983903540037,
        "drawdown_at_least_v6": metrics["max_drawdown"] >= -0.18915470153920222,
        "years_at_least_v6": metrics["positive_test_year_ratio"] >= 2 / 3,
        "turnover_below_lock": metrics["average_one_way_turnover"] < 0.50,
    }
    approved = all(replacement.values())
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "retrospective_research",
        "plan_lock_sha256": PLAN_LOCK_SHA256,
        "data_manifest": "artifacts/research_v8/data_manifest.json",
        "ablation": results,
        "metrics": metrics,
        "strict_gates": strict,
        "strict_passed": all(strict.values()),
        "replacement_gates": replacement,
        "replacement_approved": approved,
        "decision": "replace_v6" if approved else "keep_v6",
        "execution_authorized": False,
    }
    (settings.artifact_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report

