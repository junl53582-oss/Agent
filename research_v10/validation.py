from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stockpilot.data import load_panel
from stockpilot.membership import attach_point_in_time_membership, load_membership_history

from research_v9.data import attach_industry_asof, attach_membership_weight, load_industry_history

from .backtest import MODES, max_drawdown, run_v10_backtest
from .features import build_v10_dataset
from .fundamentals import attach_extended_fundamentals_asof, load_extended_fundamentals
from .research_config import V10Settings
from .research_freeze import verify_research


V6_RANK_IC = 0.025300983903540037
V8_TURNOVER = 0.4197862975359378
V8_COST = 0.0008795827208884614


def _summarize(
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
                "rank_ic_5": float(group["rank_ic_5"].mean()),
                "rank_ic_20": float(group["rank_ic_20"].mean()),
                "top30_precision": float(group["top30_precision"].mean()),
                "selected_excess_return": float(group["selected_excess_return"].mean()),
                "max_drawdown": max_drawdown(group["period_return"]),
            }
        )
    annual = pd.DataFrame(annual_rows)
    sectors = (
        sector_ics.groupby("broad_sector")[["rank_ic_5", "rank_ic_20"]]
        .agg(["mean", "count"])
        if not sector_ics.empty
        else pd.DataFrame()
    )
    if not sectors.empty:
        sectors.columns = ["_".join(column) for column in sectors.columns]
        sectors = sectors.reset_index()
    total = float((1 + equity["period_return"]).prod() - 1)
    benchmark = float((1 + equity["benchmark_return"]).prod() - 1)
    technology = sectors[sectors["broad_sector"] == "technology"] if not sectors.empty else sectors
    metrics = {
        "total_return": total,
        "benchmark_return": benchmark,
        "excess_return": total - benchmark,
        "positive_excess_years": int((annual["excess_return"] > 0).sum()),
        "rank_ic_5": float(equity["rank_ic_5"].mean()),
        "rank_ic_20": float(equity["rank_ic_20"].mean()),
        "top30_precision": float(equity["top30_precision"].mean()),
        "selected_excess_return": float(equity["selected_excess_return"].mean()),
        "technology_rank_ic_5": float(technology["rank_ic_5_mean"].iloc[0])
        if len(technology)
        else float("nan"),
        "technology_rank_ic_20": float(technology["rank_ic_20_mean"].iloc[0])
        if len(technology)
        else float("nan"),
        "max_drawdown": max_drawdown(equity["period_return"]),
        "realized_tracking_error": float(
            equity["excess_period_return"].std(ddof=1) * np.sqrt(252 / 20)
        ),
        "average_one_way_turnover": float(
            (equity["buy_turnover"] + equity["sell_turnover"]).mean() / 2
        ),
        "average_transaction_cost": float(equity["transaction_cost"].mean()),
        "maximum_stock_active_weight": float(equity["maximum_stock_active_weight"].max()),
        "maximum_sector_deviation": float(equity["maximum_sector_deviation"].max()),
        "maximum_ex_ante_tracking_error": float(equity["ex_ante_tracking_error"].max()),
        "technology_enabled_years": sorted(
            int(year)
            for year, group in equity.groupby("test_year")
            if bool(group["technology_enabled"].any())
        ),
    }
    return metrics, annual, sectors


def run_research_v10(
    market_path: str | Path = "data/market_history_v10_hfq.csv",
    membership_path: str | Path = "data/universes/000300/history_v10.csv",
    fundamental_path: str | Path = "data/fundamentals_pit_v10_extended.csv",
    industry_path: str | Path = "data/industry_history_v10.csv",
    settings: V10Settings | None = None,
) -> dict:
    settings = settings or V10Settings()
    settings.ensure_dirs()
    lock = verify_research()
    membership = load_membership_history(membership_path)
    panel = attach_point_in_time_membership(load_panel(market_path), membership)
    panel = attach_membership_weight(panel, membership)
    panel = attach_extended_fundamentals_asof(
        panel, load_extended_fundamentals(fundamental_path)
    )
    panel = attach_industry_asof(panel, load_industry_history(industry_path))
    dataset = build_v10_dataset(panel)
    equity, signals, sector_ics, model_diagnostics = run_v10_backtest(dataset, settings)
    ablation = {}
    for mode in MODES:
        mode_equity = equity[equity["mode"] == mode].copy()
        mode_sector = sector_ics[sector_ics["mode"] == mode].copy()
        metrics, annual, sectors = _summarize(mode_equity, mode_sector)
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
    model_diagnostics.to_csv(
        settings.artifact_dir / "model_validation_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics = ablation["v10_full"]
    core_metrics = ablation["core"]
    gates = {
        "core_tracking_error_lte_2pct": core_metrics["realized_tracking_error"] <= 0.02,
        "cumulative_excess_positive": metrics["excess_return"] > 0,
        "at_least_four_of_six_positive_years": metrics["positive_excess_years"] >= 4,
        "rank_ic_5_at_least_v6": metrics["rank_ic_5"] >= V6_RANK_IC,
        "rank_ic_20_positive": metrics["rank_ic_20"] > 0,
        "top30_excess_positive": metrics["selected_excess_return"] > 0,
        "technology_ic_5_nonnegative": bool(
            np.isfinite(metrics["technology_rank_ic_5"])
            and metrics["technology_rank_ic_5"] >= 0
        ),
        "technology_ic_20_nonnegative": bool(
            np.isfinite(metrics["technology_rank_ic_20"])
            and metrics["technology_rank_ic_20"] >= 0
        ),
        "max_drawdown_not_below_minus_18pct": metrics["max_drawdown"] >= -0.18,
        "realized_tracking_error_lte_6pct": metrics["realized_tracking_error"] <= 0.06,
        "turnover_not_above_v8": metrics["average_one_way_turnover"] <= V8_TURNOVER,
        "cost_not_above_v8": metrics["average_transaction_cost"] <= V8_COST,
        "stock_active_weight_within_cap": metrics["maximum_stock_active_weight"]
        <= settings.maximum_stock_active_weight + 1e-10,
        "sector_neutral": metrics["maximum_sector_deviation"] <= 1e-10,
    }
    retrospective_approved = all(gates.values())
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_v10_validation_complete",
        "plan_lock_sha256": lock["lock_sha256"],
        "environment": lock["environment"],
        "ablation": ablation,
        "metrics": metrics,
        "retrospective_gates": gates,
        "retrospective_approved": retrospective_approved,
        "future_shadow_gate": "pending" if retrospective_approved else "not_started",
        "replacement_approved": False,
        "decision": "start_v10_shadow_keep_v6" if retrospective_approved else "keep_v6",
        "production_model": "V6",
        "execution_authorized": False,
    }
    (settings.artifact_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report

