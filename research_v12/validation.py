from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_v10.fundamentals import attach_extended_fundamentals_asof, load_extended_fundamentals
from research_v9.data import attach_industry_asof, attach_membership_weight, load_industry_history
from stockpilot.data import load_panel
from stockpilot.membership import attach_point_in_time_membership, load_membership_history

from .backtest import MODES, max_drawdown, run_v12_backtest
from .config import V12Settings
from .features import build_v12_dataset
from .freeze import verify_research


V6_RANK_IC = 0.025300983903540037
V8_TURNOVER = 0.4197862975359378
V8_COST = 0.0008795827208884614


def _summarize(equity: pd.DataFrame, sector_ics: pd.DataFrame):
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
                "selected_net_marginal": float(group["selected_net_marginal"].mean()),
                "max_drawdown": max_drawdown(group["period_return"]),
                "average_equity_exposure": float(group["equity_exposure"].mean()),
            }
        )
    annual = pd.DataFrame(annual_rows)
    sectors = (
        sector_ics.groupby("broad_sector")[["rank_ic_5", "rank_ic_20"]].agg(["mean", "count"])
        if not sector_ics.empty
        else pd.DataFrame()
    )
    if not sectors.empty:
        sectors.columns = ["_".join(column) for column in sectors.columns]
        sectors = sectors.reset_index()
    technology = sectors[sectors["broad_sector"] == "technology"] if not sectors.empty else sectors
    total = float((1 + equity["period_return"]).prod() - 1)
    benchmark = float((1 + equity["benchmark_return"]).prod() - 1)
    metrics = {
        "total_return": total,
        "benchmark_return": benchmark,
        "excess_return": total - benchmark,
        "positive_excess_years": int((annual["excess_return"] > 0).sum()),
        "rank_ic_5": float(equity["rank_ic_5"].mean()),
        "rank_ic_20": float(equity["rank_ic_20"].mean()),
        "top30_precision": float(equity["top30_precision"].mean()),
        "selected_excess_return": float(equity["selected_excess_return"].mean()),
        "selected_net_marginal": float(equity["selected_net_marginal"].mean()),
        "technology_rank_ic_5": float(technology["rank_ic_5_mean"].iloc[0]) if len(technology) else float("nan"),
        "technology_rank_ic_20": float(technology["rank_ic_20_mean"].iloc[0]) if len(technology) else float("nan"),
        "max_drawdown": max_drawdown(equity["period_return"]),
        "realized_tracking_error": float(equity["excess_period_return"].std(ddof=1) * np.sqrt(252 / 20)),
        "average_one_way_turnover": float((equity["buy_turnover"] + equity["sell_turnover"]).mean() / 2),
        "average_transaction_cost": float(equity["transaction_cost"].mean()),
        "maximum_stock_active_weight": float(equity["maximum_stock_active_weight"].max()),
        "maximum_sector_deviation": float(equity["maximum_sector_deviation"].max()),
        "maximum_ex_ante_tracking_error": float(equity["ex_ante_tracking_error"].max()),
        "average_equity_exposure": float(equity["equity_exposure"].mean()),
        "risk_budget_periods": int((equity["risk_regime"] == "risk_budget").sum()),
        "global_gate_years": sorted(int(year) for year, group in equity.groupby("test_year") if bool(group["global_gate"].any())),
        "technology_gate_years": sorted(int(year) for year, group in equity.groupby("test_year") if bool(group["technology_gate"].any())),
    }
    return metrics, annual, sectors


def run_research_v12(
    market_path: str | Path = "data/market_history_v10_hfq.csv",
    membership_path: str | Path = "data/universes/000300/history_v10.csv",
    fundamental_path: str | Path = "data/fundamentals_pit_v10_extended.csv",
    industry_path: str | Path = "data/industry_history_v10.csv",
    settings: V12Settings | None = None,
) -> dict:
    settings = settings or V12Settings()
    settings.ensure_dirs()
    lock = verify_research()
    membership = load_membership_history(membership_path)
    panel = attach_point_in_time_membership(load_panel(market_path), membership)
    panel = attach_membership_weight(panel, membership)
    panel = attach_extended_fundamentals_asof(panel, load_extended_fundamentals(fundamental_path))
    panel = attach_industry_asof(panel, load_industry_history(industry_path))
    dataset = build_v12_dataset(panel, settings)
    equity, signals, sector_ics, validation = run_v12_backtest(dataset, settings)
    ablation = {}
    for mode in MODES:
        mode_equity = equity[equity["mode"] == mode].copy()
        mode_sector = sector_ics[sector_ics["mode"] == mode].copy()
        metrics, annual, sectors = _summarize(mode_equity, mode_sector)
        ablation[mode] = metrics
        mode_equity.to_csv(settings.artifact_dir / f"{mode}_equity.csv", index=False, encoding="utf-8-sig")
        annual.to_csv(settings.artifact_dir / f"{mode}_annual_metrics.csv", index=False, encoding="utf-8-sig")
        sectors.to_csv(settings.artifact_dir / f"{mode}_sector_metrics.csv", index=False, encoding="utf-8-sig")
        signals[signals["mode"] == mode].to_csv(settings.artifact_dir / f"{mode}_signals.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(settings.artifact_dir / "portfolio_validation_diagnostics.csv", index=False, encoding="utf-8-sig")
    metrics = ablation["v12_risk_budget"]
    core = ablation["core"]
    gates = {
        "core_tracking_error_lte_2pct": core["realized_tracking_error"] <= 0.02,
        "cumulative_excess_positive": metrics["excess_return"] > 0,
        "at_least_four_of_six_positive_years": metrics["positive_excess_years"] >= 4,
        "rank_ic_5_at_least_v6": metrics["rank_ic_5"] >= V6_RANK_IC,
        "rank_ic_20_positive": metrics["rank_ic_20"] > 0,
        "selected_net_marginal_positive": metrics["selected_net_marginal"] > 0,
        "technology_ic_5_nonnegative": bool(np.isfinite(metrics["technology_rank_ic_5"]) and metrics["technology_rank_ic_5"] >= 0),
        "technology_ic_20_nonnegative": bool(np.isfinite(metrics["technology_rank_ic_20"]) and metrics["technology_rank_ic_20"] >= 0),
        "max_drawdown_not_below_minus_18pct": metrics["max_drawdown"] >= -0.18,
        "realized_tracking_error_lte_12pct": metrics["realized_tracking_error"] <= 0.12,
        "turnover_not_above_v8": metrics["average_one_way_turnover"] <= V8_TURNOVER,
        "cost_not_above_v8": metrics["average_transaction_cost"] <= V8_COST,
        "stock_active_weight_within_cap": metrics["maximum_stock_active_weight"] <= settings.maximum_stock_active_weight + 1e-10,
        "sector_neutral_before_cash_overlay": metrics["maximum_sector_deviation"] <= 1e-10,
    }
    approved = all(gates.values())
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_v12_validation_complete",
        "plan_lock_sha256": lock["lock_sha256"],
        "environment": lock["environment"],
        "ablation": ablation,
        "metrics": metrics,
        "retrospective_gates": gates,
        "retrospective_approved": approved,
        "future_shadow_gate": "pending" if approved else "not_started",
        "replacement_approved": False,
        "decision": "start_v12_shadow_keep_v6" if approved else "keep_v6",
        "production_model": "V6",
        "execution_authorized": False,
    }
    (settings.artifact_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

