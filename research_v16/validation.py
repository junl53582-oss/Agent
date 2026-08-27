from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import MODES, max_drawdown, run_v16_backtest
from .config import V16Settings
from .data import load_v16_dataset
from .freeze import verify_research
from .text_model import EnsembleTextCorpus
from research_v15.features import load_event_documents


V6_RANK_IC = 0.025300983903540037
V8_TURNOVER = 0.4197862975359378
V8_COST = 0.0008795827208884614


def _summarize(equity: pd.DataFrame, sector_ics: pd.DataFrame):
    annual_rows = []
    for year, group in equity.groupby("test_year"):
        strategy = float((1 + group["period_return"]).prod() - 1)
        benchmark = float((1 + group["benchmark_return"]).prod() - 1)
        annual_rows.append({
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
        })
    annual = pd.DataFrame(annual_rows)
    if sector_ics.empty:
        sectors = pd.DataFrame()
    else:
        sectors = sector_ics.groupby("broad_sector")[["rank_ic_5", "rank_ic_20"]].agg(["mean", "count"])
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
        "minimum_raw_event_years": int(equity["raw_event_years"].min()),
        "global_gate_years": sorted(int(year) for year, group in equity.groupby("test_year") if bool(group["global_gate"].any())),
        "technology_gate_years": sorted(int(year) for year, group in equity.groupby("test_year") if bool(group["technology_gate"].any())),
    }
    return metrics, annual, sectors


def run_research_v16(
    event_path: str | Path = "data/event_documents_pit_v15.csv",
    settings: V16Settings | None = None,
):
    settings = settings or V16Settings()
    if asdict(settings) != asdict(V16Settings()):
        raise RuntimeError("V16只能按冻结默认配置运行，不接受事后参数改写")
    if Path(event_path).resolve() != Path("data/event_documents_pit_v15.csv").resolve():
        raise RuntimeError("V16只能使用冻结事件文件")
    settings.ensure_dirs()
    lock = verify_research()
    if (settings.artifact_dir / "report.json").exists():
        raise RuntimeError("V16报告已存在，禁止覆盖")
    with (settings.artifact_dir / "run.started.json").open("x", encoding="utf-8") as handle:
        json.dump({"started_at_utc": datetime.now(timezone.utc).isoformat(), "lock_sha256": lock["lock_sha256"]}, handle, indent=2)
    print("V16 loading frozen market and point-in-time features", flush=True)
    dataset = load_v16_dataset()
    print(f"V16 dataset ready rows={len(dataset)}; loading title corpus", flush=True)
    events = load_event_documents(event_path)
    corpus = EnsembleTextCorpus.build(events, settings)
    equity, signals, sector_ics, diagnostics = run_v16_backtest(dataset, corpus, settings)
    verify_research()
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
    diagnostics.to_csv(settings.artifact_dir / "text_validation_diagnostics.csv", index=False, encoding="utf-8-sig")
    metrics, core = ablation["v16_text_gated"], ablation["core"]
    expected_years = set(settings.test_years)
    gates = {
        "event_document_quality_passed": bool(lock["event_data_quality"]["passed"]),
        "raw_event_years_available": metrics["minimum_raw_event_years"] >= 1,
        "nested_global_validation_passed_each_test_year": set(metrics["global_gate_years"]) == expected_years,
        "nested_technology_validation_passed_each_test_year": set(metrics["technology_gate_years"]) == expected_years,
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
        "sector_neutral": metrics["maximum_sector_deviation"] <= 1e-10,
        "ensemble_excess_not_below_char_replica": ablation["v16_text_ungated"]["excess_return"]
        >= ablation["v15_char_replica"]["excess_return"] - 1e-12,
    }
    approved = all(gates.values())
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_v16_validation_complete",
        "plan_lock_sha256": lock["lock_sha256"],
        "environment": lock["environment"],
        "event_data_quality": lock["event_data_quality"],
        "ablation": ablation,
        "metrics": metrics,
        "retrospective_gates": gates,
        "retrospective_approved": approved,
        "future_shadow_gate": "pending" if approved else "not_started",
        "replacement_approved": False,
        "decision": "start_v16_shadow_keep_v6" if approved else "keep_v6",
        "production_model": "V6",
        "execution_authorized": False,
        "limitations": [
            "Text is announcement titles, not full announcement bodies or pretrained language-model embeddings.",
            "2020-2025 is a retrospective research window reused by earlier versions, not an untouched holdout.",
            "The word-level head depends on jieba segmentation, whose dictionary and algorithm are fixed at the frozen version.",
            "Drawdown is sampled at 20-trading-day rebalances including initial capital; daily mark-to-market drawdown may be worse.",
            "Execution and turnover reuse the frozen legacy backtest accounting; do not interpret as a broker-reconciled executable PnL.",
            "V6 IC and V8 cost gates use historical locked reference constants, not a new same-period live comparison.",
            "The char replica mode reproduces the V15 ungated line under the V16 accounting stack for attribution only."
        ],
    }
    (settings.artifact_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
