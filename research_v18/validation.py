from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_v15.features import load_event_documents

from .backtest import MODES, max_drawdown, run_v18_backtest
from .config import V18Settings
from .data import load_v18_dataset
from .embed import build_embeddings
from .freeze import verify_research


V16_UNGATED_EXCESS = 0.0592
V16_UNGATED_IC5 = -0.0251
V16_UNGATED_IC20 = -0.0168


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
            "max_drawdown": max_drawdown(group["period_return"]),
        })
    annual = pd.DataFrame(annual_rows)
    total = float((1 + equity["period_return"]).prod() - 1)
    benchmark = float((1 + equity["benchmark_return"]).prod() - 1)
    metrics = {
        "total_return": total,
        "benchmark_return": benchmark,
        "excess_return": total - benchmark,
        "positive_excess_years": int((annual["excess_return"] > 0).sum()),
        "rank_ic_5": float(equity["rank_ic_5"].mean()),
        "rank_ic_20": float(equity["rank_ic_20"].mean()),
        "max_drawdown": max_drawdown(equity["period_return"]),
        "average_one_way_turnover": float((equity["buy_turnover"] + equity["sell_turnover"]).mean() / 2),
        "average_transaction_cost": float(equity["transaction_cost"].mean()),
    }
    return metrics, annual


def run_research_v18(settings: V18Settings | None = None) -> dict:
    settings = settings or V18Settings()
    if asdict(settings) != asdict(V18Settings()):
        raise RuntimeError("V18只能按冻结默认配置运行，不接受事后参数改写")
    settings.ensure_dirs()
    lock = verify_research()
    if (settings.artifact_dir / "report.json").exists():
        raise RuntimeError("V18报告已存在，禁止覆盖")
    with (settings.artifact_dir / "run.started.json").open("x", encoding="utf-8") as handle:
        json.dump({"started_at_utc": datetime.now(timezone.utc).isoformat(), "lock_sha256": lock["lock_sha256"]}, handle, indent=2)

    print("V18 loading dataset", flush=True)
    dataset = load_v18_dataset()
    events = load_event_documents("data/event_documents_pit_v15.csv")
    embeddings = build_embeddings(settings=settings)
    if len(events) != len(embeddings):
        raise RuntimeError(f"事件行数与嵌入行数不一致: {len(events)} vs {len(embeddings)}")

    equity, signals, sector_ics = run_v18_backtest(dataset, events, embeddings, settings)
    verify_research()

    ablation = {}
    for mode in MODES:
        mode_equity = equity[equity["mode"] == mode].copy()
        mode_sector = sector_ics[sector_ics["mode"] == mode].copy()
        metrics, annual = _summarize(mode_equity, mode_sector)
        ablation[mode] = metrics
        mode_equity.to_csv(settings.artifact_dir / f"{mode}_equity.csv", index=False, encoding="utf-8-sig")
        annual.to_csv(settings.artifact_dir / f"{mode}_annual_metrics.csv", index=False, encoding="utf-8-sig")

    timing_metrics = ablation["v18_text_ungated"]
    baseline = ablation["v13_comparable"]
    gates = {
        "v18_excess_positive_vs_benchmark": timing_metrics["excess_return"] > 0,
        "v18_excess_beats_v16_ungated": timing_metrics["excess_return"] > V16_UNGATED_EXCESS,
        "v18_ic5_beats_v16": timing_metrics["rank_ic_5"] > V16_UNGATED_IC5,
        "v18_ic20_beats_v16": timing_metrics["rank_ic_20"] > V16_UNGATED_IC20,
        "v18_beat_v13_baseline": timing_metrics["excess_return"] > baseline["excess_return"],
        "v18_ic5_positive": timing_metrics["rank_ic_5"] > 0,
    }
    approved = all(gates.values())

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_v18_embedding_validation_complete",
        "plan_lock_sha256": lock["lock_sha256"],
        "environment": lock["environment"],
        "ablation": ablation,
        "v18_metrics": timing_metrics,
        "v16_reference": {
            "ungated_excess": V16_UNGATED_EXCESS,
            "ungated_ic5": V16_UNGATED_IC5,
            "ungated_ic20": V16_UNGATED_IC20,
        },
        "gates": gates,
        "approved": approved,
        "decision": "v18_replaces_v16_text" if approved else "keep_v6",
        "execution_authorized": False,
        "limitations": [
            "Embeddings are computed on announcement titles only, not full text.",
            "2020-2025 is a retrospective window reused by earlier versions.",
            "The embedding model is frozen at BAAI/bge-small-zh-v1.5 via hf-mirror.",
            "Positive IC (negative gate off) is required; historical evidence shows negative IC."
        ],
    }
    (settings.artifact_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
