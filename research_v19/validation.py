from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research_v15.features import load_event_documents
from research_v16.text_model import EnsembleTextCorpus

from .backtest import MODES, max_drawdown, run_v19_backtest
from .config import V19Settings
from .data import load_v19_dataset
from .freeze import verify_research


V16_UNGATED_EXCESS = 0.0592
V16_UNGATED_IC5 = -0.0251
V16_UNGATED_IC20 = -0.0168


def _summarize(equity: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
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


def run_research_v19(settings: V19Settings | None = None) -> dict:
    settings = settings or V19Settings()
    if asdict(settings) != asdict(V19Settings()):
        raise RuntimeError("V19只能按冻结默认配置运行，不接受事后参数改写")
    settings.ensure_dirs()
    lock = verify_research()
    if (settings.artifact_dir / "report.json").exists():
        raise RuntimeError("V19报告已存在，禁止覆盖")
    with (settings.artifact_dir / "run.started.json").open("x", encoding="utf-8") as handle:
        json.dump({"started_at_utc": datetime.now(timezone.utc).isoformat(), "lock_sha256": lock["lock_sha256"]}, handle, indent=2)

    print("V19 loading dataset", flush=True)
    dataset = load_v19_dataset()
    events = load_event_documents("data/event_documents_pit_v15.csv")
    corpus = EnsembleTextCorpus.build(events, settings)

    equity = run_v19_backtest(dataset, corpus, settings)
    verify_research()

    ablation = {}
    for mode in MODES:
        mode_equity = equity[equity["mode"] == mode].copy()
        metrics, annual = _summarize(mode_equity)
        ablation[mode] = metrics
        mode_equity.to_csv(settings.artifact_dir / f"{mode}_equity.csv", index=False, encoding="utf-8-sig")
        annual.to_csv(settings.artifact_dir / f"{mode}_annual_metrics.csv", index=False, encoding="utf-8-sig")

    adaptive = ablation["v19_adaptive"]
    ungated = ablation["v16_ungated"]
    gates = {
        "v19_excess_positive": adaptive["excess_return"] > 0,
        "v19_excess_beats_v16_ungated": adaptive["excess_return"] > V16_UNGATED_EXCESS,
        "v19_beats_same_run_v16": adaptive["excess_return"] > ungated["excess_return"],
        "v19_ic5_beats_v16": adaptive["rank_ic_5"] > V16_UNGATED_IC5,
        "v19_ic20_beats_v16": adaptive["rank_ic_20"] > V16_UNGATED_IC20,
        "v19_drawdown_no_worse": adaptive["max_drawdown"] >= ungated["max_drawdown"],
        "v19_ic5_positive": adaptive["rank_ic_5"] > 0,
    }
    approved = all(gates.values())

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_v19_regime_adaptive_validation_complete",
        "plan_lock_sha256": lock["lock_sha256"],
        "environment": lock["environment"],
        "ablation": ablation,
        "v19_metrics": adaptive,
        "v16_reference": {
            "ungated_excess": V16_UNGATED_EXCESS,
            "ungated_ic5": V16_UNGATED_IC5,
            "ungated_ic20": V16_UNGATED_IC20,
        },
        "gates": gates,
        "approved": approved,
        "decision": "v19_replaces_v16" if approved else "keep_v6",
        "execution_authorized": False,
        "limitations": [
            "Regime weights are preregistered, not fitted; they are one plausible configuration.",
            "Market regime uses the prior period benchmark return, known only after the prior period closes.",
            "2020-2025 is a retrospective window reused by earlier versions.",
            "Historical evidence suggests the IC-positive gate will fail."
        ],
    }
    (settings.artifact_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
