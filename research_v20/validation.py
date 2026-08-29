import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_v15.features import load_event_documents
from research_v16.data import load_v16_dataset
from research_v16.text_model import EnsembleTextCorpus

from .backtest import MODES, run_backtest
from .config import V20Settings
from .freeze import DIRECTORY, digest, verify, write_new


def progress(stage, **values):
    record = {"stage": stage, "pid": os.getpid(), "at_utc": datetime.now(timezone.utc).isoformat(), **values}
    path = DIRECTORY / "runtime_status.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def summarize(frame):
    equity = pd.concat([pd.Series([1.0]), (1 + frame["period_return"]).cumprod()], ignore_index=True)
    total = float(equity.iloc[-1] - 1)
    benchmark = float((1 + frame["benchmark_return"]).prod() - 1)
    annual = frame.groupby("test_year").apply(
        lambda group: (1 + group["period_return"]).prod() - (1 + group["benchmark_return"]).prod(),
        include_groups=False,
    )
    held = frame[frame["in_market"].eq(True)]
    values = {
        "total_return": total, "benchmark_return": benchmark, "excess_return": total - benchmark,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "positive_excess_years": int(annual.gt(0).sum()), "test_years": int(len(annual)),
        "rank_ic_5": float(frame["rank_ic_5"].mean()), "rank_ic_20": float(frame["rank_ic_20"].mean()),
        "technology_rank_ic_5": float(frame["technology_rank_ic_5"].mean()),
        "average_one_way_turnover": float((frame["buy_turnover"] + frame["sell_turnover"]).mean() / 2),
        "average_transaction_cost": float(frame["transaction_cost"].mean()),
        "periods": len(frame), "periods_held": len(held),
        "held_period_win_rate": float(held["period_return"].gt(0).mean()) if len(held) else None,
    }
    return {key: (None if isinstance(value, float) and not np.isfinite(value) else value) for key, value in values.items()}


def run():
    lock = verify()
    if (DIRECTORY / "report.json").exists():
        raise RuntimeError("V20 report exists; do not overwrite")
    write_new(DIRECTORY / "run.started.json", {
        "pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
    })
    try:
        progress("loading_dataset")
        dataset = load_v16_dataset()
        progress("building_corpus", rows=len(dataset), symbols=int(dataset["symbol"].nunique()))
        corpus = EnsembleTextCorpus.build(load_event_documents(), V20Settings())
        equity, decisions = run_backtest(dataset, corpus, V20Settings(), progress)
        metrics = {mode: summarize(equity[equity["mode"].eq(mode)]) for mode in MODES}
        verify()
        equity.to_csv(DIRECTORY / "equity.csv", index=False)
        decisions.to_csv(DIRECTORY / "holdings.csv", index=False)
        report = {
            "status": "retrospective_implementation_repair_complete", "lock_sha256": lock["lock_sha256"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "metrics": metrics,
            "output_sha256": {name: digest(DIRECTORY / name) for name in ("equity.csv", "holdings.csv")},
            "frozen_inputs_intact": True, "execution_authorized": False,
            "replacement_approved": False, "decision": "keep_v6",
            "limitations": [
                "2020-2025 has been examined by earlier experiments; this is not an untouched holdout.",
                "Timing uses a PIT constituent daily-rebalanced proxy, not official CSI300 index closes.",
                "Inherited execution simulator is approximate (including weight drift and untradeable exits); not live-executable performance.",
                "126 untouched future trading days and all original promotion gates remain required.",
            ],
        }
        write_new(DIRECTORY / "report.json", report)
        progress("complete", decision="keep_v6", frozen_inputs_intact=True, execution_authorized=False)
        return report
    except BaseException as error:
        progress("failed", error=str(error), traceback=traceback.format_exc(), execution_authorized=False)
        raise
