import json
import os
import traceback
from datetime import datetime, timezone

import pandas as pd

from research_v15.features import load_event_documents
from research_v16.data import load_v16_dataset
from research_v16.text_model import EnsembleTextCorpus
from research_v20.backtest import MODES, run_backtest
from research_v20.freeze import digest, write_new
from research_v20.validation import summarize

from .config import V20R1Settings
from .freeze import DIRECTORY, verify


def progress(stage, **values):
    record = {"stage": stage, "pid": os.getpid(), "at_utc": datetime.now(timezone.utc).isoformat(), **values}
    target = DIRECTORY / "runtime_status.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def run():
    lock = verify()
    if (DIRECTORY / "report.json").exists():
        raise RuntimeError("V20r1 report exists")
    write_new(DIRECTORY / "run.started.json", {
        "pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
    })
    try:
        # Scoped public pandas option, not a patch to any frozen function.
        # In particular reset_index(copy(deep=None)) no longer consolidates and
        # duplicates a multi-GiB float block. Writes still have isolated semantics.
        with pd.option_context("mode.copy_on_write", True):
            progress("loading_dataset", pandas_copy_on_write=True)
            dataset = load_v16_dataset()
            progress("building_corpus", rows=len(dataset), columns=len(dataset.columns), symbols=int(dataset["symbol"].nunique()))
            corpus = EnsembleTextCorpus.build(load_event_documents(), V20R1Settings())
            equity, holdings = run_backtest(dataset, corpus, V20R1Settings(), progress)
            metrics = {mode: summarize(equity[equity["mode"].eq(mode)]) for mode in MODES}
        verify()
        equity.to_csv(DIRECTORY / "equity.csv", index=False)
        holdings.to_csv(DIRECTORY / "holdings.csv", index=False)
        report = {
            "status": "retrospective_memory_repair_complete", "lock_sha256": lock["lock_sha256"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "metrics": metrics,
            "output_sha256": {name: digest(DIRECTORY / name) for name in ("equity.csv", "holdings.csv")},
            "frozen_inputs_intact": True, "execution_authorized": False,
            "replacement_approved": False, "decision": "keep_v6",
            "limitations": ["Repeated retrospective window, not a blind holdout", "Inherited execution approximation unresolved", "PIT constituent proxy is not official CSI300 index", "126 untouched future trading days still required"],
        }
        write_new(DIRECTORY / "report.json", report)
        progress("complete", decision="keep_v6", execution_authorized=False)
        return report
    except BaseException as error:
        progress("failed", error=str(error), traceback=traceback.format_exc(), execution_authorized=False)
        raise
