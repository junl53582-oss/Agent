import json
import os
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from research_v15.features import load_event_documents
from research_v16.data import load_v16_dataset
from research_v16.text_model import EnsembleTextCorpus
from research_v20.backtest import MODES
from research_v20.freeze import digest, write_new
from research_v20.validation import summarize
from .audit import audit_inputs, load_book
from .backtest import run_backtest
from .config import V20R2Settings
from .freeze import DIRECTORY, verify


def progress(stage, **values):
    record = {"stage": stage, "pid": os.getpid(), "at_utc": datetime.now(timezone.utc).isoformat(), **values}
    target = DIRECTORY / "runtime_status.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, target)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def checkpoint(year, equity, holdings, daily):
    folder = DIRECTORY / "checkpoints"
    folder.mkdir(exist_ok=True)
    hashes = {}
    for name, frame in (("equity", equity), ("holdings", holdings), ("daily_nav", daily)):
        path = folder / f"through_{year}_{name}.csv"
        frame.to_csv(path, index=False, mode="x")
        hashes[path.name] = digest(path)
    write_new(folder / f"through_{year}.json", {"completed_year": year, "output_sha256": hashes,
                                              "partial_result_only": True, "replacement_approved": False})


def run():
    lock = verify()
    settings = V20R2Settings()
    write_new(DIRECTORY / "run.started.json", {"pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
                                             "lock_sha256": lock["lock_sha256"]})
    try:
        with pd.option_context("mode.copy_on_write", True):
            progress("data_preflight")
            panel, book, membership = load_book(settings)
            audit = audit_inputs(panel, book, membership, settings)
            del panel
            write_new(DIRECTORY / "run_preflight.json", audit)
            progress("loading_dataset")
            dataset = load_v16_dataset()
            progress("building_corpus", rows=len(dataset), symbols=int(dataset.symbol.nunique()))
            corpus = EnsembleTextCorpus.build(load_event_documents(), settings)
            equity, holdings, daily, settlements = run_backtest(dataset, corpus, book, membership, settings, progress, checkpoint)
            metrics = {mode: summarize(equity[equity["mode"].eq(mode)]) for mode in MODES}
            for mode in MODES:
                series = pd.concat([pd.Series([1.0]), daily[daily["mode"].eq(mode)].nav], ignore_index=True)
                metrics[mode]["max_drawdown"] = float((series / series.cummax() - 1).min())
                metrics[mode]["drawdown_sampling"] = "market_open_and_post_rebalance_NAV"
            if not np.isfinite(equity[["period_return", "benchmark_return", "nav"]]).all().all():
                raise ValueError("non-finite evaluated outcomes")
        progress("verifying_outputs")
        verify()
        for name, frame in (("equity", equity), ("holdings", holdings), ("daily_nav", daily)):
            frame.to_csv(DIRECTORY / f"{name}.csv", index=False, mode="x")
        write_new(DIRECTORY / "settlements.json", {"events": settlements})
        report = {"status": "retrospective_calendar_ledger_repair_complete", "lock_sha256": lock["lock_sha256"],
                  "created_at_utc": datetime.now(timezone.utc).isoformat(), "metrics": metrics,
                  "output_sha256": {name: digest(DIRECTORY / name) for name in ("equity.csv", "holdings.csv", "daily_nav.csv", "settlements.json")},
                  "frozen_inputs_intact": True, "execution_authorized": False, "replacement_approved": False,
                  "decision": "keep_v6", "promotion_gates_evaluated": False,
                  "limitations": [
                      "Repeated retrospective window, not a blind holdout; 126 untouched future days still required.",
                      "Benchmark is a zero-cost 20-market-day PIT-snapshot portfolio with the same trade restrictions, not official CSI300 returns.",
                      "Legacy predictors and their per-security-row IC labels are unchanged; labels missing near mergers remain missing and counts are reported.",
                      "Missing quotes are nontradable last-close marks, not verified fair value; temporary missing feeds and suspensions are not distinguished.",
                      "Ordinary unrestricted public shareholders only; swaps settle on new-share listing day. Pre-listing marks stay at old last close.",
                      "HFQ economic units use raw/HFQ price anchors at swaps; no tax lots, integer share rounding or investor-specific cash election.",
                      "Daily opening-price limits are approximate, without PIT ST names, board lots or order book depth. Blocked orders retry only on next rebalance.",
                      "New calendar/NAV/benchmark accounting is not directly comparable with old V6/V8 archived cost and return metrics; gates remain unpassed."]}
        write_new(DIRECTORY / "report.json", report)
        progress("complete", decision="keep_v6", execution_authorized=False)
        return report
    except BaseException as error:
        progress("failed", error=str(error), traceback=traceback.format_exc(), execution_authorized=False)
        raise
