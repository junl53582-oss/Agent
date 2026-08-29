import gc
import json
import os
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from research_v16.data import load_v16_dataset
from research_v20.freeze import digest, write_new
from research_v20r2.ledger import PriceBook
from research_v22.replay import MARKET_PATH, MEMBERSHIP_PATH
from stockpilot.membership import load_membership_history

from .config import V29Settings
from .evaluation import evaluate_three_gates
from .freeze import DIRECTORY, PARENT_LEDGER, PARENT_SCORES, verify
from .model import build_candidate_scores
from .replay import attach_volatility, compare_control, load_scores, run_replay, schedule_from_parent


def progress(stage, **values):
    record = {"stage": stage, "pid": os.getpid(), "at_utc": datetime.now(timezone.utc).isoformat(), **values}
    temp = (DIRECTORY / "runtime_status.json").with_suffix(".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, DIRECTORY / "runtime_status.json")
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
                                                 "partial_result_only": True, "replacement_approved": False,
                                                 "execution_authorized": False})


def run():
    lock = verify()
    write_new(DIRECTORY / "run.started.json", {"pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
                                                "lock_sha256": lock["lock_sha256"]})
    try:
        settings = V29Settings()
        with pd.option_context("mode.copy_on_write", True):
            progress("loading_frozen_scores")
            scores = load_scores(PARENT_SCORES, settings.test_years)
            progress("loading_frozen_pit_dataset", score_rows=len(scores))
            dataset = load_v16_dataset()
            scores, model_diagnostics = build_candidate_scores(scores, dataset, settings, progress)
            scores = attach_volatility(scores, dataset)
            price_panel = dataset[["date", "symbol", "open", "close", "volume"]].copy()
            del dataset
            gc.collect()
            events = json.loads(settings.action_path.read_text(encoding="utf-8"))["events"]
            progress("building_common_calendar", market_rows=len(price_panel))
            book = PriceBook(price_panel, events)
            del price_panel
            gc.collect()
            membership = load_membership_history(MEMBERSHIP_PATH)
            parent_equity = pd.read_csv(PARENT_LEDGER / "equity.csv", parse_dates=["date", "entry_date", "end_date"])
            schedule = schedule_from_parent(parent_equity, book)
            if set(pd.to_datetime(scores.date.unique())) != {row[0] for row in schedule}:
                raise ValueError("V29 score dates differ from frozen ledger")
            equity, holdings, daily, settlements = run_replay(scores, book, membership, schedule, settings, progress, checkpoint)
            progress("verifying_control_reproduction")
            control = compare_control(equity, holdings, daily, settlements, PARENT_LEDGER)
            gates = evaluate_three_gates(scores, equity, settings)
            if not np.isfinite(equity[["period_return", "benchmark_return", "nav"]]).all().all():
                raise ValueError("non-finite evaluated outcomes")
        verify()
        trace_columns = ["date", "symbol", "eligible", "broad_sector", "benchmark_weight", "label_5",
                         "v10_target_20", "v16_score", "global_model_score", "v29_score", "model_confidence"]
        for name, frame in (("equity", equity), ("holdings", holdings), ("daily_nav", daily),
                            ("score_trace", scores[trace_columns])):
            frame.to_csv(DIRECTORY / f"{name}.csv", index=False, mode="x")
        write_new(DIRECTORY / "settlements.json", {"events": settlements})
        write_new(DIRECTORY / "model_diagnostics.json", model_diagnostics)
        report = {"status": "retrospective_sector_conditional_tail_complete", "lock_sha256": lock["lock_sha256"],
                  "created_at_utc": datetime.now(timezone.utc).isoformat(), "three_gates": gates,
                  "control_reproduction": control, "control_reproduction_passed": True,
                  "all_promotion_gates_passed": gates["all_three_passed"], "decision": "keep_v6",
                  "incumbent_semantics": "not_reliably_beaten_not_claimed_best_predictor",
                  "retroactive_reapproval": {"V25r1": False, "V26": False, "V28": False},
                  "replacement_approved": False, "execution_authorized": False, "frozen_inputs_intact": True,
                  "model_diagnostics_sha256": digest(DIRECTORY / "model_diagnostics.json"),
                  "output_sha256": {name: digest(DIRECTORY / name) for name in
                                    ("equity.csv", "holdings.csv", "daily_nav.csv", "score_trace.csv",
                                     "settlements.json", "model_diagnostics.json")},
                  "limitations": ["Only tail-label conditioning changed from V28; no external feature was admitted.",
                                  "The 2020-2025 window is retrospective and repeatedly observed.",
                                  "V6 remains incumbent until all three gates and 126 untouched future trading days pass."]}
        write_new(DIRECTORY / "report.json", report)
        progress("complete", decision="keep_v6", all_promotion_gates_passed=report["all_promotion_gates_passed"], execution_authorized=False)
        return report
    except BaseException as error:
        progress("failed", error=str(error), traceback=traceback.format_exc(), execution_authorized=False)
        raise
