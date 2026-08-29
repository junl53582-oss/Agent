import json
import os
import traceback
from datetime import datetime, timezone

import pandas as pd

from research_v20.freeze import digest, write_new
from research_v22.replay import load_scores
from research_v22r1.config import V22R1Settings
from .diagnostics import cost_diagnostic, equal_date_tail_spreads, selection_diagnostics, summarize_equal_date
from .freeze import DIRECTORY, V21, V22R1, verify


def progress(stage, **values):
    record = {"stage": stage, "pid": os.getpid(), "at_utc": datetime.now(timezone.utc).isoformat(), **values}
    temp = (DIRECTORY / "runtime_status.json").with_suffix(".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, DIRECTORY / "runtime_status.json")


def run():
    lock = verify()
    write_new(DIRECTORY / "run.started.json", {"pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(), "lock_sha256": lock["lock_sha256"]})
    try:
        progress("loading_frozen_outputs")
        settings = V22R1Settings()
        scores = load_scores(V21, settings.test_years)
        holdings = pd.read_csv(V22R1 / "holdings.csv", dtype={"symbol": str})
        equity = pd.read_csv(V22R1 / "equity.csv")
        progress("fixed_diagnostics", score_rows=len(scores))
        tails = equal_date_tail_spreads(scores)
        selection, active = selection_diagnostics(scores, holdings)
        tail_summary = pd.concat([summarize_equal_date(tails, "top_minus_bottom", ["target"]),
                                  summarize_equal_date(tails, "top_minus_bottom", ["test_year", "target"])], ignore_index=True)
        selection_summary = pd.concat([summarize_equal_date(selection, "overweight_minus_other", ["target"]),
                                       summarize_equal_date(selection, "overweight_minus_other", ["test_year", "target"]),
                                       summarize_equal_date(selection, "active_weighted_target", ["sector", "target"])], ignore_index=True)
        for name, frame in (("tail_periods", tails), ("tail_summary", tail_summary), ("selection_periods", selection),
                            ("selection_summary", selection_summary), ("active_map", active)):
            frame.to_csv(DIRECTORY / f"{name}.csv", index=False, mode="x")
        costs = cost_diagnostic(equity)
        files = ["tail_periods.csv", "tail_summary.csv", "selection_periods.csv", "selection_summary.csv", "active_map.csv"]
        report = {"status": "retrospective_score_to_portfolio_diagnosis_complete", "lock_sha256": lock["lock_sha256"],
                  "created_at_utc": datetime.now(timezone.utc).isoformat(), "costs": costs,
                  "output_sha256": {name: digest(DIRECTORY / name) for name in files}, "frozen_inputs_intact": True,
                  "purpose": "diagnosis_only_no_portfolio_rerun", "decision": "keep_v6_review_fixed_diagnostics",
                  "replacement_approved": False, "execution_authorized": False,
                  "limitations": ["Retrospective postmortem on already-observed V22r1 outcomes.",
                                  "Realized labels diagnose translation but cannot tune a new candidate.",
                                  "No portfolio was rerun and no parameter alternative was evaluated."]}
        write_new(DIRECTORY / "report.json", report)
        progress("complete", decision="keep_v6", execution_authorized=False)
        return report
    except BaseException as error:
        progress("failed", error=str(error), traceback=traceback.format_exc(), execution_authorized=False)
        raise
