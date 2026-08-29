import gc
import json
import os
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from research_v16.data import load_v16_dataset
from research_v20.freeze import digest, write_new
from research_v20.validation import summarize
from research_v20r2.ledger import PriceBook
from stockpilot.membership import load_membership_history
from .config import V22Settings
from .freeze import DIRECTORY, PARENT_LEDGER, PARENT_SCORES, verify
from .replay import MARKET_PATH, MEMBERSHIP_PATH, MODES, attach_volatility, compare_control, load_scores, run_replay, schedule_from_parent


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
    write_new(folder / f"through_{year}.json", {"completed_year": year, "output_sha256": hashes, "partial_result_only": True,
                                                 "replacement_approved": False, "execution_authorized": False})


def run():
    lock = verify()
    write_new(DIRECTORY / "run.started.json", {"pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(), "lock_sha256": lock["lock_sha256"]})
    try:
        settings = V22Settings()
        with pd.option_context("mode.copy_on_write", True):
            progress("loading_frozen_scores")
            scores = load_scores(PARENT_SCORES, settings.test_years)
            progress("loading_dataset_for_frozen_volatility", score_rows=len(scores))
            dataset = load_v16_dataset()
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
                raise ValueError("frozen score dates differ from parent ledger")
            equity, holdings, daily, settlements = run_replay(scores, book, membership, schedule, settings, progress, checkpoint)
            progress("verifying_control_reproduction")
            control = compare_control(equity, holdings, daily, settlements, PARENT_LEDGER)
            metrics = {mode: summarize(equity[equity["mode"].eq(mode)]) for mode in MODES}
            for mode in MODES:
                nav = pd.concat([pd.Series([1.0]), daily[daily["mode"].eq(mode)].nav], ignore_index=True)
                metrics[mode]["max_drawdown"] = float((nav / nav.cummax() - 1).min())
                metrics[mode]["drawdown_sampling"] = "market_open_and_post_rebalance_NAV"
            if not np.isfinite(equity[["period_return", "benchmark_return", "nav"]]).all().all():
                raise ValueError("non-finite evaluated outcomes")
        verify()
        for name, frame in (("equity", equity), ("holdings", holdings), ("daily_nav", daily)):
            frame.to_csv(DIRECTORY / f"{name}.csv", index=False, mode="x")
        write_new(DIRECTORY / "settlements.json", {"events": settlements})
        gates = {"cumulative_excess_positive": metrics["global_only"]["excess_return"] > 0,
                 "positive_excess_years_at_least_4": metrics["global_only"]["positive_excess_years"] >= 4,
                 "technology_ic_nonnegative": metrics["global_only"]["technology_rank_ic_5"] >= 0,
                 "max_drawdown_not_below_minus_18pct": metrics["global_only"]["max_drawdown"] >= -0.18,
                 "same_accounting_v6_v8_comparison_available": False, "untouched_126_day_future_test_complete": False}
        report = {"status": "retrospective_global_score_ablation_complete", "lock_sha256": lock["lock_sha256"],
                  "created_at_utc": datetime.now(timezone.utc).isoformat(), "metrics": metrics,
                  "control_reproduction": control, "control_reproduction_passed": True, "gates": gates,
                  "all_promotion_gates_passed": all(gates.values()), "decision": "keep_v6", "replacement_approved": False,
                  "execution_authorized": False, "frozen_inputs_intact": True,
                  "output_sha256": {name: digest(DIRECTORY / name) for name in ("equity.csv", "holdings.csv", "daily_nav.csv", "settlements.json")},
                  "limitations": ["Evidence-selected retrospective single-variable ablation; not an untouched confirmation.",
                                  "No retraining or parameter search; frozen V21 global_model_score is the sole changed portfolio score.",
                                  "Announcement-body pilot, ranking objectives, timing and technology gates are excluded.",
                                  "V6 remains formal until all original gates and 126 new future trading days pass."]}
        write_new(DIRECTORY / "report.json", report)
        progress("complete", decision="keep_v6", all_promotion_gates_passed=report["all_promotion_gates_passed"], execution_authorized=False)
        return report
    except BaseException as error:
        progress("failed", error=str(error), traceback=traceback.format_exc(), execution_authorized=False)
        raise
