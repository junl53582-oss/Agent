import json
import os
import traceback
from datetime import datetime, timezone

from announcement_body.core import write_json_new
from research_v24_body.pipeline import body_gate, ingest_events, select_prospective_events
from .freeze import DIRECTORY, LEDGER, OUTPUT, verify


def _attempt_id(now):
    return now.strftime("%Y%m%dT%H%M%S%fZ")


def _status(stage, **values):
    record = {"stage": stage, "pid": os.getpid(), "at_utc": datetime.now(timezone.utc).isoformat(), **values}
    target = DIRECTORY / "runtime_status.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, target)


def run():
    lock = verify()
    now = datetime.now(timezone.utc)
    attempt_id = _attempt_id(now)
    run_dir = DIRECTORY / "runs"
    write_json_new(run_dir / f"{attempt_id}.started.json",
                   {"pid": os.getpid(), "started_at_utc": now.isoformat(), "lock_sha256": lock["lock_sha256"]})
    try:
        _status("selecting_new_eligible_events", attempt_id=attempt_id)
        events = select_prospective_events(LEDGER)
        _status("ingesting_official_bodies", attempt_id=attempt_id, eligible_events=len(events))
        results = ingest_events(events, OUTPUT)
        gate = body_gate(results)
        report = {"status": "prospective_body_ingestion_complete", "attempt_id": attempt_id,
                  "created_at_utc": datetime.now(timezone.utc).isoformat(), "lock_sha256": lock["lock_sha256"],
                  "frozen_inputs_intact": True, "results": results, "gate": gate,
                  "body_training_approved": False, "model_training_ready": False,
                  "replacement_approved": False, "execution_authorized": False}
        write_json_new(run_dir / f"{attempt_id}.report.json", report)
        _status("complete", attempt_id=attempt_id, eligible_events=len(events), text_extracted=gate["text_extracted"],
                model_training_ready=False, execution_authorized=False)
        return report
    except BaseException as error:
        write_json_new(run_dir / f"{attempt_id}.failure.json",
                       {"attempt_id": attempt_id, "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(), "automatic_retry": False,
                        "model_training_ready": False, "execution_authorized": False})
        _status("failed", attempt_id=attempt_id, error=str(error), model_training_ready=False, execution_authorized=False)
        raise

