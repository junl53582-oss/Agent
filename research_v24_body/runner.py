import json
import os
from datetime import datetime, timezone

from announcement_body.core import write_json_new
from .freeze import DIRECTORY, LEDGER, OUTPUT, verify
from .pipeline import body_gate, ingest_events, select_prospective_events


def run():
    lock = verify()
    write_json_new(DIRECTORY / "run.started.json", {"pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
                                                     "lock_sha256": lock["lock_sha256"]})
    events = select_prospective_events(LEDGER)
    results = ingest_events(events, OUTPUT)
    gate = body_gate(results)
    report = {"status": "prospective_body_ingestion_complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "lock_sha256": lock["lock_sha256"], "frozen_inputs_intact": True, "results": results, "gate": gate,
              "body_training_approved": False, "model_training_ready": False,
              "replacement_approved": False, "execution_authorized": False,
              "next_step": "rerun only after V5r2 appends a new eligible event; then bind facts before labels"}
    write_json_new(DIRECTORY / "report.json", report)
    return report

