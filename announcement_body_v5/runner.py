import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

from announcement_body.core import SHANGHAI, write_json_new
from .freeze import DATA, DIRECTORY, verify
from .ledger import record_observation, summarize_ledger
from .source import fetch_day


def _status(stage, **values):
    record = {"stage": stage, "pid": os.getpid(), "at_utc": datetime.now(timezone.utc).isoformat(), **values}
    temp = (DIRECTORY / "runtime_status.json").with_suffix(".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, DIRECTORY / "runtime_status.json")


def observe(target_date=None):
    lock = verify()
    now = datetime.now(timezone.utc)
    shanghai_date = now.astimezone(SHANGHAI).date().isoformat()
    target_date = target_date or shanghai_date
    if target_date != shanghai_date:
        raise ValueError("V5 only observes the current Shanghai date; historical backfill is forbidden")
    _status("querying_official_metadata", target_date=target_date)
    try:
        raw_pages, query = fetch_day(target_date)
        frozen = json.loads((DIRECTORY / "data.lock.json").read_text(encoding="utf-8"))
        receipt = record_observation(DATA, DIRECTORY, observed_at=now.isoformat(), target_date=target_date,
                                     freeze_created_at=frozen["created_at_utc"], raw_pages=raw_pages, query=query)
        summary = summarize_ledger(DATA)
        report = {"status": "observation_complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "lock_sha256": lock["lock_sha256"], "frozen_inputs_intact": True,
                  "latest_observation": receipt, "ledger": summary,
                  "historical_pit_verified": False, "prospective_pit_verified": False,
                  "model_training_ready": False, "replacement_approved": False, "execution_authorized": False,
                  "limitations": ["Metadata observation does not approve document bodies.",
                                  "Prospective effective dates await a separately verified trading calendar.",
                                  "Legacy reconstructed announcements remain quarantined."]}
        report_path = DIRECTORY / "observations" / f"{receipt['observation_id']}.report.json"
        write_json_new(report_path, report)
        _status("complete", target_date=target_date, records=receipt["records_returned"],
                new=receipt["statuses"]["new"], prospective_first_seen=receipt["prospective_first_seen_verified"],
                model_training_ready=False)
        return report
    except BaseException as error:
        failure_dir = DIRECTORY / "failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        failure_path = failure_dir / (now.strftime("%Y%m%dT%H%M%S%fZ") + ".json")
        write_json_new(failure_path, {"target_date": target_date, "started_at_utc": now.isoformat(),
                                     "error": f"{type(error).__name__}: {error}", "automatic_retry": False,
                                     "historical_pit_verified": False, "model_training_ready": False,
                                     "execution_authorized": False})
        _status("failed", target_date=target_date, error=str(error), traceback=traceback.format_exc(), model_training_ready=False)
        raise

