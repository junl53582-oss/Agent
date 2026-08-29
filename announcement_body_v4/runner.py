import json
import os
import traceback
from datetime import datetime, timezone

from announcement_body.core import sha_file, write_json_new
from .audit import audit_pilot, audit_source, build_report
from .freeze import DIRECTORY, PILOT_DATA, SELECTION, SOURCE, verify


def progress(stage, **values):
    record = {"stage": stage, "pid": os.getpid(), "at_utc": datetime.now(timezone.utc).isoformat(), **values}
    temp = (DIRECTORY / "runtime_status.json").with_suffix(".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, DIRECTORY / "runtime_status.json")


def run():
    lock = verify()
    write_json_new(DIRECTORY / "run.started.json", {"pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(), "lock_sha256": lock["lock_sha256"]})
    try:
        progress("auditing_frozen_metadata")
        lock_record = json.loads((DIRECTORY / "data.lock.json").read_text(encoding="utf-8"))
        source = audit_source(SOURCE)
        pilot = audit_pilot(PILOT_DATA, SELECTION)
        report = build_report(source, pilot, lock_record["expected_source_sha256"], lock_record["expected_source_rows"])
        report.update({"created_at_utc": datetime.now(timezone.utc).isoformat(), "lock_sha256": lock["lock_sha256"],
                       "frozen_inputs_intact": True,
                       "limitations": ["The metadata archive does not contain contemporaneous first-seen timestamps.",
                                       "Revision-like titles have no explicit parent-announcement field in the frozen CSV.",
                                       "Official source publication dates are day-granularity and are not intraday availability proof.",
                                       "No market, return, label or model data were read."]})
        write_json_new(DIRECTORY / "source_audit.json", source)
        write_json_new(DIRECTORY / "pilot_audit.json", pilot)
        report["output_sha256"] = {name: sha_file(DIRECTORY / name) for name in ("source_audit.json", "pilot_audit.json")}
        write_json_new(DIRECTORY / "report.json", report)
        progress("complete", status=report["status"], rows=source["rows"],
                 first_seen=report["gates"]["historical_first_seen_proven"],
                 lineage=report["gates"]["revision_lineage_verified"], model_training_ready=False)
        return report
    except BaseException as error:
        progress("failed", error=str(error), traceback=traceback.format_exc(), model_training_ready=False, execution_authorized=False)
        raise

