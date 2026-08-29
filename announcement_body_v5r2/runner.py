import json
import os
import traceback
from datetime import datetime, timezone

from announcement_body.core import SHANGHAI, write_json_new
from announcement_body_v5.ledger import record_observation, summarize_ledger
from .freeze import DATA, DIRECTORY, MEMBERSHIP, METADATA, verify
from .source import fetch_watchlist, load_org_ids, load_watchlist


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
        raise ValueError("V5r2 only observes the current Shanghai date; historical backfill is forbidden")
    _status("loading_frozen_watchlist", target_date=target_date)
    try:
        snapshot, symbols = load_watchlist(MEMBERSHIP, target_date)
        org_ids = load_org_ids(METADATA, symbols)
        _status("querying_throttled_partitioned_metadata", target_date=target_date, snapshot=snapshot, symbols=len(symbols))
        raw_pages, query = fetch_watchlist(target_date, symbols, org_ids)
        query["membership_snapshot"] = snapshot
        frozen = json.loads((DIRECTORY / "data.lock.json").read_text(encoding="utf-8"))
        receipt = record_observation(DATA, DIRECTORY, observed_at=now.isoformat(), target_date=target_date,
                                     freeze_created_at=frozen["created_at_utc"], raw_pages=raw_pages, query=query)
        summary = summarize_ledger(DATA)
        report = {"status": "throttled_partitioned_observation_complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "lock_sha256": lock["lock_sha256"], "frozen_inputs_intact": True,
                  "watchlist_snapshot": snapshot, "watchlist_size": len(symbols), "latest_observation": receipt,
                  "ledger": summary, "historical_pit_verified": False, "prospective_pit_verified": False,
                  "model_training_ready": False, "replacement_approved": False, "execution_authorized": False,
                  "limitations": ["Observation covers the frozen CSI 300 watchlist only.",
                                  "Metadata first-seen does not approve document bodies.",
                                  "Prospective effective dates await a separately verified trading calendar."]}
        write_json_new(DIRECTORY / "observations" / f"{receipt['observation_id']}.report.json", report)
        _status("complete", target_date=target_date, watchlist=len(symbols), records=receipt["records_returned"],
                unique=receipt["unique_announcements"], new=receipt["statuses"]["new"],
                prospective_first_seen=receipt["prospective_first_seen_verified"], model_training_ready=False)
        return report
    except BaseException as error:
        failure_dir = DIRECTORY / "failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        write_json_new(failure_dir / (now.strftime("%Y%m%dT%H%M%S%fZ") + ".json"),
                       {"target_date": target_date, "started_at_utc": now.isoformat(),
                        "error": f"{type(error).__name__}: {error}", "automatic_retry": False,
                        "historical_pit_verified": False, "model_training_ready": False, "execution_authorized": False})
        _status("failed", target_date=target_date, error=str(error), traceback=traceback.format_exc(), model_training_ready=False)
        raise

