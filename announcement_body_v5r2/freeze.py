import json
from datetime import datetime, timezone
from pathlib import Path

from announcement_body.core import sha_file, write_json_new
from announcement_body_v5r1.freeze import verify as verify_parent


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "artifacts/announcement_body_v5r2"
DATA = ROOT / "data/announcement_first_seen_v5r2"
MEMBERSHIP = ROOT / "data/universes/000300/history_v10.csv"
METADATA = ROOT / "data/announcements_pit_v14.csv"
FAILED = ROOT / "artifacts/announcement_body_v5r1"


def freeze():
    if any((DIRECTORY / name).exists() for name in ("data.lock.json", "runtime_status.json")):
        raise RuntimeError("announcement body V5r2 already frozen/started")
    if DATA.exists() and any(DATA.rglob("*")):
        raise RuntimeError("V5r2 ledger must be empty before freeze")
    parent = verify_parent()
    failures = sorted((FAILED / "failures").glob("*.json"))
    if len(failures) != 1:
        raise RuntimeError("expected exactly one preserved V5r1 failure")
    status = json.loads((FAILED / "runtime_status.json").read_text(encoding="utf-8"))
    if status.get("stage") != "failed" or status.get("error") != "official response shape invalid":
        raise RuntimeError("V5r1 failure evidence changed")
    receipt = json.loads((DIRECTORY / "test_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("passed") is not True or receipt.get("live_official_query_run_before_freeze") is not False:
        raise RuntimeError("repair tests must pass before live query")
    files = sorted((ROOT / "announcement_body_v5r2").glob("*.py"))
    files += [ROOT / "tests/test_announcement_body_v5r2.py", DIRECTORY / "protocol.json", DIRECTORY / "test_receipt.json",
              FAILED / "data.lock.json", FAILED / "runtime_status.json", failures[0], MEMBERSHIP, METADATA]
    lock = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "parent_lock_sha256": parent["lock_sha256"],
            "allowed_dynamic_data_root": DATA.relative_to(ROOT).as_posix(),
            "allowed_dynamic_artifact_root": (DIRECTORY / "observations").relative_to(ROOT).as_posix(),
            "sha256": {path.relative_to(ROOT).as_posix(): sha_file(path) for path in files},
            "prospective_pit_verified": False, "historical_pit_verified": False,
            "model_training_ready": False, "execution_authorized": False}
    write_json_new(DIRECTORY / "data.lock.json", lock)
    with (DIRECTORY / "data.lock.sha256").open("x", encoding="utf-8") as stream:
        stream.write(sha_file(DIRECTORY / "data.lock.json") + "\n")
    return verify()


def verify():
    parent = verify_parent()
    actual = sha_file(DIRECTORY / "data.lock.json")
    if actual != (DIRECTORY / "data.lock.sha256").read_text().strip():
        raise RuntimeError("announcement body V5r2 lock mismatch")
    lock = json.loads((DIRECTORY / "data.lock.json").read_text(encoding="utf-8"))
    if lock["parent_lock_sha256"] != parent["lock_sha256"]:
        raise RuntimeError("announcement body V5r2 parent changed")
    for name, expected in lock["sha256"].items():
        if sha_file(ROOT / name) != expected:
            raise RuntimeError(f"announcement body V5r2 frozen file changed: {name}")
    return {"lock_sha256": actual, "frozen_inputs_intact": True, "prospective_pit_verified": False,
            "historical_pit_verified": False, "model_training_ready": False, "execution_authorized": False}

