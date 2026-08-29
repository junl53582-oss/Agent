import json
from datetime import datetime, timezone
from pathlib import Path

from announcement_body.core import sha_file, write_json_new
from announcement_body_v2.freeze import verify as verify_parent


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "artifacts/announcement_body_v2r1"
FAILED_PARENT = ROOT / "artifacts/announcement_body_v2"


def freeze():
    if any((DIRECTORY / name).exists() for name in ("data.lock.json", "run.started.json", "report.json")):
        raise RuntimeError("announcement body V2r1 already frozen/started")
    parent = verify_parent()
    failed = json.loads((FAILED_PARENT / "report.json").read_text(encoding="utf-8"))
    if failed.get("status") != "gold_binding_failed" or failed["evaluation"]["matched_facts"] != 19:
        raise RuntimeError("V2 failure evidence changed")
    receipt = json.loads((DIRECTORY / "test_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("passed") is not True:
        raise RuntimeError("tests must pass before freeze")
    files = sorted((ROOT / "announcement_body_v2r1").glob("*.py"))
    files += [ROOT / "tests/test_announcement_body_v2r1.py", DIRECTORY / "protocol.json", DIRECTORY / "test_receipt.json",
              FAILED_PARENT / "report.json", FAILED_PARENT / "facts.json", FAILED_PARENT / "evaluation.json",
              FAILED_PARENT / "runtime_status.json", ROOT / "artifacts/announcement_body_v2/gold.json"]
    lock = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "parent_lock_sha256": parent["lock_sha256"],
            "repair": "bind correction values by table columns", "sha256": {path.relative_to(ROOT).as_posix(): sha_file(path) for path in files},
            "historical_pit_verified": False, "model_training_ready": False, "execution_authorized": False}
    write_json_new(DIRECTORY / "data.lock.json", lock)
    with (DIRECTORY / "data.lock.sha256").open("x", encoding="utf-8") as stream:
        stream.write(sha_file(DIRECTORY / "data.lock.json") + "\n")
    return verify()


def verify():
    parent = verify_parent()
    actual = sha_file(DIRECTORY / "data.lock.json")
    if actual != (DIRECTORY / "data.lock.sha256").read_text().strip():
        raise RuntimeError("announcement body V2r1 lock mismatch")
    lock = json.loads((DIRECTORY / "data.lock.json").read_text(encoding="utf-8"))
    if lock["parent_lock_sha256"] != parent["lock_sha256"]:
        raise RuntimeError("announcement body V2r1 parent changed")
    for name, expected in lock["sha256"].items():
        if sha_file(ROOT / name) != expected:
            raise RuntimeError(f"announcement body V2r1 frozen file changed: {name}")
    return {"lock_sha256": actual, "frozen_inputs_intact": True, "historical_pit_verified": False,
            "model_training_ready": False, "execution_authorized": False}
