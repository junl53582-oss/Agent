import json
from datetime import datetime, timezone
from pathlib import Path

from announcement_body.core import sha_file, write_json_new
from announcement_body_v3r1.freeze import verify as verify_parent


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "artifacts/announcement_body_v4"
SOURCE = ROOT / "data/announcements_pit_v14.csv"
SELECTION = ROOT / "artifacts/announcement_body_v1/selection.json"
PILOT_DATA = ROOT / "data/announcement_body_v1"


def freeze():
    if any((DIRECTORY / name).exists() for name in ("data.lock.json", "run.started.json", "report.json")):
        raise RuntimeError("announcement body V4 already frozen/started")
    parent = verify_parent()
    receipt = json.loads((DIRECTORY / "test_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("passed") is not True or receipt.get("aggregate_coverage_read_before_freeze") is not False:
        raise RuntimeError("tests must pass before freeze without aggregate coverage audit")
    source_lock = json.loads((ROOT / "artifacts/announcement_body_v1/data.lock.json").read_text(encoding="utf-8"))
    files = sorted((ROOT / "announcement_body_v4").glob("*.py"))
    files += [ROOT / "tests/test_announcement_body_v4.py", DIRECTORY / "protocol.json", DIRECTORY / "test_receipt.json",
              ROOT / "artifacts/announcement_body_v1/data.lock.json", SELECTION]
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    for record in selection["records"]:
        folder = PILOT_DATA / (record["symbol"] + "_" + record["announcement_id"])
        files.extend([folder / "receipt.json", folder / "detail.json"])
    lock = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "parent_lock_sha256": parent["lock_sha256"],
            "source_path": SOURCE.relative_to(ROOT).as_posix(), "source_sha256": sha_file(SOURCE),
            "expected_source_sha256": source_lock["source_sha256"], "expected_source_rows": 966865,
            "sha256": {path.relative_to(ROOT).as_posix(): sha_file(path) for path in files},
            "historical_pit_verified": False, "model_training_ready": False, "execution_authorized": False}
    write_json_new(DIRECTORY / "data.lock.json", lock)
    with (DIRECTORY / "data.lock.sha256").open("x", encoding="utf-8") as stream:
        stream.write(sha_file(DIRECTORY / "data.lock.json") + "\n")
    return verify()


def verify():
    parent = verify_parent()
    actual = sha_file(DIRECTORY / "data.lock.json")
    if actual != (DIRECTORY / "data.lock.sha256").read_text().strip():
        raise RuntimeError("announcement body V4 lock mismatch")
    lock = json.loads((DIRECTORY / "data.lock.json").read_text(encoding="utf-8"))
    if lock["parent_lock_sha256"] != parent["lock_sha256"] or sha_file(SOURCE) != lock["source_sha256"]:
        raise RuntimeError("announcement body V4 parent or source changed")
    for name, expected in lock["sha256"].items():
        if sha_file(ROOT / name) != expected:
            raise RuntimeError(f"announcement body V4 frozen file changed: {name}")
    return {"lock_sha256": actual, "frozen_inputs_intact": True, "historical_pit_verified": False,
            "model_training_ready": False, "execution_authorized": False}

