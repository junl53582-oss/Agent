import json
from datetime import datetime, timezone
from pathlib import Path

from announcement_body.core import sha_file, write_json_new
from announcement_body_v2r1.freeze import verify as verify_parent


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "artifacts/announcement_body_v3"
DATA = ROOT / "data/announcement_body_v1"
SELECTION = ROOT / "artifacts/announcement_body_v1/selection.json"


def freeze():
    if any((DIRECTORY / name).exists() for name in ("data.lock.json", "run.started.json", "report.json")):
        raise RuntimeError("announcement body V3 already frozen/started")
    parent = verify_parent()
    receipt = json.loads((DIRECTORY / "test_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("passed") is not True or receipt.get("actual_gold_evaluation_run_before_freeze") is not False:
        raise RuntimeError("pre-freeze tests must pass without evaluating actual gold")
    from .gold import expanded_gold
    document_ids = expanded_gold()["documents"]
    files = sorted((ROOT / "announcement_body_v3").glob("*.py"))
    files += [ROOT / "tests/test_announcement_body_v3.py", DIRECTORY / "protocol.json", DIRECTORY / "test_receipt.json",
              ROOT / "artifacts/announcement_body_v2/gold.json", SELECTION]
    for document_id in document_ids:
        files.extend([DATA / document_id / "body.pdf", DATA / document_id / "parsed.json"])
    lock = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "parent_lock_sha256": parent["lock_sha256"],
            "scope": "eleven visually reviewed text PDFs; one scan quarantined",
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
        raise RuntimeError("announcement body V3 lock mismatch")
    lock = json.loads((DIRECTORY / "data.lock.json").read_text(encoding="utf-8"))
    if lock["parent_lock_sha256"] != parent["lock_sha256"]:
        raise RuntimeError("announcement body V3 parent changed")
    for name, expected in lock["sha256"].items():
        if sha_file(ROOT / name) != expected:
            raise RuntimeError(f"announcement body V3 frozen file changed: {name}")
    return {"lock_sha256": actual, "frozen_inputs_intact": True, "historical_pit_verified": False,
            "model_training_ready": False, "execution_authorized": False}

