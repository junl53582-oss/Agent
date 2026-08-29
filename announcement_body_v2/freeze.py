import json
from datetime import datetime, timezone
from pathlib import Path

from announcement_body.cli import verify as verify_parent
from announcement_body.core import sha_file, write_json_new


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "artifacts/announcement_body_v2"


def freeze():
    if any((DIRECTORY / name).exists() for name in ("data.lock.json", "run.started.json", "report.json")):
        raise RuntimeError("announcement body V2 already frozen/started; do not overwrite")
    parent = verify_parent()
    receipt = json.loads((DIRECTORY / "test_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("passed") is not True:
        raise RuntimeError("tests must pass before freeze")
    files = sorted((ROOT / "announcement_body_v2").glob("*.py"))
    files += [ROOT / "tests/test_announcement_body_v2.py", DIRECTORY / "protocol.json", DIRECTORY / "gold.json", DIRECTORY / "test_receipt.json",
              ROOT / "artifacts/announcement_body_v1/data.lock.json", ROOT / "artifacts/announcement_body_v1/report.json",
              ROOT / "artifacts/announcement_body_v1/visual_qa.json"]
    lock = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "parent_lock_sha256": parent["lock_sha256"],
            "purpose": "period_currency_binding_data_quality_only",
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
        raise RuntimeError("announcement body V2 lock mismatch")
    lock = json.loads((DIRECTORY / "data.lock.json").read_text(encoding="utf-8"))
    if lock["parent_lock_sha256"] != parent["lock_sha256"]:
        raise RuntimeError("announcement body V2 parent changed")
    for name, expected in lock["sha256"].items():
        if sha_file(ROOT / name) != expected:
            raise RuntimeError(f"announcement body V2 frozen file changed: {name}")
    return {"lock_sha256": actual, "frozen_inputs_intact": True, "historical_pit_verified": False,
            "model_training_ready": False, "execution_authorized": False}
