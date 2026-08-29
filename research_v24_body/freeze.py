import json
from datetime import datetime, timezone
from pathlib import Path

from research_v20.freeze import digest, write_new
from research_v24_prep.freeze import verify as verify_parent


DIRECTORY = Path("artifacts/research_v24_body")
LEDGER = Path("data/announcement_first_seen_v5r2")
OUTPUT = Path("data/announcement_bodies_v24")


def freeze():
    if any((DIRECTORY / name).exists() for name in ("plan.lock.json", "run.started.json", "report.json")):
        raise RuntimeError("V24 body pipeline already frozen/started")
    parent = verify_parent()
    receipt = json.loads((DIRECTORY / "test_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("passed") is not True:
        raise RuntimeError("tests must pass before freeze")
    files = sorted(Path("research_v24_body").glob("*.py"))
    files += [Path("tests/test_research_v24_body.py"), DIRECTORY / "protocol.json", DIRECTORY / "test_receipt.json",
              Path("announcement_body/core.py")]
    lock = {"locked_at_utc": datetime.now(timezone.utc).isoformat(), "parent_lock_sha256": parent["lock_sha256"],
            "purpose": "prospective_only_official_pdf_ingestion", "sha256": {p.as_posix(): digest(p) for p in files},
            "body_training_approved": False, "model_training_ready": False,
            "replacement_approved": False, "execution_authorized": False}
    write_new(DIRECTORY / "plan.lock.json", lock)
    with (DIRECTORY / "plan.lock.sha256").open("x", encoding="utf-8") as stream:
        stream.write(digest(DIRECTORY / "plan.lock.json") + "\n")
    return verify()


def verify():
    parent = verify_parent()
    actual = digest(DIRECTORY / "plan.lock.json")
    if actual != (DIRECTORY / "plan.lock.sha256").read_text().strip():
        raise RuntimeError("V24 body lock mismatch")
    lock = json.loads((DIRECTORY / "plan.lock.json").read_text(encoding="utf-8"))
    if lock["parent_lock_sha256"] != parent["lock_sha256"]:
        raise RuntimeError("V24 body parent changed")
    for name, expected in lock["sha256"].items():
        if digest(name) != expected:
            raise RuntimeError(f"V24 body frozen file changed: {name}")
    return {**lock, "lock_sha256": actual, "frozen_inputs_intact": True}

