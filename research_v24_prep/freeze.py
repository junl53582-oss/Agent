import json
from datetime import datetime, timezone
from pathlib import Path

from research_v20.freeze import digest, write_new
from announcement_body_v3r1.freeze import verify as verify_facts
from announcement_body_v5r2.freeze import verify as verify_ledger


DIRECTORY = Path("artifacts/research_v24_prep")


def freeze():
    if any((DIRECTORY / name).exists() for name in ("plan.lock.json", "run.started.json", "report.json")):
        raise RuntimeError("V24 preparation already frozen/started")
    facts, ledger = verify_facts(), verify_ledger()
    receipt = json.loads((DIRECTORY / "test_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("passed") is not True:
        raise RuntimeError("tests must pass before freeze")
    files = sorted(Path("research_v24_prep").glob("*.py"))
    files += [Path("tests/test_research_v24_prep.py"), DIRECTORY / "protocol.json", DIRECTORY / "test_receipt.json",
              Path("artifacts/announcement_body_v3r1/facts.json"),
              Path("artifacts/announcement_body_v5r2/observations/20260829T070833554473Z.report.json")]
    lock = {"locked_at_utc": datetime.now(timezone.utc).isoformat(),
            "fact_parent_lock_sha256": facts["lock_sha256"], "ledger_parent_lock_sha256": ledger["lock_sha256"],
            "purpose": "v24_leakage_safe_announcement_feature_and_label_contract",
            "sha256": {path.as_posix(): digest(path) for path in files},
            "model_training_ready": False, "replacement_approved": False, "execution_authorized": False}
    write_new(DIRECTORY / "plan.lock.json", lock)
    with (DIRECTORY / "plan.lock.sha256").open("x", encoding="utf-8") as stream:
        stream.write(digest(DIRECTORY / "plan.lock.json") + "\n")
    return verify()


def verify():
    facts, ledger = verify_facts(), verify_ledger()
    actual = digest(DIRECTORY / "plan.lock.json")
    if actual != (DIRECTORY / "plan.lock.sha256").read_text().strip():
        raise RuntimeError("V24 preparation lock mismatch")
    lock = json.loads((DIRECTORY / "plan.lock.json").read_text(encoding="utf-8"))
    if lock["fact_parent_lock_sha256"] != facts["lock_sha256"] or lock["ledger_parent_lock_sha256"] != ledger["lock_sha256"]:
        raise RuntimeError("V24 preparation parent changed")
    for name, expected in lock["sha256"].items():
        if digest(name) != expected:
            raise RuntimeError(f"V24 preparation frozen file changed: {name}")
    return {**lock, "lock_sha256": actual, "frozen_inputs_intact": True}

