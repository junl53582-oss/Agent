import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from research_v20.freeze import digest, write_new
from research_v22.freeze import verify as verify_parent
from .config import V22R1Settings


DIRECTORY = Path("artifacts/research_v22r1")
FAILED_PARENT = Path("artifacts/research_v22")


def settings_dict():
    return json.loads(json.dumps(asdict(V22R1Settings()), default=str))


def freeze():
    if any((DIRECTORY / name).exists() for name in ("plan.lock.json", "run.started.json", "report.json")):
        raise RuntimeError("V22r1 already frozen/started; do not overwrite")
    parent = verify_parent()
    failed = json.loads((FAILED_PARENT / "runtime_status.json").read_text(encoding="utf-8"))
    if failed.get("stage") != "failed" or "function' object has no attribute 'eq" not in failed.get("error", ""):
        raise RuntimeError("V22 failure evidence does not match the registered implementation defect")
    receipt = json.loads((DIRECTORY / "test_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("passed") is not True:
        raise RuntimeError("tests must pass before freeze")
    files = sorted(Path("research_v22r1").glob("*.py"))
    files += [Path("tests/test_research_v22r1.py"), DIRECTORY / "protocol.json", DIRECTORY / "test_receipt.json",
              FAILED_PARENT / "run.started.json", FAILED_PARENT / "runtime_status.json",
              FAILED_PARENT / "logs/run.stdout.log", FAILED_PARENT / "logs/run.stderr.log"]
    lock = {"locked_at_utc": datetime.now(timezone.utc).isoformat(), "parent_lock_sha256": parent["lock_sha256"],
            "settings": settings_dict(), "repair": "DataFrame mode column accessed with brackets instead of the pandas mode method",
            "sha256": {path.as_posix(): digest(path) for path in files}, "execution_authorized": False, "replacement_approved": False}
    write_new(DIRECTORY / "plan.lock.json", lock)
    with (DIRECTORY / "plan.lock.sha256").open("x", encoding="utf-8") as stream:
        stream.write(digest(DIRECTORY / "plan.lock.json") + "\n")
    return verify()


def verify():
    parent = verify_parent()
    actual = digest(DIRECTORY / "plan.lock.json")
    if actual != (DIRECTORY / "plan.lock.sha256").read_text().strip():
        raise RuntimeError("V22r1 lock mismatch")
    lock = json.loads((DIRECTORY / "plan.lock.json").read_text(encoding="utf-8"))
    if lock["parent_lock_sha256"] != parent["lock_sha256"] or lock["settings"] != settings_dict():
        raise RuntimeError("V22r1 parent/settings changed")
    for name, expected in lock["sha256"].items():
        if digest(name) != expected:
            raise RuntimeError(f"V22r1 frozen file changed: {name}")
    return {**lock, "lock_sha256": actual, "frozen_inputs_intact": True}
