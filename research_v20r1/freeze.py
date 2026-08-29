import json
from datetime import datetime, timezone
from pathlib import Path

from research_v20.freeze import digest, verify as verify_parent, write_new


DIRECTORY = Path("artifacts/research_v20r1")


def freeze():
    if any((DIRECTORY / name).exists() for name in ("plan.lock.json", "run.started.json", "report.json")):
        raise RuntimeError("V20r1 already frozen or started; preserve it")
    parent = verify_parent()
    files = sorted(str(path).replace("\\", "/") for path in Path("research_v20r1").glob("*.py"))
    files += ["tests/test_research_v20r1.py", str(DIRECTORY / "protocol.json"), "artifacts/research_v20/runtime_status.json"]
    lock = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_lock_sha256": parent["lock_sha256"],
        "pandas_copy_on_write": True, "precision": "unchanged_float64",
        "sha256": {path: digest(path) for path in files}, "execution_authorized": False,
    }
    write_new(DIRECTORY / "plan.lock.json", lock)
    with (DIRECTORY / "plan.lock.sha256").open("x", encoding="utf-8") as handle:
        handle.write(digest(DIRECTORY / "plan.lock.json") + "\n")
    return verify()


def verify():
    parent = verify_parent()
    actual = digest(DIRECTORY / "plan.lock.json")
    if actual != (DIRECTORY / "plan.lock.sha256").read_text().strip():
        raise RuntimeError("V20r1 lock mismatch")
    lock = json.loads((DIRECTORY / "plan.lock.json").read_text(encoding="utf-8"))
    if parent["lock_sha256"] != lock["parent_lock_sha256"]:
        raise RuntimeError("V20 parent changed")
    for name, expected in lock["sha256"].items():
        if digest(name) != expected:
            raise RuntimeError(f"V20r1 frozen file changed: {name}")
    return {**lock, "lock_sha256": actual, "frozen_inputs_intact": True}
