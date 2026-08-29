import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from research_v20.freeze import digest, write_new
from research_v20r1.freeze import verify as verify_parent
from .config import V20R2Settings


DIRECTORY = Path("artifacts/research_v20r2")


def settings_dict():
    return json.loads(json.dumps(asdict(V20R2Settings()), default=str))


def freeze():
    if any((DIRECTORY / name).exists() for name in ("plan.lock.json", "run.started.json", "report.json")):
        raise RuntimeError("V20r2 already frozen/started; preserve and use a new revision")
    parent = verify_parent()
    audit = json.loads((DIRECTORY / "data_audit.json").read_text(encoding="utf-8"))
    if audit.get("passed") is not True or audit.get("performance_test") is not False:
        raise RuntimeError("all-year preflight must pass before freeze")
    files = list(Path("research_v20r2").glob("*.py"))
    files += list(Path("data/corporate_actions_v20r2_sources").glob("*.json"))
    files += [Path("tests/test_research_v20r2.py"), V20R2Settings().action_path,
              DIRECTORY / "protocol.json", DIRECTORY / "data_audit.json", DIRECTORY / "test_receipt.json",
              Path("artifacts/research_v20r1/runtime_status.json")]
    lock = {"locked_at_utc": datetime.now(timezone.utc).isoformat(), "parent_lock_sha256": parent["lock_sha256"],
            "settings": settings_dict(), "pandas_copy_on_write": True,
            "sha256": {p.as_posix(): digest(p) for p in files}, "execution_authorized": False}
    write_new(DIRECTORY / "plan.lock.json", lock)
    with (DIRECTORY / "plan.lock.sha256").open("x", encoding="utf-8") as handle:
        handle.write(digest(DIRECTORY / "plan.lock.json") + "\n")
    return verify()


def verify():
    parent = verify_parent()
    actual = digest(DIRECTORY / "plan.lock.json")
    if actual != (DIRECTORY / "plan.lock.sha256").read_text().strip():
        raise RuntimeError("V20r2 lock digest mismatch")
    lock = json.loads((DIRECTORY / "plan.lock.json").read_text(encoding="utf-8"))
    if parent["lock_sha256"] != lock["parent_lock_sha256"] or lock["settings"] != settings_dict():
        raise RuntimeError("parent/settings changed")
    for name, expected in lock["sha256"].items():
        if digest(name) != expected:
            raise RuntimeError(f"V20r2 frozen file changed: {name}")
    return {**lock, "lock_sha256": actual, "frozen_inputs_intact": True}
