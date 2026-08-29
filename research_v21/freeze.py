import json
from datetime import datetime, timezone
from pathlib import Path

from research_v20.freeze import digest, write_new
from research_v20r2.freeze import verify as verify_parent


DIRECTORY = Path("artifacts/research_v21")
PARENT = Path("artifacts/research_v20r2")


def freeze():
    if any((DIRECTORY / name).exists() for name in ("plan.lock.json", "run.started.json", "report.json")):
        raise RuntimeError("V21 already frozen/started; do not overwrite")
    parent = verify_parent()
    report = json.loads((PARENT / "report.json").read_text(encoding="utf-8"))
    if report["lock_sha256"] != parent["lock_sha256"]:
        raise RuntimeError("parent result lock mismatch")
    for name, expected in report["output_sha256"].items():
        if digest(PARENT / name) != expected:
            raise RuntimeError("parent result modified")
    receipt = json.loads((DIRECTORY / "test_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("passed") is not True:
        raise RuntimeError("tests must pass before freeze")
    files = sorted(Path("research_v21").glob("*.py"))
    files += [Path("tests/test_research_v21.py"), DIRECTORY / "protocol.json", DIRECTORY / "test_receipt.json",
              PARENT / "report.json", PARENT / "equity.csv", PARENT / "runtime_status.json",
              Path("artifacts/autopilot/v20r2_acceptance_20260829.json")]
    lock = {"locked_at_utc": datetime.now(timezone.utc).isoformat(), "parent_lock_sha256": parent["lock_sha256"],
            "settings": parent["settings"], "purpose": "component_diagnosis_only",
            "sha256": {path.as_posix(): digest(path) for path in files},
            "execution_authorized": False, "replacement_approved": False}
    write_new(DIRECTORY / "plan.lock.json", lock)
    with (DIRECTORY / "plan.lock.sha256").open("x", encoding="utf-8") as stream:
        stream.write(digest(DIRECTORY / "plan.lock.json") + "\n")
    return verify()


def verify():
    parent = verify_parent()
    actual = digest(DIRECTORY / "plan.lock.json")
    if actual != (DIRECTORY / "plan.lock.sha256").read_text().strip():
        raise RuntimeError("V21 lock mismatch")
    lock = json.loads((DIRECTORY / "plan.lock.json").read_text(encoding="utf-8"))
    if lock["parent_lock_sha256"] != parent["lock_sha256"] or lock["settings"] != parent["settings"]:
        raise RuntimeError("V21 parent or predictor settings changed")
    for name, expected in lock["sha256"].items():
        if digest(name) != expected:
            raise RuntimeError(f"V21 frozen file changed: {name}")
    return {**lock, "lock_sha256": actual, "frozen_inputs_intact": True}
