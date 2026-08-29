import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from research_v20.freeze import digest, write_new
from research_v20r2.freeze import verify as verify_ledger
from research_v21.freeze import verify as verify_scores
from research_v26.freeze import verify as verify_previous

from .config import V28Settings


DIRECTORY = Path("artifacts/research_v28")
PARENT_LEDGER = Path("artifacts/research_v20r2")
PARENT_SCORES = Path("artifacts/research_v21")
PREVIOUS = Path("artifacts/research_v26")


def settings_dict():
    return json.loads(json.dumps(asdict(V28Settings()), default=str))


def completed_previous():
    lock = verify_previous()
    report = json.loads((PREVIOUS / "report.json").read_text(encoding="utf-8"))
    if report.get("lock_sha256") != lock["lock_sha256"] or report.get("decision") != "keep_v6":
        raise RuntimeError("V26 completed evidence mismatch")
    for name, expected in report["output_sha256"].items():
        if digest(PREVIOUS / name) != expected:
            raise RuntimeError(f"V26 output changed: {name}")
    return lock


def freeze():
    if any((DIRECTORY / name).exists() for name in ("plan.lock.json", "run.started.json", "report.json")):
        raise RuntimeError("V28 already frozen/started")
    ledger, scores, previous = verify_ledger(), verify_scores(), completed_previous()
    receipt = json.loads((DIRECTORY / "test_receipt.json").read_text(encoding="utf-8"))
    admission = json.loads((DIRECTORY / "data_admission.json").read_text(encoding="utf-8"))
    if receipt.get("passed") is not True or admission.get("new_features_admitted_this_version") != []:
        raise RuntimeError("tests or data admission do not match protocol")
    files = sorted(Path("research_v28").glob("*.py"))
    files += [Path("tests/test_research_v28.py"), DIRECTORY / "protocol.json", DIRECTORY / "data_admission.json",
              DIRECTORY / "test_receipt.json", PREVIOUS / "report.json", PARENT_LEDGER / "report.json",
              PARENT_LEDGER / "equity.csv", PARENT_LEDGER / "holdings.csv", PARENT_LEDGER / "daily_nav.csv",
              PARENT_LEDGER / "settlements.json", PARENT_SCORES / "report.json", Path("artifacts/research_v16/plan.lock.json"),
              Path("artifacts/research_v14/data_quality.json"), Path("artifacts/announcement_body_v4/report.json"),
              Path("data/market_history_v10_hfq.csv"), Path("data/universes/000300/history_v10.csv"), V28Settings().action_path]
    files += sorted(PARENT_SCORES.glob("scores_*.csv"))
    lock = {"locked_at_utc": datetime.now(timezone.utc).isoformat(), "ledger_lock_sha256": ledger["lock_sha256"],
            "score_lock_sha256": scores["lock_sha256"], "previous_lock_sha256": previous["lock_sha256"],
            "settings": settings_dict(), "single_change": "three-gate confidence-scaled direction/tail probability framework",
            "retroactive_reapproval_prohibited": ["V25r1", "V26"],
            "sha256": {path.as_posix(): digest(path) for path in files},
            "execution_authorized": False, "replacement_approved": False}
    write_new(DIRECTORY / "plan.lock.json", lock)
    (DIRECTORY / "plan.lock.sha256").write_text(digest(DIRECTORY / "plan.lock.json") + "\n", encoding="utf-8")
    return verify()


def verify():
    ledger, scores = verify_ledger(), verify_scores()
    actual = digest(DIRECTORY / "plan.lock.json")
    if actual != (DIRECTORY / "plan.lock.sha256").read_text().strip():
        raise RuntimeError("V28 lock mismatch")
    lock = json.loads((DIRECTORY / "plan.lock.json").read_text(encoding="utf-8"))
    if lock["ledger_lock_sha256"] != ledger["lock_sha256"] or lock["score_lock_sha256"] != scores["lock_sha256"] or lock["settings"] != settings_dict():
        raise RuntimeError("V28 parents/settings changed")
    for name, expected in lock["sha256"].items():
        if digest(name) != expected:
            raise RuntimeError(f"V28 frozen file changed: {name}")
    return {**lock, "lock_sha256": actual, "frozen_inputs_intact": True}
