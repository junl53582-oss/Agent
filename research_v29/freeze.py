import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from research_v20.freeze import digest, write_new
from research_v20r2.freeze import verify as verify_ledger
from research_v21.freeze import verify as verify_scores
from research_v28.freeze import verify as verify_v28

from .config import V29Settings


DIRECTORY = Path("artifacts/research_v29")
PARENT_LEDGER = Path("artifacts/research_v20r2")
PARENT_SCORES = Path("artifacts/research_v21")
V28 = Path("artifacts/research_v28")


def settings_dict():
    return json.loads(json.dumps(asdict(V29Settings()), default=str))


def completed_v28():
    lock = verify_v28()
    report = json.loads((V28 / "report.json").read_text(encoding="utf-8"))
    if report.get("lock_sha256") != lock["lock_sha256"] or report.get("decision") != "keep_v6":
        raise RuntimeError("V28 completed evidence mismatch")
    for name, expected in report["output_sha256"].items():
        if digest(V28 / name) != expected:
            raise RuntimeError(f"V28 output changed: {name}")
    return lock


def freeze():
    if any((DIRECTORY / name).exists() for name in ("plan.lock.json", "run.started.json", "report.json")):
        raise RuntimeError("V29 already frozen/started")
    ledger, scores, v28 = verify_ledger(), verify_scores(), completed_v28()
    receipt = json.loads((DIRECTORY / "test_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("passed") is not True or receipt.get("performance_outputs_read_before_freeze") is not False:
        raise RuntimeError("V29 tests or no-peeking receipt invalid")
    files = sorted(Path("research_v29").glob("*.py"))
    files += [Path("tests/test_research_v29.py"), DIRECTORY / "protocol.json", DIRECTORY / "test_receipt.json",
              V28 / "report.json", V28 / "plan.lock.json", PARENT_LEDGER / "report.json", PARENT_LEDGER / "equity.csv",
              PARENT_LEDGER / "holdings.csv", PARENT_LEDGER / "daily_nav.csv", PARENT_LEDGER / "settlements.json",
              PARENT_SCORES / "report.json", Path("artifacts/research_v16/plan.lock.json"),
              Path("artifacts/research_v28/data_admission.json"), Path("data/market_history_v10_hfq.csv"),
              Path("data/universes/000300/history_v10.csv"), V29Settings().action_path]
    files += sorted(PARENT_SCORES.glob("scores_*.csv"))
    lock = {"locked_at_utc": datetime.now(timezone.utc).isoformat(), "ledger_lock_sha256": ledger["lock_sha256"],
            "score_lock_sha256": scores["lock_sha256"], "v28_lock_sha256": v28["lock_sha256"],
            "settings": settings_dict(), "single_change": "date-and-PIT-sector conditional top-20-percent tail labels",
            "unchanged_from_v28": ["direction heads", "regression anchors", "confidence tiers", "portfolio", "three gates"],
            "retroactive_reapproval_prohibited": ["V25r1", "V26", "V28"],
            "sha256": {path.as_posix(): digest(path) for path in files},
            "execution_authorized": False, "replacement_approved": False}
    write_new(DIRECTORY / "plan.lock.json", lock)
    (DIRECTORY / "plan.lock.sha256").write_text(digest(DIRECTORY / "plan.lock.json") + "\n", encoding="utf-8")
    return verify()


def verify():
    ledger, scores = verify_ledger(), verify_scores()
    actual = digest(DIRECTORY / "plan.lock.json")
    if actual != (DIRECTORY / "plan.lock.sha256").read_text().strip():
        raise RuntimeError("V29 lock mismatch")
    lock = json.loads((DIRECTORY / "plan.lock.json").read_text(encoding="utf-8"))
    if lock["ledger_lock_sha256"] != ledger["lock_sha256"] or lock["score_lock_sha256"] != scores["lock_sha256"] or lock["settings"] != settings_dict():
        raise RuntimeError("V29 parents/settings changed")
    for name, expected in lock["sha256"].items():
        if digest(name) != expected:
            raise RuntimeError(f"V29 frozen file changed: {name}")
    return {**lock, "lock_sha256": actual, "frozen_inputs_intact": True}
