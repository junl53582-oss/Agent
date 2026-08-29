import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from research_v20.freeze import digest, write_new
from research_v20r2.freeze import verify as verify_ledger
from research_v21.freeze import verify as verify_scores
from .config import V22Settings


DIRECTORY = Path("artifacts/research_v22")
PARENT_LEDGER = Path("artifacts/research_v20r2")
PARENT_SCORES = Path("artifacts/research_v21")


def settings_dict():
    return json.loads(json.dumps(asdict(V22Settings()), default=str))


def freeze():
    if any((DIRECTORY / name).exists() for name in ("plan.lock.json", "run.started.json", "report.json")):
        raise RuntimeError("V22 already frozen/started; do not overwrite")
    ledger, scores = verify_ledger(), verify_scores()
    receipt = json.loads((DIRECTORY / "test_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("passed") is not True:
        raise RuntimeError("tests must pass before freeze")
    files = sorted(Path("research_v22").glob("*.py"))
    files += [Path("tests/test_research_v22.py"), DIRECTORY / "protocol.json", DIRECTORY / "test_receipt.json",
              PARENT_LEDGER / "report.json", PARENT_LEDGER / "equity.csv", PARENT_LEDGER / "holdings.csv",
              PARENT_LEDGER / "daily_nav.csv", PARENT_LEDGER / "settlements.json", PARENT_SCORES / "report.json"]
    files += sorted(PARENT_SCORES.glob("scores_*.csv"))
    files += [Path("data/market_history_v10_hfq.csv"), Path("data/universes/000300/history_v10.csv"), V22Settings().action_path]
    lock = {"locked_at_utc": datetime.now(timezone.utc).isoformat(), "ledger_lock_sha256": ledger["lock_sha256"],
            "score_lock_sha256": scores["lock_sha256"], "settings": settings_dict(),
            "single_change": "portfolio_score=global_model_score", "sha256": {path.as_posix(): digest(path) for path in files},
            "execution_authorized": False, "replacement_approved": False}
    write_new(DIRECTORY / "plan.lock.json", lock)
    with (DIRECTORY / "plan.lock.sha256").open("x", encoding="utf-8") as stream:
        stream.write(digest(DIRECTORY / "plan.lock.json") + "\n")
    return verify()


def verify():
    ledger, scores = verify_ledger(), verify_scores()
    actual = digest(DIRECTORY / "plan.lock.json")
    if actual != (DIRECTORY / "plan.lock.sha256").read_text().strip():
        raise RuntimeError("V22 lock mismatch")
    lock = json.loads((DIRECTORY / "plan.lock.json").read_text(encoding="utf-8"))
    if lock["ledger_lock_sha256"] != ledger["lock_sha256"] or lock["score_lock_sha256"] != scores["lock_sha256"] or lock["settings"] != settings_dict():
        raise RuntimeError("V22 parents/settings changed")
    for name, expected in lock["sha256"].items():
        if digest(name) != expected:
            raise RuntimeError(f"V22 frozen file changed: {name}")
    return {**lock, "lock_sha256": actual, "frozen_inputs_intact": True}
