import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from research_v20.freeze import digest, write_new
from research_v20r2.freeze import verify as verify_ledger
from research_v21.freeze import verify as verify_scores

from .config import V25R1Settings


DIRECTORY = Path("artifacts/research_v25r1")
PARENT_LEDGER = Path("artifacts/research_v20r2")
PARENT_SCORES = Path("artifacts/research_v21")


def settings_dict():
    return json.loads(json.dumps(asdict(V25R1Settings()), default=str))


def freeze():
    if any((DIRECTORY / name).exists() for name in ("plan.lock.json", "run.started.json", "report.json")):
        raise RuntimeError("V25r1 already frozen/started; preserve and use a new revision")
    ledger, scores = verify_ledger(), verify_scores()
    receipt = json.loads((DIRECTORY / "test_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("passed") is not True:
        raise RuntimeError("tests must pass before freeze")
    v16 = json.loads(Path("artifacts/research_v16/plan.lock.json").read_text(encoding="utf-8"))
    quality = v16.get("quality", {})
    if quality.get("passed") is not True or quality.get("fundamental_pit_violations") != 0 or quality.get("industry_future_violations") != 0:
        raise RuntimeError("parent PIT data quality gates are not intact")
    files = sorted(Path("research_v25r1").glob("*.py"))
    files += [Path("tests/test_research_v25r1.py"), DIRECTORY / "protocol.json", DIRECTORY / "test_receipt.json",
              PARENT_LEDGER / "report.json", PARENT_LEDGER / "equity.csv", PARENT_LEDGER / "holdings.csv",
              PARENT_LEDGER / "daily_nav.csv", PARENT_LEDGER / "settlements.json", PARENT_SCORES / "report.json",
              Path("artifacts/research_v16/plan.lock.json"), Path("data/market_history_v10_hfq.csv"),
              Path("data/universes/000300/history_v10.csv"), V25R1Settings().action_path]
    files += sorted(PARENT_SCORES.glob("scores_*.csv"))
    lock = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "ledger_lock_sha256": ledger["lock_sha256"],
        "score_lock_sha256": scores["lock_sha256"],
        "settings": settings_dict(),
        "single_change": "replace the V10 full-window LightGBM contribution with an equal ensemble of fixed 8/5/3-year LightGBM windows",
        "data_quality": {
            "membership_snapshots": quality["membership_snapshots"],
            "membership_min_size": quality["membership_min_size"],
            "membership_max_size": quality["membership_max_size"],
            "market_symbols": quality["market_symbols"],
            "market_min": quality["market_min"],
            "market_max": quality["market_max"],
            "fundamental_pit_violations": quality["fundamental_pit_violations"],
            "industry_future_violations": quality["industry_future_violations"],
        },
        "sha256": {path.as_posix(): digest(path) for path in files},
        "execution_authorized": False,
        "replacement_approved": False,
    }
    write_new(DIRECTORY / "plan.lock.json", lock)
    (DIRECTORY / "plan.lock.sha256").write_text(digest(DIRECTORY / "plan.lock.json") + "\n", encoding="utf-8")
    return verify()


def verify():
    ledger, scores = verify_ledger(), verify_scores()
    actual = digest(DIRECTORY / "plan.lock.json")
    if actual != (DIRECTORY / "plan.lock.sha256").read_text().strip():
        raise RuntimeError("V25r1 lock mismatch")
    lock = json.loads((DIRECTORY / "plan.lock.json").read_text(encoding="utf-8"))
    if lock["ledger_lock_sha256"] != ledger["lock_sha256"] or lock["score_lock_sha256"] != scores["lock_sha256"] or lock["settings"] != settings_dict():
        raise RuntimeError("V25r1 parents/settings changed")
    for name, expected in lock["sha256"].items():
        if digest(name) != expected:
            raise RuntimeError(f"V25r1 frozen file changed: {name}")
    return {**lock, "lock_sha256": actual, "frozen_inputs_intact": True}

