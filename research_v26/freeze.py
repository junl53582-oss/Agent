import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from research_v20.freeze import digest, write_new
from research_v20r2.freeze import verify as verify_ledger
from research_v21.freeze import verify as verify_scores
from research_v25r1.freeze import verify as verify_previous

from .config import V26Settings


DIRECTORY = Path("artifacts/research_v26")
PARENT_LEDGER = Path("artifacts/research_v20r2")
PARENT_SCORES = Path("artifacts/research_v21")
PREVIOUS = Path("artifacts/research_v25r1")


def settings_dict():
    return json.loads(json.dumps(asdict(V26Settings()), default=str))


def verified_previous_result():
    lock = verify_previous()
    report = json.loads((PREVIOUS / "report.json").read_text(encoding="utf-8"))
    if report.get("lock_sha256") != lock["lock_sha256"] or report.get("frozen_inputs_intact") is not True:
        raise RuntimeError("V25r1 result does not match its frozen lock")
    for name, expected in report.get("output_sha256", {}).items():
        if digest(PREVIOUS / name) != expected:
            raise RuntimeError(f"V25r1 output changed: {name}")
    if report.get("decision") != "keep_v6":
        raise RuntimeError("V25r1 decision evidence is incomplete")
    return report


def freeze():
    if any((DIRECTORY / name).exists() for name in ("plan.lock.json", "run.started.json", "report.json")):
        raise RuntimeError("V26 already frozen/started; preserve and use a new revision")
    ledger, scores, previous = verify_ledger(), verify_scores(), verified_previous_result()
    receipt = json.loads((DIRECTORY / "test_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("passed") is not True:
        raise RuntimeError("tests must pass before freeze")
    v16 = json.loads(Path("artifacts/research_v16/plan.lock.json").read_text(encoding="utf-8"))
    quality = v16.get("quality", {})
    if quality.get("passed") is not True or quality.get("fundamental_pit_violations") != 0 or quality.get("industry_future_violations") != 0:
        raise RuntimeError("parent PIT quality gates are not intact")
    files = sorted(Path("research_v26").glob("*.py"))
    files += [Path("tests/test_research_v26.py"), DIRECTORY / "protocol.json", DIRECTORY / "test_receipt.json",
              PREVIOUS / "report.json", PARENT_LEDGER / "report.json", PARENT_LEDGER / "equity.csv",
              PARENT_LEDGER / "holdings.csv", PARENT_LEDGER / "daily_nav.csv", PARENT_LEDGER / "settlements.json",
              PARENT_SCORES / "report.json", Path("artifacts/research_v16/plan.lock.json"),
              Path("data/market_history_v10_hfq.csv"), Path("data/universes/000300/history_v10.csv"),
              V26Settings().action_path]
    files += sorted(PARENT_SCORES.glob("scores_*.csv"))
    lock = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "ledger_lock_sha256": ledger["lock_sha256"], "score_lock_sha256": scores["lock_sha256"],
        "previous_lock_sha256": previous["lock_sha256"], "settings": settings_dict(),
        "single_change": "replace only the V10 LightGBM residual-magnitude objective with binary positive-residual probability",
        "data_quality": {key: quality[key] for key in ("membership_snapshots", "membership_min_size", "membership_max_size",
                                                         "market_symbols", "market_min", "market_max",
                                                         "fundamental_pit_violations", "industry_future_violations")},
        "sha256": {path.as_posix(): digest(path) for path in files},
        "execution_authorized": False, "replacement_approved": False,
    }
    write_new(DIRECTORY / "plan.lock.json", lock)
    (DIRECTORY / "plan.lock.sha256").write_text(digest(DIRECTORY / "plan.lock.json") + "\n", encoding="utf-8")
    return verify()


def verify():
    ledger, scores = verify_ledger(), verify_scores()
    actual = digest(DIRECTORY / "plan.lock.json")
    if actual != (DIRECTORY / "plan.lock.sha256").read_text().strip():
        raise RuntimeError("V26 lock mismatch")
    lock = json.loads((DIRECTORY / "plan.lock.json").read_text(encoding="utf-8"))
    if lock["ledger_lock_sha256"] != ledger["lock_sha256"] or lock["score_lock_sha256"] != scores["lock_sha256"] or lock["settings"] != settings_dict():
        raise RuntimeError("V26 parents/settings changed")
    for name, expected in lock["sha256"].items():
        if digest(name) != expected:
            raise RuntimeError(f"V26 frozen file changed: {name}")
    return {**lock, "lock_sha256": actual, "frozen_inputs_intact": True}

