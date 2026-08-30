from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from stockpilot.prospective_r3.certification import certify_observation, persist_certification
from stockpilot.prospective_r3.orchestrator import DailyDependencies, run_daily as run_v1r3_daily
from stockpilot.prospective_r2.integrity import verify_immutable, write_immutable_json

from .config import OperationalSettings
from .preflight import DailyPreflightBlocked, run_preflight
from .settlement import run_operational_settlement


def _verify_operational_lock(settings: OperationalSettings) -> dict:
    from .freeze import verify_lock

    return verify_lock(settings)


def _adapt_lock(result: dict) -> dict:
    # V1r3's frozen certifier names the active operational parent generically.
    return {**result, "v1r3_lock_sha256": result["v1r4_lock_sha256"]}


def run_daily(
    *,
    target_date: str | None = None,
    now: datetime | None = None,
    settings: OperationalSettings | None = None,
    dependencies: DailyDependencies | None = None,
    operational_lock_verifier=None,
) -> dict:
    settings = settings or OperationalSettings()
    now = now or datetime.now(timezone.utc)
    verifier = operational_lock_verifier or _verify_operational_lock
    lock_adapter = lambda configured: _adapt_lock(verifier(configured))
    preflight = run_preflight(
        target_date=target_date,
        now=now,
        settings=settings,
        lock_verifier=lock_adapter,
    )
    if preflight["daily_run_allowed"] is not True:
        raise DailyPreflightBlocked(preflight)
    dependencies = dependencies or DailyDependencies()
    def certification_runner(observation: dict, configured: OperationalSettings) -> dict:
        result = certify_observation(
            observation,
            configured,
            lock_verifier=lock_adapter,
        )
        return persist_certification(result, configured)
    operational = replace(
        dependencies,
        lock_verifier=lock_adapter,
        certification_runner=certification_runner,
        settlement_runner=dependencies.settlement_runner or run_operational_settlement,
    )
    result = run_v1r3_daily(
        target_date=preflight["target_date"],
        now=now,
        settings=settings,
        dependencies=operational,
    )
    operational_receipt = {
        "version": settings.version,
        "target_date": preflight["target_date"],
        "preflight": preflight,
        "daily_receipt_path": result["daily_receipt_path"],
        "daily_receipt_sha256": result["daily_receipt_sha256"],
        "daily_status": result["daily_status"],
        "v6_modified": False,
        "v30_logic_modified": False,
        "v30r1_logic_modified": False,
        "v31_trained": False,
        "model_retrain_runs": 0,
        "factor_research_runs": 0,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    receipt_path = settings.operational_receipts_root / f"{preflight['target_date']}.json"
    digest = write_immutable_json(receipt_path, operational_receipt)
    return result | operational_receipt | {
        "operational_receipt_path": receipt_path.as_posix(),
        "operational_receipt_sha256": digest,
    }


run_official_daily = run_daily
