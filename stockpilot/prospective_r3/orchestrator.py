from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

from stockpilot.prediction_forward_r2 import run_forward_r2
from stockpilot.prospective_r2.calendar import (
    load_verified_calendar,
    validate_current_session,
)
from stockpilot.prospective_r2.feature_store import build_feature_panel, write_feature_panel
from stockpilot.prospective_r2.integrity import read_verified_json, verify_immutable, write_immutable_json
from stockpilot.prospective_r2.observation import (
    capture_sources_once,
    load_verified_observations,
    reserve_daily_attempt,
)
from stockpilot.prospective_r2.orchestrator import _build_revision
from stockpilot.prospective_r2.sources import (
    load_normalized_source,
    load_pit_context,
    production_source_fetchers,
)

from .certification import certify_observation, persist_certification
from .config import OperationalSettings
from .settlement import certify_label_record, load_verified_label_records, run_approved_settlement
from .status import aggregate_daily_status, build_runtime_status


SHANGHAI = ZoneInfo("Asia/Shanghai")


def _default_lock_verifier(settings: OperationalSettings) -> dict:
    from .freeze import verify_lock

    return verify_lock(settings)


def _default_certification(observation: dict, settings: OperationalSettings) -> dict:
    result = certify_observation(observation, settings)
    return persist_certification(result, settings)


@dataclass(frozen=True)
class DailyDependencies:
    lock_verifier: Callable[[OperationalSettings], dict] = _default_lock_verifier
    context_loader: Callable[[str, OperationalSettings], tuple[pd.DataFrame, dict]] = load_pit_context
    source_fetcher_factory: Callable[[set[str], str, datetime, OperationalSettings], dict] = (
        production_source_fetchers
    )
    certification_runner: Callable[[dict, OperationalSettings], dict] = _default_certification
    observation_certifier: Callable = certify_observation
    label_certifier: Callable = certify_label_record
    prediction_runner: Callable[[str, OperationalSettings], dict] | None = None
    settlement_runner: Callable[[str, OperationalSettings], dict] | None = None


def _default_prediction_runner(target_date: str, settings: OperationalSettings) -> dict:
    market = Path(settings.prediction_market_template.format(date=target_date))
    ranking = Path(settings.prediction_ranking_template.format(date=target_date))
    if not market.exists() or not ranking.exists():
        return {
            "status": "INPUT_NOT_AVAILABLE",
            "market_exists": market.exists(),
            "ranking_exists": ranking.exists(),
            "model_logic_changed": False,
            "execution_authorized": False,
        }
    result = run_forward_r2(market, target_date, ranking_path=ranking)
    return {"status": "RECORDED", "model_logic_changed": False, **result}


def _runtime_status(
    settings: OperationalSettings,
    observations: list[dict],
    labels: list[dict],
    dependencies: DailyDependencies,
) -> dict:
    return build_runtime_status(
        settings,
        observations,
        labels,
        observation_certifier=dependencies.observation_certifier,
        label_certifier=dependencies.label_certifier,
    ).to_dict()


def _write_daily_receipt(settings: OperationalSettings, target_date: str, value: dict) -> dict:
    target = settings.daily_receipts_root / f"{target_date}.json"
    write_immutable_json(target, value)
    return value | {
        "daily_receipt_path": target.as_posix(),
        "daily_receipt_sha256": verify_immutable(target),
    }


def run_daily(
    *,
    target_date: str | None = None,
    now: datetime | None = None,
    settings: OperationalSettings | None = None,
    dependencies: DailyDependencies | None = None,
) -> dict:
    settings = settings or OperationalSettings()
    dependencies = dependencies or DailyDependencies()
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    actual_date = now.astimezone(SHANGHAI).date().isoformat()
    target_date = target_date or actual_date

    # All of these happen before reservation and before constructing providers.
    calendar = load_verified_calendar(settings.calendar_path)
    validate_current_session(target_date, actual_date, calendar)
    locks = dependencies.lock_verifier(settings)
    if locks.get("frozen_inputs_intact") is not True:
        raise RuntimeError("V1r3 frozen inputs are not intact")
    before_observations = load_verified_observations(settings)
    before_labels = load_verified_label_records(settings.labels_root)
    readiness_before = _runtime_status(
        settings, before_observations, before_labels, dependencies
    )
    attempt = reserve_daily_attempt(
        target_date,
        now,
        parent_lock_sha256=locks["v1r3_lock_sha256"],
        settings=settings,
    )
    base = {
        "version": settings.version,
        "date": target_date,
        "attempt_id": attempt["observation_attempt_id"],
        "reserved_at": attempt["reserved_at"],
        "reservation_sha256": attempt["reservation_sha256"],
        "git_commit_sha": attempt["git_commit"],
        "parent_locks": locks,
        "trading_calendar_hash": calendar.file_sha256,
        "automatic_retry": False,
        "manual_retry": False,
        "retry_allowed": False,
        "readiness_before": readiness_before,
        "model_training_ready": False,
        "replacement_evaluation_ready": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
        "v31_trained": False,
    }

    observation: dict | None = None
    certification: dict | None = None
    feature: dict | None = None
    observation_stage = {"status": "FAILED", "failure_reason": None}
    prediction: dict = {"status": "INPUT_NOT_AVAILABLE"}
    settlement: dict = {"status": "SETTLEMENT_NOT_RUN", "mature_records_written": 0}
    try:
        universe_panel, context = dependencies.context_loader(target_date, settings)
        universe = set(universe_panel["symbol"].astype(str).str.zfill(6))
        fetchers = dependencies.source_fetcher_factory(universe, target_date, now, settings)
        observation = capture_sources_once(
            attempt=attempt,
            target_date=target_date,
            observed_at=now,
            universe=universe,
            source_fetchers=fetchers,
            membership_snapshot_hash=context["membership_snapshot_sha256"],
            industry_mapping_hash=context["industry_mapping_sha256"],
            trading_calendar_hash=calendar.file_sha256,
            settings=settings,
        )
        observation_stage = {
            "status": observation["status"],
            "sha256": observation["observation_sha256"],
            "network_request_count": observation["network_request_count"],
            "source_statuses": {
                name: item["source_status"] for name, item in observation["sources"].items()
            },
        }
        certification = dependencies.certification_runner(observation, settings)

        expectations = load_normalized_source(
            observation["sources"].get("earnings_expectations", {})
        )
        announcements = load_normalized_source(
            observation["sources"].get("announcements", {})
        )
        fund_flows = load_normalized_source(observation["sources"].get("fund_flows", {}))
        revision = _build_revision(
            observation, expectations, [*before_observations, observation]
        )
        panel = build_feature_panel(
            universe_panel,
            date=target_date,
            observation_id=observation["observation_id"],
            observation_hash=observation["observation_sha256"],
            expectations=expectations,
            announcements=announcements,
            fund_flows=fund_flows,
            revision=revision,
            source_provenance={
                name: {
                    "status": value["source_status"],
                    "receipt_sha256": value["receipt_sha256"],
                }
                for name, value in observation["sources"].items()
            },
        )
        feature = write_feature_panel(
            panel,
            settings.features_root,
            source_provenance={
                "observation_sha256": observation["observation_sha256"],
                "certification_sha256": certification.get("certification_sha256"),
                "membership_snapshot_sha256": context["membership_snapshot_sha256"],
                "industry_mapping_sha256": context["industry_mapping_sha256"],
            },
        )
        prediction_runner = dependencies.prediction_runner or _default_prediction_runner
        prediction = prediction_runner(target_date, settings)
    except KeyboardInterrupt:
        terminal = base | {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "daily_status": "INTERRUPTED",
            "observation": observation_stage,
            "prediction": {"status": "INTERRUPTED"},
            "label_settlement": {"status": "INTERRUPTED"},
            "readiness_after": readiness_before,
        }
        _write_daily_receipt(settings, target_date, terminal)
        raise
    except Exception as error:
        observation_stage = observation_stage | {
            "status": observation_stage.get("status") or "FAILED",
            "failure_reason": f"{type(error).__name__}: {error}",
        }
        prediction = {"status": "INPUT_NOT_AVAILABLE", "failure_reason": str(error)}

    # Mature historical labels are evaluated independently from today's source stage.
    try:
        settlement_runner = dependencies.settlement_runner or run_approved_settlement
        settlement = settlement_runner(target_date, settings)
    except KeyboardInterrupt:
        terminal = base | {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "daily_status": "INTERRUPTED",
            "observation": observation_stage,
            "prediction": prediction,
            "label_settlement": {"status": "INTERRUPTED"},
            "readiness_after": readiness_before,
        }
        _write_daily_receipt(settings, target_date, terminal)
        raise
    except Exception as error:
        settlement = {
            "status": "SETTLEMENT_NOT_RUN",
            "mature_records_written": 0,
            "failure_reason": f"{type(error).__name__}: {error}",
        }

    observations_after = load_verified_observations(settings)
    labels_after = load_verified_label_records(settings.labels_root)
    readiness_after = _runtime_status(
        settings, observations_after, labels_after, dependencies
    )
    daily_status = aggregate_daily_status(
        observation_stage["status"], prediction["status"], settlement["status"]
    )
    terminal = base | {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "daily_status": daily_status,
        "observation": observation_stage,
        "observation_certification": certification,
        "feature_panel": None
        if feature is None
        else {
            "path": feature["panel_path"],
            "sha256": feature["panel_sha256"],
            "manifest_sha256": feature["manifest_sha256"],
        },
        "prediction": prediction,
        "label_settlement": settlement,
        "readiness_after": readiness_after,
    }
    return _write_daily_receipt(settings, target_date, terminal)


def load_daily_receipt(path: str | Path) -> dict:
    return read_verified_json(path)


run_official_daily = run_daily
