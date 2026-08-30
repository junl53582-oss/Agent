from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

from stockpilot.prediction_forward_r2 import run_forward_r2

from .calendar import load_verified_calendar, validate_current_session
from .config import OperationalSettings
from .feature_store import build_feature_panel, write_feature_panel
from .freeze import verify_runtime_locks
from .integrity import read_verified_json, sha256_bytes, verify_immutable, write_immutable_json
from .observation import (
    capture_sources_once,
    load_verified_observations,
    reserve_daily_attempt,
)
from .readiness import derive_readiness
from .revision import SnapshotProof, build_revision_panel
from .sources import load_normalized_source, load_pit_context, production_source_fetchers


SHANGHAI = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class DailyDependencies:
    lock_verifier: Callable[[OperationalSettings], dict] = verify_runtime_locks
    context_loader: Callable[[str, OperationalSettings], tuple[pd.DataFrame, dict]] = load_pit_context
    source_fetcher_factory: Callable[[set[str], str, datetime, OperationalSettings], dict] = (
        production_source_fetchers
    )
    prediction_runner: Callable[[str, OperationalSettings], dict] | None = None
    settlement_runner: Callable[[str, OperationalSettings], dict] | None = None


def _proof(observation: dict, source: dict) -> SnapshotProof:
    raw = "".join(source.get("raw_response_sha256") or [])
    return SnapshotProof(
        observation_id=observation["observation_id"],
        observed_at=observation["observed_at"],
        snapshot_hash=source["normalized_data_sha256"],
        source_hash=sha256_bytes(raw.encode("ascii")),
    )


def _build_revision(
    current_observation: dict,
    current_expectations: pd.DataFrame | None,
    prior_observations: list[dict],
) -> pd.DataFrame | None:
    if current_expectations is None:
        return None
    current_source = current_observation["sources"]["earnings_expectations"]
    candidates = [
        item
        for item in prior_observations
        if item["observation_id"] != current_observation["observation_id"]
        and item.get("sources", {}).get("earnings_expectations", {}).get("success")
    ]
    if not candidates:
        return build_revision_panel(
            None,
            current_expectations,
            previous_proof=None,
            current_proof=_proof(current_observation, current_source),
        )
    previous_observation = sorted(candidates, key=lambda item: item["observed_at"])[-1]
    previous_source = previous_observation["sources"]["earnings_expectations"]
    previous = load_normalized_source(previous_source)
    return build_revision_panel(
        previous,
        current_expectations,
        previous_proof=_proof(previous_observation, previous_source),
        current_proof=_proof(current_observation, current_source),
    )


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


def _default_settlement_runner(target_date: str, settings: OperationalSettings) -> dict:
    # V1r2 refuses to infer a benchmark or company-action provenance.  A future
    # daily market ingest must provide both approved sources before settlement.
    del target_date, settings
    return {
        "status": "NOT_RUN_APPROVED_SOURCES_UNAVAILABLE",
        "mature_records_written": 0,
        "requires": [
            "immutable_market_source",
            "immutable_benchmark_source",
            "verified_corporate_action_manifest",
        ],
        "execution_authorized": False,
    }


def _write_daily_receipt(
    settings: OperationalSettings, target_date: str, payload: dict
) -> dict:
    path = settings.daily_receipts_root / f"{target_date}.json"
    write_immutable_json(path, payload)
    return payload | {
        "daily_receipt_path": path.as_posix(),
        "daily_receipt_sha256": verify_immutable(path),
    }


def run_daily(
    *,
    target_date: str | None = None,
    now: datetime | None = None,
    settings: OperationalSettings | None = None,
    dependencies: DailyDependencies | None = None,
) -> dict:
    """Run the only authorized prospective daily chain.

    Calendar validation occurs before reservation and all provider callbacks.
    Once the exclusive reservation succeeds, every terminal state is retained.
    """
    settings = settings or OperationalSettings()
    dependencies = dependencies or DailyDependencies()
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    actual_date = now.astimezone(SHANGHAI).date().isoformat()
    target_date = target_date or actual_date
    calendar = load_verified_calendar(settings.calendar_path)
    validate_current_session(target_date, actual_date, calendar)
    locks = dependencies.lock_verifier(settings)
    before_observations = load_verified_observations(settings)
    readiness_before = derive_readiness(
        before_observations, [], thresholds=settings.thresholds
    ).to_dict()
    attempt = reserve_daily_attempt(
        target_date,
        now,
        parent_lock_sha256=locks["v1r2_lock_sha256"],
        settings=settings,
    )
    base_receipt = {
        "version": settings.version,
        "date": target_date,
        "attempt_id": attempt["observation_attempt_id"],
        "reserved_at": attempt["reserved_at"],
        "reservation_sha256": attempt["reservation_sha256"],
        "completed_at": None,
        "git_commit_sha": attempt["git_commit"],
        "parent_locks": locks,
        "trading_calendar_hash": calendar.file_sha256,
        "automatic_retry": False,
        "manual_retry": False,
        "retry_allowed": False,
        "readiness_before": readiness_before,
        "replacement_evaluation_ready": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
        "v31_trained": False,
    }
    try:
        universe_panel, context = dependencies.context_loader(target_date, settings)
        universe = set(universe_panel["symbol"].astype(str).str.zfill(6))
        fetchers = dependencies.source_fetcher_factory(
            universe, target_date, now, settings
        )
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
        sources = observation["sources"]
        expectations = load_normalized_source(
            sources.get("earnings_expectations", {})
        )
        announcements = load_normalized_source(sources.get("announcements", {}))
        fund_flows = load_normalized_source(sources.get("fund_flows", {}))
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
                for name, value in sources.items()
            },
        )
        feature = write_feature_panel(
            panel,
            settings.features_root,
            source_provenance={
                "observation_sha256": observation["observation_sha256"],
                "membership_snapshot_sha256": context["membership_snapshot_sha256"],
                "industry_mapping_sha256": context["industry_mapping_sha256"],
            },
        )
        prediction_runner = dependencies.prediction_runner or _default_prediction_runner
        settlement_runner = dependencies.settlement_runner or _default_settlement_runner
        prediction = prediction_runner(target_date, settings)
        # Settlement is deliberately independent of today's source success.
        settlement = settlement_runner(target_date, settings)
        observations_after = load_verified_observations(settings)
        readiness_after = derive_readiness(
            observations_after, [], thresholds=settings.thresholds
        ).to_dict()
        terminal = base_receipt | {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": (
                "COMPLETE"
                if observation["status"] == "SUCCESS"
                else observation["status"]
            ),
            "universe_hash": observation["universe_hash"],
            "membership_hash": context["membership_snapshot_sha256"],
            "industry_hash": context["industry_mapping_sha256"],
            "observation": {
                "status": observation["status"],
                "sha256": observation["observation_sha256"],
                "source_statuses": {
                    name: value["source_status"] for name, value in sources.items()
                },
                "source_coverage": {
                    name: value.get("universe_coverage") for name, value in sources.items()
                },
                "raw_hashes": {
                    name: value.get("raw_response_sha256", []) for name, value in sources.items()
                },
                "normalized_hashes": {
                    name: value.get("normalized_data_sha256") for name, value in sources.items()
                },
                "network_request_count": observation["network_request_count"],
            },
            "feature_panel": {
                "path": feature["panel_path"],
                "sha256": feature["panel_sha256"],
                "manifest_sha256": feature["manifest_sha256"],
            },
            "prediction": prediction,
            "label_settlement": settlement,
            "qualified_observation": readiness_after["pit_observation_count"]
            > readiness_before["pit_observation_count"],
            "readiness_after": readiness_after,
        }
        return _write_daily_receipt(settings, target_date, terminal)
    except KeyboardInterrupt:
        terminal = base_receipt | {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": "INTERRUPTED",
            "failure_reason": "KeyboardInterrupt",
            "network_request_count": None,
            "readiness_after": readiness_before,
        }
        _write_daily_receipt(settings, target_date, terminal)
        raise
    except Exception as error:
        terminal = base_receipt | {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": "FAILED",
            "failure_reason": f"{type(error).__name__}: {error}",
            "network_request_count": None,
            "readiness_after": readiness_before,
        }
        _write_daily_receipt(settings, target_date, terminal)
        raise


def load_daily_receipt(path: str | Path) -> dict:
    return read_verified_json(path)


# Public name makes the operational policy explicit while keeping ``run_daily``
# available for concise internal calls and tests.
run_official_daily = run_daily
