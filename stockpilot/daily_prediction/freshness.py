"""Read-only freshness and integrity status for the formal DAILY prediction product."""

from __future__ import annotations

import json
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from stockpilot.prospective_r2.calendar import load_verified_calendar
from stockpilot.prospective_r2.integrity import verify_immutable

from .product import DailyPredictionSettings
from .product import latest as load_latest_prediction

SHANGHAI = ZoneInfo("Asia/Shanghai")
DEFAULT_SCHEDULED_TIME = time(18, 45)
KNOWN_NO_BACKFILL_DATES = ("2026-09-01", "2026-09-03", "2026-09-04")


def _iso_at(value: pd.Timestamp, clock: time) -> str:
    return f"{value.date().isoformat()}T{clock.isoformat()}+08:00"


def _calendar_position(
    now: datetime,
    settings: DailyPredictionSettings,
    scheduled_time: time,
) -> dict[str, Any]:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    calendar = load_verified_calendar(settings.runtime_settings.calendar_path)
    sessions = calendar.sessions()
    local = now.astimezone(SHANGHAI)
    today = pd.Timestamp(local.date())
    window_open = local.timetz().replace(tzinfo=None) >= settings.earliest_prediction_time

    completed = sessions[(sessions < today) | ((sessions == today) & window_open)]
    expected = completed[-1] if len(completed) else None
    if calendar.is_session(today) and not window_open:
        future = sessions[sessions >= today]
    else:
        future = sessions[sessions > today]
    next_target = future[0] if len(future) else None
    return {
        "calendar": calendar,
        "sessions": sessions,
        "local": local,
        "expected": expected,
        "next_target": next_target,
        "earliest_legal_time": (
            _iso_at(next_target, settings.earliest_prediction_time)
            if next_target is not None
            else None
        ),
        "scheduled_time": (
            _iso_at(next_target, scheduled_time) if next_target is not None else None
        ),
    }


def _base_status(position: dict[str, Any], settings: DailyPredictionSettings) -> dict[str, Any]:
    calendar = position["calendar"]
    expected = position["expected"]
    next_target = position["next_target"]
    return {
        "as_of_shanghai": position["local"].isoformat(),
        "expected_latest_session": expected.date().isoformat() if expected is not None else None,
        "next_verified_prediction_date": (
            next_target.date().isoformat() if next_target is not None else None
        ),
        "earliest_legal_time": position["earliest_legal_time"],
        "scheduled_time": position["scheduled_time"],
        "calendar_source": calendar.source,
        "calendar_hash": calendar.file_sha256,
        "model_id": settings.model_id,
        "known_no_backfill_dates": list(KNOWN_NO_BACKFILL_DATES),
        "read_only": True,
        "provider_requests": 0,
        "model_runs": 0,
        "broker_requests": 0,
        "execution_authorized": False,
    }


def _invalid(base: dict[str, Any], reason: str) -> dict[str, Any]:
    return base | {
        "freshness_status": "INVALID",
        "prediction_status": "PREDICTION_BLOCKED",
        "integrity_status": "INVALID",
        "latest_prediction_date": None,
        "lag_sessions": None,
        "reason": reason,
        "prediction": None,
    }


def evaluate_daily_prediction_freshness(
    *,
    now: datetime | None = None,
    settings: DailyPredictionSettings | None = None,
    scheduled_time: time = DEFAULT_SCHEDULED_TIME,
) -> dict[str, Any]:
    """Verify the formal product and compare it with the latest completed session.

    The function is intentionally read-only. It never invokes acquisition, feature
    materialization, model inference, backfill, settlement, or broker code.
    """

    settings = settings or DailyPredictionSettings()
    now = now or datetime.now(timezone.utc)
    try:
        position = _calendar_position(now, settings, scheduled_time)
    except Exception as error:  # noqa: BLE001 - invalid calendar must be user-visible
        return {
            "freshness_status": "INVALID",
            "prediction_status": "PREDICTION_BLOCKED",
            "integrity_status": "INVALID",
            "reason": f"CALENDAR_INVALID:{type(error).__name__}:{error}",
            "prediction": None,
            "read_only": True,
            "provider_requests": 0,
            "model_runs": 0,
            "broker_requests": 0,
            "execution_authorized": False,
        }

    base = _base_status(position, settings)
    if not settings.latest_path.is_file():
        return base | {
            "freshness_status": "NO_FORMAL_PREDICTION",
            "prediction_status": "NO_FORMAL_PREDICTION",
            "integrity_status": "NOT_AVAILABLE",
            "latest_prediction_date": None,
            "lag_sessions": None,
            "reason": "NO_FORMAL_DAILY_PREDICTION_EXISTS",
            "prediction": None,
        }

    try:
        pointer = json.loads(settings.latest_path.read_text(encoding="utf-8"))
        prediction_date = str(pointer["prediction_date"])
        directory = Path(pointer["artifact_path"])
        expected_directory = settings.prediction_root / prediction_date
        if directory.resolve() != expected_directory.resolve():
            raise ValueError("LATEST_POINTER_OUTSIDE_EXPECTED_DATE_DIRECTORY")
        manifest_hash = verify_immutable(directory / "prediction_manifest.json")
        prediction_hash = verify_immutable(directory / "prediction.json")
        if pointer.get("prediction_manifest_hash") != manifest_hash:
            raise ValueError("LATEST_POINTER_MANIFEST_HASH_MISMATCH")
        if pointer.get("prediction_json_hash") != prediction_hash:
            raise ValueError("LATEST_POINTER_PREDICTION_HASH_MISMATCH")
        value = load_latest_prediction(settings)
    except Exception as error:  # noqa: BLE001 - malformed product must fail closed
        return _invalid(base, f"FORMAL_PRODUCT_INVALID:{type(error).__name__}:{error}")

    prediction_timestamp = pd.Timestamp(prediction_date).normalize()
    sessions: pd.DatetimeIndex = position["sessions"]
    expected: pd.Timestamp | None = position["expected"]
    if prediction_timestamp not in sessions:
        return _invalid(base, "FORMAL_PREDICTION_DATE_NOT_VERIFIED_SESSION")
    if expected is None or prediction_timestamp > expected:
        return _invalid(base, "FORMAL_PREDICTION_DATE_IS_AFTER_LATEST_COMPLETED_SESSION")
    if value.get("status") != "PREDICTION_AVAILABLE":
        return _invalid(base, "FORMAL_PRODUCT_STATUS_NOT_AVAILABLE")

    lag = int(((sessions > prediction_timestamp) & (sessions <= expected)).sum())
    freshness = "CURRENT" if lag == 0 else "STALE"
    return base | {
        "freshness_status": freshness,
        "prediction_status": "PREDICTION_AVAILABLE",
        "integrity_status": "VALID",
        "latest_prediction_date": prediction_date,
        "lag_sessions": lag,
        "reason": "LATEST_COMPLETED_SESSION_MATCH" if lag == 0 else "FORMAL_PREDICTION_STALE",
        "prediction": value,
    }


def trading_session_lag(
    older_date: str,
    expected_date: str | None,
    *,
    settings: DailyPredictionSettings | None = None,
) -> int | None:
    """Return verified-session lag for a research snapshot, or ``None`` if invalid."""

    if expected_date is None:
        return None
    settings = settings or DailyPredictionSettings()
    sessions = load_verified_calendar(settings.runtime_settings.calendar_path).sessions()
    older = pd.Timestamp(older_date).normalize()
    expected = pd.Timestamp(expected_date).normalize()
    if older not in sessions or expected not in sessions or older > expected:
        return None
    return int(((sessions > older) & (sessions <= expected)).sum())
