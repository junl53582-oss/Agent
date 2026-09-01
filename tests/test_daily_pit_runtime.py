from __future__ import annotations

from datetime import datetime, timezone

import pytest

from stockpilot.daily_pit import runtime
from stockpilot.daily_pit.cli import main


def _valid_lock(settings) -> dict:
    del settings
    return {
        "effective_daily_input_lock_intact": True,
        "effective_operational_lock_intact": True,
        "daily_011_lock_intact": True,
        "failures": [],
    }


def test_seal_requires_011_before_delegating(monkeypatch) -> None:
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(runtime.runtime009, "seal_inputs", forbidden)
    with pytest.raises(runtime.DailyActivationLockError, match="DAILY_INPUT_LOCK_INVALID"):
        runtime.seal_inputs(
            "2026-09-02",
            effective_verifier=lambda settings: {
                "effective_daily_input_lock_intact": False,
                "failures": ["011"],
            },
        )
    assert called is False


def test_seal_delegates_only_after_daily_partition_binding(monkeypatch) -> None:
    settings = runtime.DailyRuntimeSettings()
    bound = object()
    monkeypatch.setattr(runtime, "_daily_009_settings", lambda date, value: bound)
    monkeypatch.setattr(
        runtime.runtime009,
        "seal_inputs",
        lambda date, now, settings: {"date": date, "bound": settings is bound},
    )
    result = runtime.seal_inputs(
        "2026-09-02",
        now=datetime(2026, 9, 2, 11, tzinfo=timezone.utc),
        settings=settings,
        effective_verifier=_valid_lock,
    )
    assert result["bound"] is True
    assert result["daily_011_verified"] is True


def test_permanently_blocked_date_cannot_be_backfilled(monkeypatch) -> None:
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(runtime.runtime009, "generate_prediction", forbidden)
    with pytest.raises(RuntimeError, match="2026-09-01_PERMANENTLY_BLOCKED"):
        runtime.generate_prediction(
            "2026-09-01",
            effective_verifier=_valid_lock,
        )
    assert called is False


def test_preflight_lock_failure_never_delegates(monkeypatch) -> None:
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(runtime.runtime009, "preflight", forbidden)
    result = runtime.preflight(
        "2026-09-02",
        effective_verifier=lambda settings: {
            "effective_daily_input_lock_intact": False,
            "failures": ["011"],
        },
    )
    assert result["daily_prediction_allowed"] is False
    assert result["status"] == "GEN2_EFFECTIVE_DAILY_INPUT_LOCK_INVALID"
    assert called is False


def test_cli_requires_explicit_real_provider_acknowledgement(monkeypatch) -> None:
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("stockpilot.daily_pit.cli.acquire_market", forbidden)
    with pytest.raises(SystemExit, match="REAL_PROVIDER_ACQUISITION_REQUIRED"):
        main(["acquire-market", "2026-09-02"])
    assert called is False
