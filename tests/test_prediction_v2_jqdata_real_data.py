from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from stockpilot.prediction_v2_data.jqdata_real_data import (
    Runtime,
    Settings,
    _next_sessions,
    credentials_from_environment,
    store_raw_partition,
)


def _settings(tmp_path: Path, budget: int = 800_000) -> Settings:
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    return Settings(tmp_path / "data", tmp_path / "artifacts", protocol, date(2026, 9, 5), budget)


def test_credentials_are_environment_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JQDATA_USERNAME", "user-secret")
    monkeypatch.setenv("JQDATA_PASSWORD", "password-secret")
    assert credentials_from_environment() == ("user-secret", "password-secret")
    monkeypatch.delenv("JQDATA_PASSWORD")
    with pytest.raises(RuntimeError, match="NOT_AVAILABLE_IN_ENVIRONMENT"):
        credentials_from_environment()


def test_date_level_publication_uses_next_verified_session() -> None:
    calendar = pd.DatetimeIndex(["2026-06-01", "2026-06-02", "2026-06-04", "2026-06-05"])
    result = _next_sessions(pd.Series(["2026-06-01", "2026-06-02", "2026-06-05"]), calendar)
    assert result.iloc[0] == pd.Timestamp("2026-06-02")
    assert result.iloc[1] == pd.Timestamp("2026-06-04")
    assert pd.isna(result.iloc[2])


def test_raw_partition_is_hash_bound_and_contains_no_identity(tmp_path: Path) -> None:
    jq = SimpleNamespace(get_query_count=lambda: {"spare": 1_000_000})
    runtime = Runtime(jq, _settings(tmp_path), 1_000_000)
    frame = pd.DataFrame(
        {"date": ["2026-06-01"], "code": ["000001.XSHE"], "value": [1.0]}
    )
    stored, receipt = store_raw_partition(
        runtime,
        "TEST_DATA",
        {"date": "2026-06-01"},
        frame,
        "date",
        "PIT_SAFE",
        1,
    )
    raw = runtime.settings.root / receipt["raw_path"]
    assert len(stored) == 1
    assert raw.exists()
    assert receipt["row_count"] == 1
    assert receipt["account_identity_persisted"] is False
    assert receipt["credential_values_persisted"] is False
    encoded = json.dumps(receipt)
    assert "user-secret" not in encoded
    assert "password-secret" not in encoded


def test_immutable_partition_conflict_fails_closed(tmp_path: Path) -> None:
    jq = SimpleNamespace(get_query_count=lambda: {"spare": 1_000_000})
    runtime = Runtime(jq, _settings(tmp_path), 1_000_000)
    first = pd.DataFrame({"date": ["2026-06-01"], "code": ["000001.XSHE"], "value": [1.0]})
    second = first.assign(value=2.0)
    arguments = (runtime, "TEST_DATA", {"date": "2026-06-01"})
    store_raw_partition(*arguments, first, "date", "PIT_SAFE", 1)
    with pytest.raises(RuntimeError, match="IMMUTABLE_PARTITION_CONFLICT"):
        store_raw_partition(*arguments, second, "date", "PIT_SAFE", 1)


def test_local_quota_budget_pauses_before_query(tmp_path: Path) -> None:
    jq = SimpleNamespace(get_query_count=lambda: {"spare": 900})
    runtime = Runtime(jq, _settings(tmp_path, budget=50), 1_000)
    called = False

    def provider_call() -> None:
        nonlocal called
        called = True

    with pytest.raises(RuntimeError, match="JQDATA_DAILY_QUOTA_PAUSED"):
        runtime.query("TOO_LARGE", 1, provider_call)
    assert called is False
    state = json.loads((runtime.settings.root / "state/checkpoint.json").read_text())
    assert state["status"] == "JQDATA_DAILY_QUOTA_PAUSED"


def test_frozen_factor_protocol_is_bounded_and_label_free() -> None:
    path = Path("artifacts/prediction_v2/jqdata_real_data/protocol.json")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    groups = protocol["factor_selection"]
    factors = [
        factor
        for group in ("quality", "growth", "risk", "momentum", "emotion")
        for factor in groups[group]
    ]
    assert len(factors) == 20
    assert len(set(factors)) == 20
    assert protocol["scope"]["return_labels_read"] is False
    assert protocol["scope"]["rank_ic_computed"] is False
