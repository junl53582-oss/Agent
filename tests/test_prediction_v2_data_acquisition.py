from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from stockpilot.prediction_v2_data.contracts import (
    build_earnings_surprise,
    validate_actual_versions,
    validate_analyst_estimates,
    validate_announcement_documents,
)


def _protocol() -> dict:
    return json.loads(
        Path("artifacts/prediction_v2/data_acquisition/protocol.json").read_text(encoding="utf-8")
    )


def test_protocol_hash_is_bound() -> None:
    root = Path("artifacts/prediction_v2/data_acquisition")
    assert hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest() == (
        root / "protocol.json.sha256"
    ).read_text(encoding="ascii").strip()


def test_current_snapshot_shape_cannot_pass_analyst_contract() -> None:
    frame = pd.DataFrame(
        {"symbol": ["000001"], "observed_at_utc": ["2026-09-05T10:00:00+00:00"], "forecast_eps_1": [1.0]}
    )
    result = validate_analyst_estimates(frame, _protocol())
    assert result["passed"] is False
    assert "estimate_id" in result["missing_columns"]
    assert "forecast_period" in result["missing_columns"]


def test_announcement_contract_rejects_same_day_effective_date() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["000001"],
            "announcement_id": ["1"],
            "published_at_source": ["2025-01-02T00:00:00+08:00"],
            "effective_trading_date": ["2025-01-02"],
            "document_sha256": ["a" * 64],
            "text_sha256": ["b" * 64],
            "revision_of_announcement_id": [None],
            "source_uri": ["https://static.cninfo.com.cn/example.pdf"],
        }
    )
    result = validate_announcement_documents(frame, _protocol())
    assert result["checks"]["not_same_day_when_publication_time_is_date_only"] is False
    assert result["passed"] is False


def test_revision_rows_require_supersession_links() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["000001"],
            "estimate_id": ["e2"],
            "institution_id": ["i1"],
            "published_at": ["2025-01-02T08:00:00+08:00"],
            "forecast_period": ["2025-12-31"],
            "metric": ["EPS"],
            "estimate_value": [1.0],
            "currency": ["CNY/share"],
            "revision_status": ["REVISED"],
            "supersedes_estimate_id": [None],
            "raw_record_sha256": ["c" * 64],
        }
    )
    result = validate_analyst_estimates(frame, _protocol())
    assert result["checks"]["revisions_linked"] is False
    assert result["passed"] is False


def test_actual_revision_rows_require_supersession_links() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["000001"],
            "actual_id": ["a2"],
            "report_period": ["2025-12-31"],
            "metric": ["EPS"],
            "actual_value": [1.2],
            "published_at": ["2026-03-01T08:00:00+08:00"],
            "revision_status": ["REVISED"],
            "supersedes_actual_id": [None],
            "raw_record_sha256": ["d" * 64],
        }
    )
    result = validate_actual_versions(frame, _protocol())
    assert result["checks"]["revisions_linked"] is False
    assert result["passed"] is False


def test_surprise_uses_only_latest_pre_release_estimate_per_institution() -> None:
    estimates = pd.DataFrame(
        {
            "symbol": ["000001"] * 4,
            "estimate_id": ["e1", "e2", "e3", "future"],
            "institution_id": ["i1", "i1", "i2", "i2"],
            "published_at": [
                "2026-01-01T08:00:00+08:00",
                "2026-02-01T08:00:00+08:00",
                "2026-01-15T08:00:00+08:00",
                "2026-04-01T08:00:00+08:00",
            ],
            "forecast_period": ["2025-12-31"] * 4,
            "metric": ["EPS"] * 4,
            "estimate_value": [0.8, 1.0, 1.2, 99.0],
            "revision_status": ["ORIGINAL", "REVISED", "ORIGINAL", "REVISED"],
        }
    )
    actuals = pd.DataFrame(
        {
            "symbol": ["000001"],
            "actual_id": ["a1"],
            "report_period": ["2025-12-31"],
            "metric": ["EPS"],
            "actual_value": [1.3],
            "published_at": ["2026-03-01T08:00:00+08:00"],
        }
    )
    result = build_earnings_surprise(estimates, actuals)
    assert len(result) == 1
    assert result.loc[0, "estimate_count"] == 2
    assert result.loc[0, "consensus"] == 1.1
    assert abs(result.loc[0, "surprise"] - 0.2) < 1e-12
    assert result.loc[0, "strictly_pre_release"]
