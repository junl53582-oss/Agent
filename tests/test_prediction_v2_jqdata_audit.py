from __future__ import annotations

import json
from pathlib import Path

import pytest

from stockpilot.prediction_v2_data.jqdata_audit import (
    _safe_account_info,
    classify_readiness,
    load_credentials,
    render_report,
)


def _result(start: str = "2025-05-28", end: str = "2026-06-04") -> dict:
    return {
        "sdk_version": "1.9.8",
        "server_version": "2.0.0",
        "account": {
            "date_range_start": start,
            "date_range_end": end,
            "expire_time": "2026-12-06",
        },
        "quota_after": {"total": 1_000_000, "spare": 999_900},
        "finance_catalog": {
            "table_count": 77,
            "relevant_tables": [
                "FUND_REPORT_DATE",
                "STK_FIN_FORCAST",
                "STK_PERFORMANCE_LETTERS",
                "STK_REPORT_DISCLOSURE",
            ],
        },
        "analyst_schema_validation": {
            "candidate_tables": [],
            "passed": False,
            "reason": "NO_ANALYST_VINTAGE_TABLE_DISCOVERED",
        },
        "table_audits": {
            "STK_FIN_FORCAST": {
                "oldest_observed": "2025-06-20",
                "newest_observed": "2026-08-25",
            }
        },
        "market_probe": {"passed": True},
        "fundamentals_probe": {"passed": True},
    }


def test_credentials_are_loaded_but_never_in_safe_account_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / ".env"
    path.write_text("JQDATA_USERNAME=secret-user\nJQDATA_PASSWORD=secret-password\n")
    monkeypatch.delenv("JQDATA_USERNAME", raising=False)
    monkeypatch.delenv("JQDATA_PASSWORD", raising=False)
    username, password, source = load_credentials(path)
    assert (username, password, source) == (
        "secret-user",
        "secret-password",
        "IGNORED_ENV_FILE",
    )
    safe = _safe_account_info(
        {
            "mob": "secret-user",
            "password": "secret-password",
            "date_range_start": "2025-05-28",
        }
    )
    encoded = json.dumps(safe)
    assert "secret-user" not in encoded
    assert "secret-password" not in encoded


def test_missing_credentials_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JQDATA_USERNAME", raising=False)
    monkeypatch.delenv("JQDATA_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="JQDATA_CREDENTIALS_NOT_AVAILABLE"):
        load_credentials(tmp_path / "missing.env")


def test_one_year_entitlement_and_no_analyst_tables_are_blocked() -> None:
    result = _result()
    decision = classify_readiness(result)
    result["decision"] = decision
    assert decision["status"] == "PREDICTION_V2_JQDATA_FOUNDATION_BLOCKED"
    assert decision["historical_coverage"] == "FAIL"
    assert decision["historical_analyst_expectations"] == "NOT_AVAILABLE"
    assert decision["challenger_experiment"] == "NOT_STARTED"
    report = render_report(result)
    assert "secret" not in report.lower()
    assert "Production Tencent-first routing changed: `FALSE`" in report


def test_five_year_history_still_requires_explicit_analyst_table() -> None:
    result = _result(start="2018-01-01", end="2026-06-04")
    decision = classify_readiness(result)
    assert decision["historical_coverage"] == "PASS"
    assert decision["historical_analyst_expectations"] == "NOT_AVAILABLE"
    assert decision["status"] == "PREDICTION_V2_JQDATA_FOUNDATION_BLOCKED"


def test_true_analyst_table_and_history_can_pass_foundation_gate() -> None:
    result = _result(start="2018-01-01", end="2026-06-04")
    result["finance_catalog"]["relevant_tables"].append("STK_ANALYST_CONSENSUS_VINTAGE")
    result["analyst_schema_validation"] = {
        "candidate_tables": ["STK_ANALYST_CONSENSUS_VINTAGE"],
        "passed": True,
        "reason": "CANONICAL_FIELDS_AND_REVISION_LINEAGE_VERIFIED",
    }
    decision = classify_readiness(result)
    assert decision["historical_coverage"] == "PASS"
    assert decision["historical_analyst_expectations"] == "PASS"
    assert decision["status"] == "PREDICTION_V2_JQDATA_FOUNDATION_READY"
