from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pit_data_v2.core import ObservationSettings, _observed_dates, capture_sources


def _expectation_pages():
    record = {"SECURITY_CODE": "000001", "SECURITY_NAME_ABBR": "A", "YEAR1": 2026, "EPS1": 1.0, "YEAR2": 2027, "EPS2": 1.2}
    return [(b"page-1", {"result": {"data": [record]}}), (b"page-2", {"result": {"data": [dict(record)]}})]


def _industry(path: Path):
    pd.DataFrame({"symbol": ["000001"], "industry_effective_date": ["2020-01-01"], "industry": ["Bank"]}).to_csv(path, index=False)


def test_forward_collector_uses_audited_duplicate_rule_and_isolates_flow_failure(tmp_path: Path):
    industry = tmp_path / "industry.csv"
    _industry(industry)
    settings = ObservationSettings(industry_path=industry)

    def failed_flow():
        raise ConnectionError("flow unavailable")

    sources = capture_sources(
        tmp_path / "observation",
        target_date="2026-08-31",
        now=datetime(2026, 8, 31, tzinfo=timezone.utc),
        watchlist={"000001"},
        settings=settings,
        expectation_fetcher=_expectation_pages,
        flow_fetcher=failed_flow,
    )
    assert sources["earnings_expectations"]["status"] == "complete"
    assert sources["earnings_expectations"]["duplicate_audit"]["duplicate_rows_removed"] == 1
    assert sources["fund_flows"]["status"] == "failed"


def test_baseline_date_prevents_same_date_recapture(tmp_path: Path):
    baseline = tmp_path / "baseline" / "one"
    baseline.mkdir(parents=True)
    (baseline / "manifest.json").write_text('{"observed_date_shanghai":"2026-08-30"}', encoding="utf-8")
    settings = ObservationSettings(baseline_root=tmp_path / "baseline", data_root=tmp_path / "new")
    assert _observed_dates(settings) == {"2026-08-30"}


def test_forward_settings_keep_model_and_execution_gates():
    settings = ObservationSettings()
    assert settings.minimum_training_observations == 20
    assert settings.minimum_expectation_coverage == 0.60
    assert settings.minimum_flow_coverage == 0.95
