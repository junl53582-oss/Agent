from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pit_data_v1r2.core import ObservationSettings, capture_sources


def expectation_pages():
    body = {
        "result": {
            "data": [
                {
                    "SECURITY_CODE": "000001",
                    "SECURITY_NAME_ABBR": "A",
                    "YEAR1": 2026,
                    "EPS1": 1.0,
                    "YEAR2": 2027,
                    "EPS2": 1.2,
                }
            ]
        }
    }
    return [(b'{"expectation":1}', body)]


def test_expectation_evidence_survives_flow_provider_failure(tmp_path: Path):
    industry = tmp_path / "industry.csv"
    pd.DataFrame(
        {
            "symbol": ["000001"],
            "industry_effective_date": ["2020-01-01"],
            "industry": ["Bank"],
        }
    ).to_csv(industry, index=False)
    settings = ObservationSettings(industry_path=industry)

    def failed_flow():
        raise ConnectionError("provider unavailable")

    result = capture_sources(
        tmp_path / "observation",
        target_date="2026-08-30",
        now=datetime(2026, 8, 30, tzinfo=timezone.utc),
        watchlist={"000001"},
        settings=settings,
        expectation_fetcher=expectation_pages,
        flow_fetcher=failed_flow,
    )
    assert result["expectations"]["status"] == "complete"
    assert result["fund_flows"]["status"] == "failed"
    assert (tmp_path / "observation" / "expectations.csv").exists()
    assert not (tmp_path / "observation" / "fund_flows.csv").exists()


def test_source_failure_never_promotes_model_readiness(tmp_path: Path):
    from pit_data_v1r2.core import _source_failure

    result = _source_failure(RuntimeError("x"))
    assert result["model_training_ready"] is False
    assert result["automatic_retry"] is False
