from datetime import datetime, timezone

import pandas as pd

from pit_data_v1.core import (
    industry_prosperity,
    normalize_expectations,
    normalize_flows,
    sha256_bytes,
)


NOW = datetime(2026, 8, 30, 1, tzinfo=timezone.utc)


def page(payload):
    raw = b'{"frozen":"raw"}'
    return [(raw, payload)]


def test_expectation_snapshot_keeps_raw_probability_inputs_and_first_seen():
    payload = {
        "result": {
            "data": [
                {
                    "SECURITY_CODE": "000001",
                    "SECURITY_NAME_ABBR": "A",
                    "RATING_ORG_NUM": 8,
                    "YEAR1": 2026,
                    "EPS1": 1.2,
                    "YEAR2": 2027,
                    "EPS2": 1.5,
                },
                {"SECURITY_CODE": "000002", "SECURITY_NAME_ABBR": "B"},
            ]
        }
    }
    result = normalize_expectations(page(payload), {"000001"}, NOW)
    assert result["symbol"].tolist() == ["000001"]
    assert result.loc[0, "forecast_eps_1"] == 1.2
    assert result.loc[0, "observed_at_utc"] == NOW.isoformat()
    assert result.loc[0, "raw_page_sha256"] == sha256_bytes(b'{"frozen":"raw"}')


def test_flow_source_timestamp_cannot_be_replaced_by_observation_date():
    payload = {
        "data": {
            "diff": [
                {
                    "f12": "000001",
                    "f14": "A",
                    "f2": 10.0,
                    "f62": 100.0,
                    "f124": 1787875200,
                }
            ]
        }
    }
    result = normalize_flows(page(payload), {"000001"}, NOW)
    assert result.loc[0, "source_timestamp_utc"] != result.loc[0, "observed_at_utc"]
    assert result.loc[0, "main_net_inflow"] == 100.0


def test_industry_revision_requires_a_prior_same_forecast_year():
    current = pd.DataFrame(
        {
            "symbol": ["000001", "000002"],
            "industry": ["Bank", "Bank"],
            "forecast_year_1": [2026, 2026],
            "forecast_eps_1": [1.2, 2.0],
            "forecast_eps_2": [1.5, 2.2],
        }
    )
    baseline = industry_prosperity(current)
    assert baseline.loc[0, "revision_coverage"] == 0
    prior = current[["symbol", "forecast_year_1", "forecast_eps_1"]].copy()
    prior["forecast_eps_1"] = [1.0, 2.1]
    revised = industry_prosperity(current, prior)
    assert revised.loc[0, "revision_coverage"] == 2
    assert revised.loc[0, "positive_revision_breadth"] == 0.5


def test_training_gate_is_not_implicitly_promoted_by_capture():
    from pit_data_v1.core import ObservationSettings

    settings = ObservationSettings()
    assert settings.minimum_training_observations == 20
