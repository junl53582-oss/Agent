from pathlib import Path

import pandas as pd
import pytest

from research_v10.features import V10_FEATURES
from stockpilot.prediction_v2_data.jqdata_feature_overlap_audit import (
    AuditSettings,
    _assert_no_label_columns,
    _family,
    _novelty_class,
    _shortlist,
    _temporal_ready,
    load_gen2_features_only,
    sha256_file,
)


def test_label_columns_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="RETURN_LABEL_ACCESS_FORBIDDEN"):
        _assert_no_label_columns(["roe_rank", "future_return_20d"])


def test_gen2_loader_projects_only_safe_columns(tmp_path: Path) -> None:
    rows = 3
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=rows),
            "symbol": ["000001", "000002", "000003"],
            "broad_sector": ["A", "B", "C"],
            **{feature: [0.1, 0.2, 0.3] for feature in V10_FEATURES},
            "future_return_20d": [999.0, 999.0, 999.0],
            "v10_target_20": [999.0, 999.0, 999.0],
        }
    )
    path = tmp_path / "panel.parquet"
    frame.to_parquet(path, index=False)
    loaded = load_gen2_features_only(path, sha256_file(path))
    assert list(loaded.columns) == ["date", "symbol", "broad_sector", *V10_FEATURES]
    assert "future_return_20d" not in loaded
    assert "v10_target_20" not in loaded


def test_roles_do_not_treat_sparse_events_as_dense_features() -> None:
    settings = AuditSettings(Path("jq"), Path("gen2"), Path("protocol"), Path("artifacts"))
    event = {"observations": 120, "active_dates": 5, "median_symbols_per_active_date": 1, "unique_values": 120}
    continuous = {"observations": 120, "active_dates": 5, "median_symbols_per_active_date": 40, "unique_values": 120}
    assert _temporal_ready("EVENT_SPARSE", event, settings)
    assert not _temporal_ready("CONTINUOUS", continuous, settings)


def test_family_mapping_is_explicit() -> None:
    assert _family("jq_company_forecast_profit_max") == "company_forecast"
    assert _family("jq_valuation_pe_ratio_percentile") == "valuation"
    with pytest.raises(RuntimeError, match="UNKNOWN_JQ_FEATURE_FAMILY"):
        _family("jq_unknown_x")


def test_novelty_thresholds_are_frozen() -> None:
    settings = AuditSettings(Path("jq"), Path("gen2"), Path("protocol"), Path("artifacts"))
    assert _novelty_class(0.86, settings) == "HIGH_REDUNDANCY"
    assert _novelty_class(0.70, settings) == "PARTIAL_REDUNDANCY"
    assert _novelty_class(0.40, settings) == "LOW_REDUNDANCY"
    assert _novelty_class(None, settings) == "NOT_ESTIMABLE"


def test_shortlist_is_deterministic_and_capped() -> None:
    settings = AuditSettings(
        Path("jq"),
        Path("gen2"),
        Path("protocol"),
        Path("artifacts"),
        shortlist_target=20,
        shortlist_maximum=25,
    )
    rows = []
    for index in range(30):
        rows.append(
            {
                "feature": f"jq_quality_f{index:02d}",
                "family": "quality",
                "role": "CONTINUOUS",
                "selection_status": "KEEP_FOR_RESIDUAL_AUDIT",
                "active_dates": 200,
                "observations": 10_000,
                "median_symbols_per_active_date": 50,
                "maximum_abs_gen2_rank_corr": index / 100,
            }
        )
    first = _shortlist(pd.DataFrame(rows), settings)
    second = _shortlist(pd.DataFrame(rows), settings)
    assert len(first) == 20
    assert first["feature"].tolist() == second["feature"].tolist()
    assert not first["predictive_alpha_claim"].any()
