from __future__ import annotations

import json

import numpy as np
import pandas as pd

from stockpilot.alpha_diagnostic.gen3 import (
    EXPERIMENTS,
    NEW_FEATURES,
    REGIME_INTERACTIONS,
    Gen3Settings,
    add_gen3_features,
    feature_registry,
    freeze_protocol,
)
from stockpilot.research_challenger.metrics import moving_block_bootstrap_delta


def _feature_frame() -> pd.DataFrame:
    rows = []
    for day in pd.date_range("2024-01-02", periods=2):
        for index in range(60):
            rows.append(
                {
                    "date": day,
                    "symbol": f"{index:06d}",
                    "industry": "A" if index % 2 else "B",
                    "broad_sector": "technology" if index % 3 == 0 else "industrial",
                    "regime": "risk_off" if day.day == 2 else "risk_on",
                    "benchmark_weight_rank": index / 60,
                    "volatility_20": 0.1 + index / 1_000,
                    "volatility_60": 0.2 + index / 1_000,
                    "momentum_60": (index - 30) / 100,
                    "downside_volatility_60": 0.1 + index / 2_000,
                    "ret_5": (index - 20) / 200,
                    "ret_20": (index - 25) / 150,
                    "volume_ratio_20": 0.5 + index / 100,
                    "volume_trend_60": (index - 30) / 200,
                    "future_return_20d": 1000 - index,
                }
            )
    return pd.DataFrame(rows)


def test_gen3_features_are_deterministic_and_do_not_read_future_target() -> None:
    frame = _feature_frame()
    first = add_gen3_features(frame)
    modified = frame.copy()
    modified["future_return_20d"] *= -999
    second = add_gen3_features(modified)
    columns = [*NEW_FEATURES, *REGIME_INTERACTIONS]
    pd.testing.assert_frame_equal(first[columns], second[columns])
    assert np.isfinite(first[columns].to_numpy()).all()


def test_feature_registry_is_complete_and_explicitly_research_only() -> None:
    registry = feature_registry()
    assert {row["feature_name"] for row in registry} == {
        *NEW_FEATURES,
        *REGIME_INTERACTIONS,
    }
    assert all(row["pit_available"] for row in registry)
    assert all(not row["frozen_contract_modified"] for row in registry)
    assert all(row["inputs"] and row["lookback"] for row in registry)


def test_paired_block_bootstrap_is_deterministic() -> None:
    dates = pd.date_range("2020-01-01", periods=120)
    baseline = pd.Series(np.sin(np.arange(120)) / 10, index=dates)
    challenger = baseline + 0.01
    first = moving_block_bootstrap_delta(
        challenger, baseline, replications=100, block_length=20, seed=42
    )
    second = moving_block_bootstrap_delta(
        challenger, baseline, replications=100, block_length=20, seed=42
    )
    assert first == second
    assert first["ci_lower"] > 0


def test_protocol_freezes_small_hypothesis_driven_registry(tmp_path) -> None:
    settings = Gen3Settings(artifact_dir=tmp_path)
    protocol = freeze_protocol(settings)
    registry = json.loads((tmp_path / "experiment_registry.json").read_text("utf-8"))
    assert len(EXPERIMENTS) == 12
    assert len(registry) == 16
    assert protocol["nested_discipline"]["2025_untouched"] is False
    assert protocol["nested_discipline"]["manual_retuning_against_2025_after_protocol"] is False
    assert protocol["evaluation"]["random_split"] is False
    assert "champion promotion" in protocol["forbidden"]
