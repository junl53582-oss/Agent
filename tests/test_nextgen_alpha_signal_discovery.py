from __future__ import annotations

import json

import numpy as np
import pandas as pd

from stockpilot.alpha_diagnostic.nextgen import (
    EXPERIMENTS,
    LABELS,
    SIGNAL_FAMILIES,
    NextgenSettings,
    add_nextgen_labels,
    build_nextgen_signals,
    freeze_protocol,
    signal_registry,
)


def _panel() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2023-01-02", periods=140, freq="B")
    for symbol_index in range(24):
        for date_index, date in enumerate(dates):
            value = 100 + symbol_index + date_index * (0.01 + symbol_index / 10_000)
            ret = np.sin(date_index / 7 + symbol_index) / 100
            rows.append(
                {
                    "date": date,
                    "symbol": f"{symbol_index:06d}",
                    "broad_sector": "technology" if symbol_index % 2 else "industrial",
                    "industry": "A" if symbol_index % 3 else "B",
                    "ret_1": ret,
                    "ret_20": ret * 5,
                    "close": value,
                    "amount": 1_000_000 + symbol_index * 10_000 + date_index * 100,
                    "volume_ratio_20": 1 + np.cos(date_index / 9) / 10,
                    "future_return_20d": (symbol_index - 12) / 100 + date_index / 100_000,
                    "return_rank_20d": (symbol_index + 1) / 24,
                    "industry_alpha_rank_20d": ((symbol_index * 7) % 24 + 1) / 24,
                    "benchmark_weight": symbol_index + 1,
                    "volatility_20": 0.1 + symbol_index / 1_000,
                }
            )
    return pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)


def test_nextgen_signals_are_past_only_and_deterministic() -> None:
    frame = _panel()
    first = build_nextgen_signals(frame)
    changed = frame.copy()
    changed["future_return_20d"] *= -999
    second = build_nextgen_signals(changed)
    features = [feature for family in SIGNAL_FAMILIES.values() for feature in family]
    pd.testing.assert_frame_equal(first[features], second[features])
    assert np.isfinite(first[features].to_numpy()).all()


def test_label_construction_keeps_future_values_out_of_features() -> None:
    signals = build_nextgen_signals(_panel())
    labeled = add_nextgen_labels(signals)
    pd.testing.assert_series_equal(
        labeled["target_gen2_rank"], signals["return_rank_20d"], check_names=False
    )
    assert labeled["target_beta_residual"].between(0, 1).all()
    assert labeled["target_vol_adjusted"].between(0, 1).all()


def test_market_neutral_rank_is_analytically_identical() -> None:
    frame = _panel()
    values = frame["future_return_20d"]
    market_neutral = values - values.groupby(frame["date"]).transform("mean")
    raw_rank = values.groupby(frame["date"]).rank(pct=True)
    neutral_rank = market_neutral.groupby(frame["date"]).rank(pct=True)
    pd.testing.assert_series_equal(raw_rank, neutral_rank)
    label = next(item for item in LABELS if item["label_id"] == "L3_MARKET_NEUTRAL_RANK")
    assert label["status"] == "NOT_TRAINED_ANALYTICALLY_IDENTICAL_TO_L0_RANK"


def test_registry_contains_only_three_new_signal_families() -> None:
    evaluable = [row for row in signal_registry() if row["status"] == "PRE_REGISTERED_EVALUABLE"]
    assert len(evaluable) == 12
    assert {row["family"] for row in evaluable} == set(SIGNAL_FAMILIES)
    assert all(row["pit_availability"] for row in evaluable)
    assert not any("future" in source for row in evaluable for source in row["source_columns"])


def test_protocol_and_resume_state_freeze_bounded_budget(tmp_path) -> None:
    settings = NextgenSettings(artifact_dir=tmp_path)
    protocol = freeze_protocol(settings)
    state = json.loads((tmp_path / "current_research_state.json").read_text("utf-8"))
    registry = json.loads((tmp_path / "experiment_registry.json").read_text("utf-8"))
    assert len(EXPERIMENTS) == 12
    assert len(registry) == settings.experiment_config_budget
    assert protocol["nested_discipline"]["untouched_holdout"] is False
    assert state["current_phase"] == "PROTOCOL_FROZEN"
    assert state["pending_experiments"] == [item["id"] for item in EXPERIMENTS]
