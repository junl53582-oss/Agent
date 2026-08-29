from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stockpilot.prediction.calibration import PlattCalibrator
from stockpilot.prediction.certification import PredictionCertificationResult
from stockpilot.prediction.confidence import confidence_scores
from stockpilot.prediction.data import pit_data_audit
from stockpilot.prediction.drift import feature_drift_report
from stockpilot.prediction.labels import add_prediction_labels
from stockpilot.prediction.metrics import binary_metrics, calibration_table
from stockpilot.prediction.models import LogisticRidge
from stockpilot.prediction.settlement import settle_predictions
from stockpilot.prediction.split import PurgedWalkForwardSplit
from stockpilot.prediction.storage import write_immutable_prediction_snapshot


def _panel(days: int = 80, symbols: int = 3, start: str = "2020-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=days)
    rows = []
    for date_number, date in enumerate(dates):
        for symbol_number in range(symbols):
            rows.append({
                "date": date, "symbol": f"{symbol_number + 1:06d}",
                "open": 10 + date_number + symbol_number / 10, "close": 10.5 + date_number,
                "eligible": True,
            })
    return pd.DataFrame(rows)


def _labeled() -> pd.DataFrame:
    return add_prediction_labels(_panel(), thresholds={1: 0.0021, 5: 0.0021, 20: 0.0021})


def test_no_future_label_in_training() -> None:
    frame = add_prediction_labels(_panel(600, start="2018-01-01"), thresholds={1: 0, 5: 0, 20: 0})
    split = PurgedWalkForwardSplit((2020,), 6)
    folds = split.split(frame, 5)
    assert pd.to_datetime(frame.loc[folds[0].train_index, "label_end_date_5d"]).max() < folds[0].validation_start


def test_purge_gap_respected() -> None:
    frame = add_prediction_labels(_panel(600, start="2018-01-01"), thresholds={1: 0, 5: 0, 20: 0})
    fold = PurgedWalkForwardSplit((2020,), 6).split(frame, 5)[0]
    calendar = pd.DatetimeIndex(pd.to_datetime(frame["date"]).drop_duplicates().sort_values())
    assert calendar.get_loc(fold.validation_start) - calendar.get_loc(fold.purge_cutoff) == 6


def test_20d_label_maturity() -> None:
    frame = _labeled()
    first = frame.sort_values(["date", "symbol"]).iloc[0]
    dates = pd.bdate_range("2020-01-01", periods=80)
    assert first["entry_date"] == dates[1]
    assert first["label_end_date_20d"] == dates[21]
    assert first["future_return_20d"] == pytest.approx((10 + 21) / (10 + 1) - 1)


def test_prediction_probability_range() -> None:
    frame = pd.DataFrame({"x": np.linspace(-2, 2, 100), "y": [0, 1] * 50})
    probability = LogisticRidge().fit(frame, ["x"], "y").predict_proba(frame)
    assert np.all((probability > 0) & (probability < 1))


def test_probability_not_rank() -> None:
    frame = pd.DataFrame({"x": np.linspace(-3, 3, 100), "y": ([0] * 50) + ([1] * 50)})
    probability = LogisticRidge().fit(frame, ["x"], "y").predict_proba(frame)
    rank = pd.Series(probability).rank(pct=True).to_numpy()
    assert not np.allclose(probability, rank)


def test_calibrator_uses_oof_only() -> None:
    with pytest.raises(ValueError, match="overlap"):
        PlattCalibrator().fit(
            np.linspace(0.2, 0.8, 40), np.array([0, 1] * 20),
            calibration_ids={"same"}, model_training_ids={"same"},
        )


def test_random_split_forbidden() -> None:
    with pytest.raises(ValueError, match="random"):
        PurgedWalkForwardSplit((2020,), 2, shuffle=True)


def test_latest_prediction_uses_pit_only() -> None:
    frame = _panel(5, 1)
    frame["membership_snapshot_date"] = frame["date"]
    frame["available_date"] = frame["date"]
    frame["industry_effective_date"] = frame["date"]
    frame.loc[0, "available_date"] = frame.loc[0, "date"] + pd.Timedelta(days=1)
    audit = pit_data_audit(frame)
    assert not audit["checks"]["fundamentals_not_future"]


def test_prediction_snapshot_immutable(tmp_path: Path) -> None:
    frame = pd.DataFrame({"date": ["2026-01-01"], "symbol": ["000001"], "rank_5d": [1], "p_up_5d": [0.6]})
    path = tmp_path / "2026-01-01.csv"
    assert write_immutable_prediction_snapshot(frame, path)[0]
    assert not write_immutable_prediction_snapshot(frame, path)[0]


def test_prediction_snapshot_hash_mismatch_fails(tmp_path: Path) -> None:
    frame = pd.DataFrame({"date": ["2026-01-01"], "symbol": ["000001"], "rank_5d": [1], "p_up_5d": [0.6]})
    path = tmp_path / "2026-01-01.csv"
    write_immutable_prediction_snapshot(frame, path)
    changed = frame.assign(p_up_5d=0.7)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        write_immutable_prediction_snapshot(changed, path)


def _passed_checks() -> dict[str, bool]:
    return {
        "data_verified": True, "pit_verified": True, "label_maturity_verified": True,
        "leakage_test_passed": True, "purged_walk_forward_passed": True,
        "calibration_passed": True, "baseline_beaten": True, "stability_passed": True,
        "regime_passed": True, "probability_quality_passed": True,
        "cost_aware_stress_passed": True,
    }


def test_126_days_not_required_for_prediction_ready() -> None:
    result = PredictionCertificationResult.evaluate(future_126d_confirmed=False, **_passed_checks())
    assert result.production_prediction_ready
    assert not result.future_126d_confirmed


def test_execution_authorized_remains_false() -> None:
    result = PredictionCertificationResult.evaluate(future_126d_confirmed=True, **_passed_checks())
    assert result.production_prediction_ready and not result.execution_authorized


def test_failed_validation_blocks_prediction_ready() -> None:
    checks = _passed_checks()
    checks["baseline_beaten"] = False
    result = PredictionCertificationResult.evaluate(**checks)
    assert not result.production_prediction_ready


def test_drift_reduces_confidence() -> None:
    reference = pd.DataFrame({"x": np.linspace(-1, 1, 1000)})
    current = pd.DataFrame({"x": np.linspace(10, 11, 100)})
    _, status, multiplier = feature_drift_report(reference, current, ["x"])
    stable, _ = confidence_scores(pd.Series([0.7]), oos_skill=1, calibration_quality=1,
                                  regime_consistency=1, sector_stability=1,
                                  feature_completeness=pd.Series([1]), drift_multiplier=1)
    drifted, _ = confidence_scores(pd.Series([0.7]), oos_skill=1, calibration_quality=1,
                                   regime_consistency=1, sector_stability=1,
                                   feature_completeness=pd.Series([1]), drift_multiplier=multiplier)
    assert status == "SEVERE" and drifted.iloc[0] < stable.iloc[0]


def test_probability_bucket_calibration() -> None:
    table = calibration_table(np.array([0, 1, 0, 1]), np.array([0.2, 0.55, 0.6, 0.8]))
    assert table["sample_size"].sum() == 4
    assert "actual_up_rate" in table


def test_baseline_comparison() -> None:
    y = np.array([0, 0, 1, 1])
    skilled = binary_metrics(y, np.array([0.1, 0.3, 0.7, 0.9]))
    naive = binary_metrics(y, np.full(4, 0.5))
    assert skilled["brier"] < naive["brier"] and skilled["log_loss"] < naive["log_loss"]


def test_prediction_settlement() -> None:
    market = _panel(30, 1)
    prediction = pd.DataFrame({
        "date": [market["date"].min()], "symbol": ["000001"],
        "p_up_1d": [0.6], "p_up_5d": [0.7], "p_up_20d": [0.8],
        "expected_return_5d": [0.01], "expected_return_20d": [0.02],
    })
    ledger = settle_predictions(prediction, market)
    assert len(ledger) == 3 and ledger["settled"].all()
    assert ledger.loc[ledger["horizon"] == 5, "exit_date"].iloc[0] == pd.bdate_range("2020-01-01", periods=30)[6]


def test_prediction_api(monkeypatch: pytest.MonkeyPatch) -> None:
    import stockpilot.api as api

    snapshot = pd.DataFrame({
        "symbol": ["000001", "000002"], "rank_5d": [1, 2], "rank_1d": [1, 2], "rank_20d": [1, 2],
        "p_up_1d": [0.6, 0.4], "p_up_5d": [0.7, 0.3], "p_up_20d": [0.65, 0.35],
        "confidence_level": ["HIGH", "LOW"],
    })
    monkeypatch.setattr(api, "_prediction_snapshot", lambda: snapshot)
    result = api.latest_predictions(limit=10, min_probability=0.6, horizon=5, confidence="HIGH")
    assert len(result) == 1 and result[0]["symbol"] == "000001"
