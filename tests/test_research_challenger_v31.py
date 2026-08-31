from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stockpilot.prediction.labels import add_prediction_labels
from stockpilot.research_challenger.config import ChallengerSettings
from stockpilot.research_challenger.data import (
    add_research_targets,
    assert_feature_columns_safe,
    factor_inventory,
)
from stockpilot.research_challenger.factors import (
    bh_fdr,
    daily_rank_ic_matrix,
    residualize_cross_section,
    select_factors_train_only,
)
from stockpilot.research_challenger.metrics import (
    daily_rank_metrics,
    evaluate_topk,
    moving_block_bootstrap_delta,
)
from stockpilot.research_challenger.models import (
    TrainOnlyPreprocessor,
    deterministic_full_date_sample,
    fit_candidate_models,
)
from stockpilot.research_challenger.pipeline import _write_json_new
from stockpilot.research_challenger.split import build_fold
from stockpilot.research_challenger import freeze as freeze_module


def _calendar_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-02", "2022-12-30")
    rows = []
    for date in dates:
        for symbol in range(30):
            row = {
                "date": date,
                "symbol": f"{symbol:06d}",
                "label_end_date_1d": date + pd.offsets.BDay(2),
                "label_end_date_5d": date + pd.offsets.BDay(6),
                "label_end_date_20d": date + pd.offsets.BDay(21),
                "future_return_1d": symbol / 1000,
                "future_return_5d": symbol / 500,
                "future_return_20d": symbol / 200,
                "industry": "A" if symbol < 15 else "B",
                "broad_sector": "one" if symbol < 15 else "two",
                "benchmark_weight": 1 / 30,
                "benchmark_weight_rank": symbol / 30 - 0.5,
                "amount_rank": symbol / 30 - 0.5,
                "entry_tradable": True,
                "entry_tradable_20": True,
                "execution_return": symbol / 500,
                "execution_return_20": symbol / 200,
            }
            rows.append(row)
    return add_research_targets(pd.DataFrame(rows), (1, 5, 20))


def test_future_feature_injection_rejected() -> None:
    with pytest.raises(ValueError, match="cannot enter features"):
        assert_feature_columns_safe(("momentum", "future_return_5d"))
    with pytest.raises(ValueError, match="cannot enter features"):
        assert_feature_columns_safe(("quality", "label_5"))


def test_open_to_open_label_semantics_are_t_plus_one_to_h_plus_one() -> None:
    dates = pd.bdate_range("2026-01-05", periods=30)
    frame = pd.DataFrame(
        {
            "date": dates,
            "symbol": "000001",
            "open": np.arange(100.0, 130.0),
        }
    )
    result = add_prediction_labels(frame, (1, 5, 20), {1: 0, 5: 0, 20: 0})
    first = result.iloc[0]
    assert first["prediction_entry_open"] == 101
    assert first["future_return_1d"] == pytest.approx(102 / 101 - 1)
    assert first["future_return_5d"] == pytest.approx(106 / 101 - 1)
    assert first["future_return_20d"] == pytest.approx(121 / 101 - 1)


def test_purged_walk_forward_order_and_maturity() -> None:
    frame = _calendar_frame()
    fold = build_fold(
        frame,
        2022,
        20,
        training_window_years=8,
        validation_years=1,
        purge_gap_trading_days=21,
    )
    assert frame.loc[fold.train_index, "date"].max() < frame.loc[fold.validation_index, "date"].min()
    assert frame.loc[fold.validation_index, "date"].max() < frame.loc[fold.test_index, "date"].min()
    assert pd.to_datetime(frame.loc[fold.train_index, "label_end_date_20d"]).max() < fold.validation_start
    assert pd.to_datetime(frame.loc[fold.refit_index, "label_end_date_20d"]).max() < fold.test_start


@pytest.mark.parametrize("horizon,gap", [(1, 2), (5, 6), (20, 21)])
def test_each_horizon_honors_frozen_purge_gap(horizon: int, gap: int) -> None:
    fold = build_fold(
        _calendar_frame(),
        2022,
        horizon,
        training_window_years=8,
        validation_years=1,
        purge_gap_trading_days=gap,
    )
    assert fold.purge_gap_trading_days == gap


def test_immature_rows_cannot_enter_refit() -> None:
    frame = _calendar_frame()
    fold = build_fold(
        frame, 2022, 5, training_window_years=8, validation_years=1, purge_gap_trading_days=6
    )
    assert (pd.to_datetime(frame.loc[fold.refit_index, "label_end_date_5d"]) < pd.Timestamp("2022-01-01")).all()


def test_train_only_preprocessing_does_not_fit_oos() -> None:
    train = pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-01", "2020-01-02"]), "symbol": ["1", "2"], "a": [0.0, 2.0]}
    )
    test = pd.DataFrame(
        {"date": pd.to_datetime(["2021-01-01"]), "symbol": ["3"], "a": [1000.0]}
    )
    processor = TrainOnlyPreprocessor().fit(train, ("a",))
    assert processor.mean_["a"] < 2
    assert "2021-01-01:3" not in processor.fit_row_ids_
    assert np.isfinite(processor.transform(test, ("a",))).all()


def test_daily_rank_ic_known_example() -> None:
    ascending = list(range(30))
    descending = list(reversed(ascending))
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 30 + ["2020-01-02"] * 30),
            "score": ascending + descending,
            "ret": ascending + descending,
        }
    )
    result = daily_rank_metrics(frame, "score", "ret")
    assert result["rank_ic"].tolist() == pytest.approx([1.0, 1.0])


def test_daily_factor_matrix_known_signs() -> None:
    frame = pd.DataFrame(
        {"date": ["2020-01-01"] * 5, "good": [1, 2, 3, 4, 5], "bad": [5, 4, 3, 2, 1], "target": [1, 2, 3, 4, 5]}
    )
    result = daily_rank_ic_matrix(frame, ("good", "bad"), "target")
    assert result.iloc[0]["good"] == pytest.approx(1.0)
    assert result.iloc[0]["bad"] == pytest.approx(-1.0)


def test_bh_fdr_known_ordering() -> None:
    result = bh_fdr(pd.Series([0.001, 0.01, 0.5]))
    assert result.iloc[0] <= result.iloc[1] <= result.iloc[2]
    assert result.iloc[0] == pytest.approx(0.003)


def test_factor_selection_is_train_only_and_has_no_fallback() -> None:
    dates = pd.bdate_range("2020-01-01", periods=140)
    rows = []
    for date in dates:
        for symbol in range(30):
            rows.append(
                {
                    "date": date,
                    "symbol": str(symbol),
                    "good": symbol / 30,
                    "noise": ((symbol * 7 + date.day) % 29) / 29,
                    "return_rank_5d": symbol / 30,
                }
            )
    settings = replace(
        ChallengerSettings(),
        factor_columns=("good", "noise"),
        minimum_ic_dates=20,
        minimum_year_direction_consistency=0.0,
    )
    selection = select_factors_train_only(pd.DataFrame(rows), settings)
    assert "good" in selection.selected
    assert "return_rank_5d" not in selection.selected


def test_cross_sectional_neutralization_removes_industry_level() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 40),
            "industry": ["A"] * 20 + ["B"] * 20,
            "benchmark_weight_rank": np.linspace(-0.5, 0.5, 40),
        }
    )
    values = pd.Series([1.0] * 20 + [3.0] * 20)
    residual = residualize_cross_section(frame, values, "industry")
    assert abs(residual.groupby(frame["industry"]).mean()).max() < 1e-10


def test_full_date_sampling_preserves_queries() -> None:
    frame = _calendar_frame().iloc[:9000]
    sampled = deterministic_full_date_sample(frame, 3000)
    original_counts = frame.groupby("date").size()
    sampled_counts = sampled.groupby("date").size()
    assert (sampled_counts == original_counts.loc[sampled_counts.index]).all()


def test_candidate_models_are_deterministic() -> None:
    rng = np.random.default_rng(42)
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=30), 50)
    frame = pd.DataFrame(
        {
            "date": dates,
            "symbol": [f"{i % 50:06d}" for i in range(len(dates))],
            "a": rng.normal(size=len(dates)),
            "b": rng.normal(size=len(dates)),
        }
    )
    frame["target"] = frame.groupby("date")["a"].rank(pct=True)
    settings = replace(
        ChallengerSettings(),
        factor_columns=("a", "b"),
        training_row_cap=2_000,
        lightgbm_rounds=5,
    )
    first, _, _, _ = fit_candidate_models(frame, frame.iloc[:100], ("a", "b"), "target", settings)
    second, _, _, _ = fit_candidate_models(frame, frame.iloc[:100], ("a", "b"), "target", settings)
    for model in first:
        assert first[model] == pytest.approx(second[model])


def test_nonfinite_mature_target_is_removed_before_lambdarank() -> None:
    rng = np.random.default_rng(7)
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=30), 50)
    frame = pd.DataFrame(
        {
            "date": dates,
            "symbol": [f"{i % 50:06d}" for i in range(len(dates))],
            "a": rng.normal(size=len(dates)),
            "b": rng.normal(size=len(dates)),
        }
    )
    frame["target"] = frame.groupby("date")["a"].rank(pct=True)
    frame.loc[frame.index[::101], "target"] = np.nan
    settings = replace(
        ChallengerSettings(),
        factor_columns=("a", "b"),
        training_row_cap=2_000,
        lightgbm_rounds=5,
    )
    predictions, diagnostics, _, _ = fit_candidate_models(
        frame, frame.iloc[:100], ("a", "b"), "target", settings
    )
    assert all(np.isfinite(values).all() for values in predictions.values())
    assert max(row["training_rows"] for row in diagnostics) < len(frame)


def test_topk_cost_adjustment_known_example() -> None:
    frame = _calendar_frame()
    one_date = frame[frame["date"].eq(frame["date"].min())].copy()
    one_date["score"] = np.arange(len(one_date))
    result = evaluate_topk(
        one_date,
        "score",
        5,
        10,
        rebalance_every=5,
        buy_rate=0.001,
        sell_rate=0.002,
    )
    assert result.iloc[0]["net_return"] < result.iloc[0]["gross_return"]
    assert result.iloc[0]["transaction_cost"] == pytest.approx(0.003)


def test_bootstrap_is_deterministic_and_paired() -> None:
    champion = pd.Series(np.linspace(-0.02, 0.02, 100))
    challenger = champion + 0.01
    first = moving_block_bootstrap_delta(challenger, champion, replications=100, block_length=10, seed=42)
    second = moving_block_bootstrap_delta(challenger, champion, replications=100, block_length=10, seed=42)
    assert first == second
    assert first["ci_lower"] > 0


def test_research_targets_preserve_industry_neutral_semantics() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 4),
            "industry": ["A", "A", "B", "B"],
            "future_return_5d": [0.1, 0.2, -0.2, 0.0],
        }
    )
    result = add_research_targets(frame, (5,))
    assert result.groupby("industry")["industry_alpha_5d"].mean().abs().max() < 1e-12


def test_factor_inventory_is_machine_readable() -> None:
    settings = replace(ChallengerSettings(), factor_columns=("momentum_120_rank",))
    frame = pd.DataFrame({"momentum_120_rank": [0.1], "momentum_120": [0.2]})
    inventory = factor_inventory(frame, settings)
    assert inventory.loc[0, "factor_name"] == "momentum_120_rank"
    assert inventory.loc[0, "lookback"] == "120 trading days"


def test_v31_protocol_disables_benchmark_target_and_production() -> None:
    protocol = json.loads(Path("artifacts/research_v31/protocol.json").read_text(encoding="utf-8"))
    assert protocol["targets"]["benchmark_relative"].startswith("DISABLED")
    assert protocol["immutable_decisions"]["production_prediction_ready"] is False
    assert protocol["immutable_decisions"]["execution_authorized"] is False


def test_v31_has_no_prospective_write_or_training_entrypoint() -> None:
    source = Path("stockpilot/research_challenger/pipeline.py").read_text(encoding="utf-8")
    assert "data/prospective" not in source
    assert "production_prediction_ready\": True" not in source
    assert "execution_authorized\": True" not in source


def test_immutable_artifact_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "decision.json"
    _write_json_new(path, {"decision": "first"})
    with pytest.raises(RuntimeError, match="immutable"):
        _write_json_new(path, {"decision": "second"})


def test_random_split_api_is_not_present() -> None:
    source = Path("stockpilot/research_challenger/split.py").read_text(encoding="utf-8")
    assert "train_test_split" not in source
    assert "shuffle" not in source


def test_protocol_fixates_challenger_before_oos() -> None:
    protocol = json.loads(Path("artifacts/research_v31/protocol.json").read_text(encoding="utf-8"))
    assert protocol["created_before_oos_evaluation"] is True
    assert protocol["models"]["pre_registered_challenger"] == "lightgbm_lambdarank"
    assert protocol["walk_forward"]["final_oos_may_not_select_features_models_targets_or_weights"] is True


def test_lock_verifier_selects_latest_existing_amendment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = replace(ChallengerSettings(), artifact_dir=tmp_path / "root")
    settings.artifact_dir.mkdir(parents=True)
    old_root = tmp_path / "001"
    latest_root = tmp_path / "002"
    for root, amendment_id in ((old_root, "OLD"), (latest_root, "LATEST")):
        root.mkdir()
        payload = {
            "amendment_id": amendment_id,
            "files": {},
            "production_prediction_ready": False,
            "execution_authorized": False,
        }
        lock = root / "plan.lock.json"
        lock.write_text(json.dumps(payload), encoding="utf-8")
        lock.with_suffix(".json.sha256").write_text(
            freeze_module.sha256(lock) + "\n", encoding="ascii"
        )
    monkeypatch.setattr(freeze_module, "AMENDMENT_ROOTS", (old_root, latest_root))
    result = freeze_module.verify_plan_lock(settings)
    assert result["intact"] is True
    assert result["amendment_id"] == "LATEST"
