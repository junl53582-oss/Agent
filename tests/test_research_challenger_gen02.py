from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stockpilot.research_challenger.config import ChallengerSettings
from stockpilot.research_challenger.gen02 import (
    Gen02Settings,
    _choose_configuration,
    _stability_metrics,
    _write_json,
    audit_holdout,
    development_protocol,
    evaluate_holdout_2026,
    load_development_dataset,
    open_holdout_once,
    record_test_receipt,
)
from stockpilot.research_challenger.gen02_portfolio import (
    PortfolioPolicy,
    evaluate_portfolio_policy,
    staggered_sleeve_membership,
)


def _portfolio_frame(dates: int = 3) -> pd.DataFrame:
    rows = []
    for date_index, date in enumerate(pd.bdate_range("2025-01-02", periods=dates)):
        for symbol in range(40):
            rows.append(
                {
                    "date": date,
                    "symbol": f"{symbol:06d}",
                    "score": float(symbol + (date_index if symbol % 2 else -date_index)),
                    "future_return_5d": symbol / 10_000,
                    "entry_tradable": True,
                    "execution_return": symbol / 10_000,
                    "broad_sector": "A" if symbol < 20 else "B",
                    "benchmark_weight": 1 / 40,
                    "benchmark_weight_rank": symbol / 39,
                    "amount_rank": symbol / 39,
                }
            )
    return pd.DataFrame(rows)


def test_holdout_audit_fails_closed_when_2026_evidence_exists(tmp_path: Path) -> None:
    (tmp_path / "artifacts/comparison/ridge").mkdir(parents=True)
    (tmp_path / "artifacts/prediction_v30/live/predictions").mkdir(parents=True)
    (tmp_path / "artifacts/comparison/ridge/summary.json").write_text(
        json.dumps({"data_end": "2026-08-21"}), encoding="utf-8"
    )
    (tmp_path / "artifacts/comparison/ridge/latest_signals.csv").write_text(
        "date,symbol\n2026-08-21,000001\n", encoding="utf-8"
    )
    (tmp_path / "artifacts/prediction_v30/live/predictions/2026-08-21.csv").write_text(
        "date,symbol\n2026-08-21,000001\n", encoding="utf-8"
    )
    (tmp_path / "RESEARCH_DECISIONS.md").write_text("2026-08-21 inspected", encoding="utf-8")
    result = audit_holdout(tmp_path)
    assert result["untouched_2026_holdout"] is False
    assert result["holdout_open_permitted"] is False


def test_development_loader_pushes_2026_filter_into_parquet_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    dates = pd.to_datetime(["2025-12-30", "2025-12-31"])
    frame = pd.DataFrame(
        {
            "date": dates,
            "symbol": ["1", "2"],
            "membership_snapshot_date": dates,
            "available_date": dates,
            "industry_effective_date": dates,
            "industry": ["A", "A"],
            "eligible": [True, True],
            "in_universe": [True, True],
            "future_return_1d": [0.01, -0.01],
            "future_return_5d": [0.02, -0.02],
            "future_return_20d": [0.03, -0.03],
            "feature": [1.0, 2.0],
        }
    )

    def fake_read(path: Path, **kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        return frame.copy()

    monkeypatch.setattr(
        "stockpilot.research_challenger.gen02.verify_dataset_manifest",
        lambda settings: {"manifest_rows": 2, "source_hashes": {}},
    )
    monkeypatch.setattr(pd, "read_parquet", fake_read)
    base = replace(
        ChallengerSettings(),
        factor_columns=("feature",),
        horizons=(1, 5, 20),
    )
    result, evidence = load_development_dataset(base)
    assert captured["filters"] == [("date", "<", pd.Timestamp("2026-01-01"))]
    assert result["date"].max() < pd.Timestamp("2026-01-01")
    assert evidence["prospective_rows_used"] == 0


def test_development_loader_rejects_2026_row_even_if_filter_backend_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    date = pd.Timestamp("2026-01-02")
    frame = pd.DataFrame(
        {
            "date": [date],
            "symbol": ["1"],
            "membership_snapshot_date": [date],
            "available_date": [date],
            "industry_effective_date": [date],
            "industry": ["A"],
            "eligible": [True],
            "in_universe": [True],
            "future_return_1d": [0.01],
            "future_return_5d": [0.02],
            "future_return_20d": [0.03],
            "feature": [1.0],
        }
    )
    monkeypatch.setattr(
        "stockpilot.research_challenger.gen02.verify_dataset_manifest",
        lambda settings: {"manifest_rows": 1, "source_hashes": {}},
    )
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: frame.copy())
    base = replace(ChallengerSettings(), factor_columns=("feature",))
    with pytest.raises(RuntimeError, match="2026_DATA_ENTERED"):
        load_development_dataset(base)


def test_development_protocol_separates_development_and_disqualified_holdout() -> None:
    protocol = development_protocol()
    assert protocol["development_period"] == [2020, 2021, 2022, 2023, 2024, 2025]
    assert protocol["holdout_2026"] == "DISQUALIFIED_NOT_UNTOUCHED"
    assert protocol["2026_label_or_performance_read_allowed"] is False
    assert protocol["models"]["lambdarank"] == "FROZEN_REJECTED_NOT_RETUNED"


def test_holdout_open_command_is_required_and_consumed_state_is_immutable(
    tmp_path: Path,
) -> None:
    _write_json(tmp_path / "development_protocol.json", {"frozen": True})
    _write_json(tmp_path / "precommit_decision.json", {"challenger": "ridge"})
    state = open_holdout_once(tmp_path, True)
    assert state["holdout_consumed"] is True
    with pytest.raises(RuntimeError, match="IMMUTABLE"):
        open_holdout_once(tmp_path, True)


def test_false_holdout_blocks_before_any_evaluation(tmp_path: Path) -> None:
    _write_json(tmp_path / "holdout_audit.json", {"untouched_2026_holdout": False})
    with pytest.raises(RuntimeError, match="UNTOUCHED_2026_HOLDOUT_FALSE"):
        evaluate_holdout_2026(Gen02Settings(artifact_dir=tmp_path))
    assert not (tmp_path / "holdout_state.json").exists()


def test_buffer_rule_retains_rank_inside_exit_zone() -> None:
    frame = _portfolio_frame(2)
    first_date, second_date = sorted(frame["date"].unique())
    # Put the prior top symbol at rank 25 on the second date. A Top20/exit30
    # buffer must retain it, reducing turnover relative to raw Top20.
    prior_top = "000039"
    second = frame["date"].eq(second_date)
    frame.loc[second & frame["symbol"].eq(prior_top), "score"] = 14.5
    raw, _ = evaluate_portfolio_policy(
        frame,
        "score",
        5,
        PortfolioPolicy("raw", 20),
        rebalance_every=1,
        buy_rate=0.001,
        sell_rate=0.002,
    )
    buffered, _ = evaluate_portfolio_policy(
        frame,
        "score",
        5,
        PortfolioPolicy("buffer", 20, buffer_exit_rank=30),
        rebalance_every=1,
        buy_rate=0.001,
        sell_rate=0.002,
    )
    assert buffered.iloc[1]["buy_turnover"] <= raw.iloc[1]["buy_turnover"]


def test_transaction_cost_known_example_includes_entry_and_final_exit() -> None:
    frame = _portfolio_frame(1)
    periods, _ = evaluate_portfolio_policy(
        frame,
        "score",
        5,
        PortfolioPolicy("equal", 20),
        rebalance_every=5,
        buy_rate=0.001,
        sell_rate=0.002,
    )
    assert periods.iloc[0]["transaction_cost"] == pytest.approx(0.003)
    assert periods.iloc[0]["net_return"] == pytest.approx(
        periods.iloc[0]["gross_return"] - 0.003
    )


def test_sector_balanced_policy_uses_existing_v6_quota_logic() -> None:
    frame = _portfolio_frame(1)
    # Raw scores place the entire top 20 in sector B. V6 quota logic must spread it.
    periods, _ = evaluate_portfolio_policy(
        frame,
        "score",
        5,
        PortfolioPolicy("sector", 20, sector_balanced=True),
        rebalance_every=5,
        buy_rate=0.0,
        sell_rate=0.0,
    )
    assert periods.iloc[0]["maximum_sector_weight"] <= 0.5 + 1e-12


def test_staggered_sleeves_update_only_one_cohort_each_day() -> None:
    snapshots = staggered_sleeve_membership([["A"], ["B"], ["C"], ["D"]], horizon=3)
    assert snapshots[0] == {"A": pytest.approx(1 / 3)}
    assert snapshots[2] == {
        "A": pytest.approx(1 / 3),
        "B": pytest.approx(1 / 3),
        "C": pytest.approx(1 / 3),
    }
    assert "A" not in snapshots[3]
    assert snapshots[3]["D"] == pytest.approx(1 / 3)


def test_gen02_adds_no_versioned_code_directory_and_does_not_touch_formal_chain() -> None:
    forbidden = [
        Path("research_v32"),
        Path("research_v33"),
        Path("stockpilot/research_challenger_r2"),
    ]
    assert not any(path.exists() for path in forbidden)
    source = Path("stockpilot/research_challenger/gen02.py").read_text(encoding="utf-8")
    assert "artifacts/prospective_alpha_v1r4" in source  # read-only parent hash binding
    assert "production_prediction_ready\": True" not in source
    assert "execution_authorized\": True" not in source


def test_no_new_random_split_or_model_zoo() -> None:
    source = Path("stockpilot/research_challenger/gen02.py").read_text(encoding="utf-8")
    assert "train_test_split" not in source
    assert "XGBoost" not in source
    assert "CatBoost" not in source
    assert "Transformer" not in source


def test_stability_metrics_skip_insufficient_industry_slice() -> None:
    rows = []
    date = pd.Timestamp("2025-01-02")
    for symbol in range(40):
        rows.append(
            {
                "date": date,
                "symbol": f"{symbol:06d}",
                "horizon": 5,
                "regime": "risk_on",
                "broad_sector": "large" if symbol < 35 else "tiny",
                "future_return_5d": symbol / 1000,
                "score_v6": symbol,
                "score_ridge": symbol,
                "score_lightgbm_regression": symbol,
            }
        )
    regimes, industries = _stability_metrics(pd.DataFrame(rows))
    assert not regimes.empty
    assert set(industries["broad_sector"]) == {"large"}


def test_candidate_gate_numpy_bools_are_serializable() -> None:
    model_metrics = pd.DataFrame(
        [
            {
                "model": model,
                "horizon": 5,
                "mean_rank_ic": np.float64(0.03),
                "rank_ic_ir": np.float64(0.1),
                "positive_rank_ic_ratio": np.float64(0.55),
            }
            for model in ("ridge", "lightgbm_regression")
        ]
    )
    yearly = pd.DataFrame(
        [
            {"model": model, "horizon": 5, "test_year": 2025, "mean_rank_ic": 0.01}
            for model in ("ridge", "lightgbm_regression")
        ]
    )
    portfolios = pd.DataFrame(
        [
            {
                "model": model,
                "horizon": 5,
                "portfolio_policy": "equal_top20",
                "net_research_proxy_alpha": 0.01,
                "max_drawdown": -0.2,
                "annualized_turnover": 10.0,
                "average_maximum_sector_weight": 0.4,
            }
            for model in ("ridge", "lightgbm_regression")
        ]
    )
    best, table = _choose_configuration(model_metrics, yearly, portfolios, Gen02Settings())
    assert json.loads(table.iloc[0]["gates"])["rank_ic_positive"] is True
    assert isinstance(best["all_gates_passed"], bool)


def test_test_receipt_records_the_test_command_not_the_cli_subcommand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "stockpilot.research_challenger.gen02.Gen02Settings",
        lambda: Gen02Settings(artifact_dir=tmp_path),
    )
    receipt = record_test_receipt("targeted", "python -m pytest -q tests/x.py", "1 passed")
    assert receipt["command"].endswith("tests/x.py")
    assert json.loads((tmp_path / "targeted_test_receipt.json").read_text(encoding="utf-8"))[
        "command"
    ].endswith("tests/x.py")


def test_final_manifest_implementation_is_recursive_and_binds_effective_lock() -> None:
    source = Path("stockpilot/research_challenger/gen02.py").read_text(encoding="utf-8")
    assert 'artifact_dir.rglob("*")' in source
    assert '"development_lock_sha256": effective_development["lock_sha256"]' in source
