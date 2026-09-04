from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stockpilot.daily_pit.runtime import DailyRuntimeSettings
from stockpilot.forward_evidence.monitor import (
    ForwardEvidenceSettings,
    _metrics,
    build_state,
    initialize,
    run_daily,
    verify_forward_evidence,
)
from stockpilot.prospective_r2.integrity import (
    read_verified_json,
    verify_immutable,
    write_immutable_frame,
    write_immutable_json,
)

DATE = "2026-09-03"


@dataclass(frozen=True)
class _IsolatedDailyRuntimeSettings(DailyRuntimeSettings):
    """Keep forward-monitor fixtures away from operational market data."""

    test_frozen_market_path: Path = Path("_test_frozen_market.csv")

    def pit_settings(self):
        return replace(
            super().pit_settings(),
            frozen_market_path=self.test_frozen_market_path,
        )


def _baseline(_: ForwardEvidenceSettings) -> dict:
    return {
        "baseline_sha": "baseline",
        "git_sha": "head",
        "model_id": "GEN2-LGBM-20D-SECTOR-BALANCED-TOP20",
        "model_spec_hash": "model-spec",
        "feature_policy_hash": "features",
        "training_policy_hash": "training",
        "portfolio_policy_hash": "portfolio",
        "cost_policy_hash": "cost",
        "effective_lock_identity": {
            "human_007": "007",
            "operational_008": "008",
            "runtime_009": "009",
            "runtime_010": "010",
            "daily_pit_011": "011",
        },
        "protected_changes": [],
        "lock_status": "VALID",
    }


def _settings(tmp_path: Path) -> ForwardEvidenceSettings:
    runtime = _IsolatedDailyRuntimeSettings(
        data_root=tmp_path / "core",
        prediction_root=tmp_path / "core/predictions",
        settlement_root=tmp_path / "core/settlements",
        input_seal_root=tmp_path / "core/input_seals",
        reservation_root=tmp_path / "core/attempts",
        portfolio_root=tmp_path / "core/portfolio",
        daily_input_root=tmp_path / "core/daily_inputs",
        test_frozen_market_path=tmp_path / "core/frozen_market.csv",
    )
    return ForwardEvidenceSettings(
        root=tmp_path / "forward",
        baseline_sha="baseline",
        runtime_settings=runtime,
        verify_git_boundary=False,
        bootstrap_replications=20,
    )


def _write_core_prediction(settings: ForwardEvidenceSettings, date: str = DATE) -> None:
    daily = settings.runtime_settings.daily_input_root / date
    daily.mkdir(parents=True)
    symbols = [f"{index:06d}" for index in range(1, 61)]
    panel = pd.DataFrame(
        {
            "symbol": symbols,
            "benchmark_weight_rank": [index / 60 for index in range(60)],
            "volatility_60_rank": [1 - index / 60 for index in range(60)],
            "momentum": [index % 10 / 10 for index in range(60)],
            "liquidity": [index % 7 / 7 for index in range(60)],
        }
    )
    panel.to_parquet(daily / "panel.parquet", index=False)
    write_immutable_json(daily / "manifest.json", {"target_date": date})
    write_immutable_json(
        daily / "source_receipt.json",
        {
            "target_date": date,
            "provider_sources": {"fixture": 60},
            "acquired_at_utc": "2026-09-03T11:00:00+00:00",
            "target_rows": 60,
            "request_end_date": date,
            "provider_request_count": 0,
        },
    )
    write_immutable_json(daily / "market_manifest.json", {"target_date": date})
    market_rows = []
    for offset, session in enumerate(pd.bdate_range("2026-05-01", date)):
        for index, symbol in enumerate(symbols):
            market_rows.append(
                {
                    "date": session,
                    "symbol": symbol,
                    "open": 10 + index / 100 + offset / 1000,
                    "close": 10 + index / 100 + offset / 1000,
                    "volume": 1000,
                }
            )
    pd.DataFrame(market_rows).to_csv(daily / "market.csv", index=False)
    frozen = settings.runtime_settings.pit_settings().frozen_market_path
    frozen.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(market_rows).to_csv(frozen, index=False)
    prediction_dir = settings.runtime_settings.prediction_root / date
    frame = pd.DataFrame(
        {
            "date": date,
            "symbol": symbols,
            "industry": "fixture",
            "broad_sector": ["technology" if index < 30 else "defensive" for index in range(60)],
            "benchmark_weight": 1 / 60,
            "score": [60 - index for index in range(60)],
            "rank": [index + 1 for index in range(60)],
            "selected_for_new_portfolio": [index < 20 for index in range(60)],
            "portfolio_weight": [0.05 if index < 20 else None for index in range(60)],
            "prediction_date": date,
            "portfolio_action": "REBALANCE",
            "research_only": True,
            "production_prediction_ready": False,
            "execution_authorized": False,
        }
    )
    csv_hash = write_immutable_frame(
        prediction_dir / "prediction.csv", frame, ["prediction_date", "symbol"]
    )
    seal_hash = write_immutable_json(
        settings.runtime_settings.input_seal_root / f"{date}.json", {"date": date}
    )
    receipt = {
        "prediction_date": date,
        "created_at_utc": "2026-09-03T11:00:00+00:00",
        "model_id": settings.model_id,
        "model_spec_hash": "model-spec",
        "input_seal_sha256": seal_hash,
        "training_evidence": {"model_signature": {"fixture": True}},
        "prediction_csv_sha256": csv_hash,
        "is_rebalance_day": True,
        "portfolio_action": "REBALANCE",
        "label_maturity_date": "2026-10-09",
        "future_label_fields_present": False,
    }
    json_hash = write_immutable_json(prediction_dir / "prediction.json", receipt)
    write_immutable_json(
        prediction_dir / "manifest.json",
        {"prediction.json": json_hash, "prediction.csv": csv_hash},
    )


def _write_core_settlement(settings: ForwardEvidenceSettings, date: str = DATE) -> None:
    prediction = pd.read_csv(
        settings.runtime_settings.prediction_root / date / "prediction.csv",
        dtype={"symbol": str},
    )
    settlement = prediction[
        [
            "symbol",
            "score",
            "rank",
            "selected_for_new_portfolio",
            "portfolio_weight",
            "benchmark_weight",
        ]
    ].copy()
    settlement["prediction_date"] = date
    settlement["actual_return_20d"] = settlement["score"] / 10_000
    settlement["settled"] = True
    core_dir = settings.runtime_settings.settlement_root / date
    csv_hash = write_immutable_frame(
        core_dir / "settlement.csv", settlement, ["prediction_date", "symbol"]
    )
    write_immutable_json(
        core_dir / "settlement.json",
        {
            "prediction_date": date,
            "maturity_date": "2026-10-09",
            "settlement_csv_sha256": csv_hash,
            "research_proxy_return": 0.002,
            "market_witness_sha256": "witness",
            "market_source_sha256": "market",
        },
    )


def test_initialization_is_frozen_and_resumable(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    first = initialize(settings, baseline_verifier=_baseline)
    second = initialize(settings, baseline_verifier=_baseline)
    assert first["protocol_sha256"] == second["protocol_sha256"]
    assert read_verified_json(settings.protocol_path)["historical_optimization_allowed"] is False
    assert build_state(settings)["matured_predictions"] == 0


def test_pending_prediction_is_registered_without_outcome_access(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    initialize(settings, baseline_verifier=_baseline)
    _write_core_prediction(settings)
    result = run_daily(
        DATE,
        now=datetime(2026, 9, 3, 12, tzinfo=timezone.utc),
        settings=settings,
        baseline_verifier=_baseline,
    )
    record = read_verified_json(settings.prediction_root / DATE / "prediction.json")
    assert result["new_prediction_generated"] is True
    assert record["maturity_status"] == "PENDING_MATURITY"
    assert "actual_return_20d" not in record["scores_and_ranks"][0]
    assert result["broker_requests"] == 0
    assert result["provider_requests"] == 0


def test_mature_settlement_drives_forward_only_metrics(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    initialize(settings, baseline_verifier=_baseline)
    _write_core_prediction(settings)
    run_daily(
        DATE,
        now=datetime(2026, 9, 3, 12, tzinfo=timezone.utc),
        settings=settings,
        baseline_verifier=_baseline,
    )
    _write_core_settlement(settings)
    result = run_daily(
        DATE,
        now=datetime(2026, 10, 10, 12, tzinfo=timezone.utc),
        settings=settings,
        baseline_verifier=_baseline,
    )
    state = json_load(settings.state_path)
    assert result["total_matured_sessions"] == 1
    assert state["cumulative_rank_ic"] == 1.0
    assert (
        state["forward_metrics"]["quantile_mean_returns"]["Q5"]
        > state["forward_metrics"]["quantile_mean_returns"]["Q1"]
    )
    assert state["top20_20bps_proxy"] is not None


def test_registry_manifest_chain_and_hashes_are_verified(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    initialize(settings, baseline_verifier=_baseline)
    _write_core_prediction(settings)
    run_daily(
        DATE,
        now=datetime(2026, 9, 3, 12, tzinfo=timezone.utc),
        settings=settings,
        baseline_verifier=_baseline,
    )
    verified = verify_forward_evidence(settings, baseline_verifier=_baseline)
    assert verified["prediction_records"] == 1
    assert verify_immutable(settings.prediction_root / DATE / "manifest.json")


def test_before_data_window_fails_closed_without_provider_call(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(
        "stockpilot.forward_evidence.monitor.daily_pipeline.acquire_market", forbidden
    )
    result = run_daily(
        DATE,
        confirm_real_provider_acquisition=True,
        now=datetime(2026, 9, 3, 8, tzinfo=timezone.utc),
        settings=settings,
        baseline_verifier=_baseline,
    )
    assert result["new_prediction_generated"] is False
    assert result["pit_status"] == "NO_FORWARD_PREDICTION:DATA_WINDOW_NOT_OPEN"
    assert called is False
    assert list(settings.attempt_root.glob("*/*.json"))


def test_metrics_do_not_mix_historical_baseline(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    value = _metrics(
        [
            {
                "spearman_rank_ic": 0.1,
                "pearson_ic": 0.2,
                "residual_rank_ic": 0.05,
                "quantile_returns": {"Q1": -0.01, "Q2": 0, "Q3": 0.01, "Q4": 0.02, "Q5": 0.03},
                "top_k": {
                    f"top{k}": {
                        "gross_return": 0.03,
                        "gross_proxy_alpha": 0.01,
                        "name_retention": 0.5,
                        "cost_sensitivity": {str(bps): 0.01 for bps in settings.cost_bps},
                    }
                    for k in settings.top_ks
                },
                "prediction_time_regime": {
                    "market_regime": "neutral",
                    "volatility_regime": "low_vol",
                },
            }
        ],
        settings,
    )
    assert value["rank_ic"] == 0.1
    assert value["settled_sessions"] == 1


def json_load(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))
