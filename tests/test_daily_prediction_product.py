from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from stockpilot.daily_pit.runtime import DailyRuntimeSettings
from stockpilot.daily_prediction.product import (
    DailyPredictionError,
    DailyPredictionSettings,
    explain,
    history,
    latest,
    main,
    predict_daily,
    publish_prediction,
)
from stockpilot.prospective_r2.integrity import (
    read_verified_json,
    verify_immutable,
    write_immutable_bytes,
    write_immutable_frame,
    write_immutable_json,
)

DATE = "2026-09-03"
READY = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
EARLY = datetime(2026, 9, 3, 8, tzinfo=timezone.utc)


def _settings(tmp_path: Path) -> DailyPredictionSettings:
    runtime = DailyRuntimeSettings(
        data_root=tmp_path / "core",
        prediction_root=tmp_path / "core/predictions",
        settlement_root=tmp_path / "core/settlements",
        input_seal_root=tmp_path / "core/input_seals",
        reservation_root=tmp_path / "core/attempts",
        portfolio_root=tmp_path / "core/portfolio",
        daily_input_root=tmp_path / "core/daily_inputs",
    )
    return DailyPredictionSettings(
        root=tmp_path / "product",
        runtime_settings=runtime,
        baseline_sha="baseline",
        verify_git_boundary=False,
        require_product_protocol=False,
    )


def _baseline(_: DailyPredictionSettings) -> dict:
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


def _regime(*args, **kwargs) -> dict:
    del args, kwargs
    return {
        "market_regime": "neutral",
        "volatility_regime": "low_vol",
        "positive_20d_breadth": 0.52,
        "classified_from_prediction_time_data_only": True,
    }


def _core(settings: DailyPredictionSettings, date: str = DATE) -> dict:
    daily = settings.runtime_settings.daily_input_root / date
    write_immutable_json(daily / "manifest.json", {"target_date": date, "rows": 60})
    write_immutable_json(
        daily / "market_manifest.json", {"target_date": date, "provider": "fixture"}
    )
    write_immutable_bytes(daily / "market.csv", b"date,symbol,open,close,volume\n")
    write_immutable_json(
        daily / "source_receipt.json",
        {
            "target_date": date,
            "provider_sources": {"fixture": 60},
            "acquired_at_utc": "2026-09-03T11:00:00+00:00",
        },
    )
    seal_hash = write_immutable_json(
        settings.runtime_settings.input_seal_root / f"{date}.json", {"target_date": date}
    )
    symbols = [f"{index:06d}" for index in range(1, 61)]
    ranking = pd.DataFrame(
        {
            "date": date,
            "symbol": symbols,
            "industry": [f"industry-{index % 4}" for index in range(60)],
            "broad_sector": ["technology" if index < 30 else "defensive" for index in range(60)],
            "benchmark_weight": 1 / 60,
            "score": [1 - index / 100 for index in range(60)],
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
    root = settings.runtime_settings.prediction_root / date
    ranking_hash = write_immutable_frame(
        root / "prediction.csv", ranking, ["prediction_date", "symbol"]
    )
    receipt = {
        "prediction_date": date,
        "created_at_utc": "2026-09-03T11:00:00+00:00",
        "model_id": settings.model_id,
        "model_spec_hash": "model-spec",
        "training_evidence": {"model_signature": "a" * 64},
        "input_seal_sha256": seal_hash,
        "portfolio_action": "REBALANCE",
        "future_label_fields_present": False,
    }
    receipt_hash = write_immutable_json(root / "prediction.json", receipt)
    manifest_hash = write_immutable_json(
        root / "manifest.json",
        {"prediction.json": receipt_hash, "prediction.csv": ranking_hash},
    )
    return {
        "receipt": receipt,
        "ranking_path": root / "prediction.csv",
        "manifest_hash": manifest_hash,
        "provider_requests": 0,
    }


def _predict(settings: DailyPredictionSettings) -> dict:
    _core(settings)
    return predict_daily(
        DATE,
        now=READY,
        settings=settings,
        baseline_verifier=_baseline,
        regime_classifier=_regime,
    )


def test_valid_date_produces_complete_ranking(tmp_path: Path) -> None:
    value = _predict(_settings(tmp_path))
    assert value["status"] == "PREDICTION_AVAILABLE"
    assert len(value["predictions"]) == 60
    assert [row["rank"] for row in value["predictions"]] == list(range(1, 61))


def test_top10_is_generated(tmp_path: Path) -> None:
    value = _predict(_settings(tmp_path))
    assert len(value["top10"]) == 10
    assert sum(row["selected_top10"] for row in value["predictions"]) == 10


def test_top20_is_generated_without_changing_portfolio_policy(tmp_path: Path) -> None:
    value = _predict(_settings(tmp_path))
    assert len(value["top20"]) == 20
    assert sum(row["selected_top20"] for row in value["predictions"]) == 20
    assert value["portfolio_action"] == "REBALANCE"


def test_ranking_is_deterministic(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    first = _predict(settings)
    second = predict_daily(
        DATE,
        now=READY,
        settings=settings,
        baseline_verifier=_baseline,
        regime_classifier=_regime,
    )
    assert first["predictions"] == second["predictions"]
    assert first["prediction_id"] == second["prediction_id"]


def test_prediction_artifacts_are_immutable(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _predict(settings)
    root = settings.prediction_root / DATE
    for name in (
        "prediction.json",
        "ranking.csv",
        "top10.csv",
        "top20.csv",
        "prediction_manifest.json",
        f"DAILY_STOCK_PREDICTION_REPORT_{DATE}.md",
    ):
        assert verify_immutable(root / name)


def test_duplicate_identical_run_is_idempotent(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _predict(settings)
    value = predict_daily(
        DATE,
        now=READY,
        settings=settings,
        baseline_verifier=_baseline,
        regime_classifier=_regime,
    )
    assert value["idempotent"] is True


def test_conflicting_core_identity_is_rejected(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    core = _core(settings)
    publish_prediction(DATE, core, _baseline(settings), READY, settings, regime_classifier=_regime)
    with pytest.raises(DailyPredictionError, match="PREDICTION_CONFLICT"):
        publish_prediction(
            DATE,
            core | {"manifest_hash": "different"},
            _baseline(settings),
            READY,
            settings,
            regime_classifier=_regime,
        )


def test_pre_window_prediction_is_rejected_without_provider_call(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True

    value = predict_daily(
        DATE,
        confirm_real_provider_acquisition=True,
        now=EARLY,
        settings=settings,
        baseline_verifier=_baseline,
        acquisition_runner=forbidden,
    )
    assert value["reason_code"] == "DATA_WINDOW_NOT_OPEN"
    assert called is False


def test_invalid_pit_materialization_is_rejected(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    def acquired(*args, **kwargs):
        del args, kwargs
        return {"provider_requests_made": 1}

    def invalid(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("PIT_JOIN_INVALID")

    value = predict_daily(
        DATE,
        confirm_real_provider_acquisition=True,
        now=READY,
        settings=settings,
        baseline_verifier=_baseline,
        acquisition_runner=acquired,
        materializer=invalid,
    )
    assert value["reason_code"] == "FEATURE_INVALID"


def test_invalid_seal_is_rejected(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    daily = settings.runtime_settings.daily_input_root / DATE
    write_immutable_json(daily / "market_manifest.json", {"date": DATE})
    write_immutable_json(daily / "manifest.json", {"date": DATE})

    def invalid(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("SEAL_HASH_MISMATCH")

    value = predict_daily(
        DATE,
        now=READY,
        settings=settings,
        baseline_verifier=_baseline,
        sealer=invalid,
    )
    assert value["reason_code"] == "INPUT_INVALID"
    assert "SEAL" in value["reason"]


def test_invalid_lock_is_rejected_before_any_pipeline_action(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True

    value = predict_daily(
        DATE,
        now=READY,
        settings=settings,
        baseline_verifier=lambda _: (_ for _ in ()).throw(RuntimeError("011")),
        acquisition_runner=forbidden,
    )
    assert value["reason_code"] == "LOCK_INVALID"
    assert called is False


def test_product_never_authorizes_execution(tmp_path: Path) -> None:
    value = _predict(_settings(tmp_path))
    assert value["execution_authorized"] is False
    assert value["real_orders"] is False
    assert value["real_trades"] is False


def test_product_never_calls_broker(tmp_path: Path) -> None:
    value = _predict(_settings(tmp_path))
    assert value["broker_requests"] == 0


def test_report_corresponds_exactly_to_json_top_ranking(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    value = _predict(settings)
    report = (
        settings.prediction_root / DATE / f"DAILY_STOCK_PREDICTION_REPORT_{DATE}.md"
    ).read_text(encoding="utf-8")
    top10 = pd.read_csv(settings.prediction_root / DATE / "top10.csv", dtype={"symbol": str})
    assert top10["symbol"].tolist() == value["top10"]
    for symbol in value["top10"]:
        assert symbol in report


def test_latest_reads_the_verified_prediction(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    expected = _predict(settings)
    value = latest(settings)
    assert value["prediction_id"] == expected["prediction_id"]
    assert value["top10"] == expected["top10"]


def test_history_reads_without_recomputing_or_mutating(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _predict(settings)
    before = verify_immutable(settings.prediction_root / DATE / "prediction.json")
    rows = history(settings, limit=10)
    after = verify_immutable(settings.prediction_root / DATE / "prediction.json")
    assert rows[0]["date"] == DATE
    assert rows[0]["maturity_status"] == "PENDING_MATURITY"
    assert before == after


def test_explain_does_not_invent_feature_contributions(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    value = _predict(settings)
    result = explain(value["top10"][0], DATE, settings)
    assert result["feature_contribution_status"] == "FEATURE_CONTRIBUTION_NOT_AVAILABLE"
    assert "causal" in result["interpretation"]


def test_no_prediction_attempt_has_auditable_report(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    value = predict_daily(
        DATE,
        now=EARLY,
        settings=settings,
        baseline_verifier=_baseline,
    )
    root = Path(value["artifact_path"])
    assert read_verified_json(root / "status.json")["reason_code"] == "DATA_WINDOW_NOT_OPEN"
    assert verify_immutable(root / f"DAILY_STOCK_PREDICTION_REPORT_{DATE}.md")
    assert json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def test_latest_and_explain_without_prediction_are_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = _settings(tmp_path)
    assert latest(settings)["status"] == "NO_PREDICTION"
    assert explain("000001", DATE, settings)["status"] == "NO_PREDICTION"
    monkeypatch.setattr(
        "stockpilot.daily_prediction.product.DailyPredictionSettings", lambda: settings
    )
    assert main(["latest"]) == 0
    assert "NO_FORMAL_DAILY_PREDICTION_EXISTS" in capsys.readouterr().out
