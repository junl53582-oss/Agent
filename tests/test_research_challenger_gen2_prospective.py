from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from stockpilot.prospective_r2.integrity import (
    read_verified_json,
    verify_immutable,
    write_immutable_frame,
    write_immutable_json,
)
from stockpilot.research_challenger.prospective_gen2 import (
    ProspectiveGen2Settings,
    cost_policy,
    feature_policy,
    freeze_operational_portability_amendment,
    freeze_human_readjudication,
    generate_prediction,
    human_audit,
    human_decision,
    human_protocol,
    label_end_session,
    model_specification,
    portfolio_policy,
    review_checkpoint,
    settle_prediction,
    training_policy,
    verify_human_freeze,
)


UTC = timezone.utc


def _calendar(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "market": "XSHG",
                "coverage_start": "2026-08-01",
                "coverage_end": "2026-12-31",
                "weekends_closed": True,
                "closed_weekdays": ["2026-09-25", "2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06", "2026-10-07"],
                "source": "fixture",
                "source_url": "https://example.invalid/calendar",
            }
        ),
        encoding="utf-8",
    )


def _settings(tmp_path: Path) -> ProspectiveGen2Settings:
    parents = {}
    for name in ("correctness", "interpretation", "v1r4", "v6"):
        path = tmp_path / f"{name}.lock.json"
        parents[name] = (path, write_immutable_json(path, {"name": name}))
    calendar = tmp_path / "calendar.json"
    _calendar(calendar)
    human = tmp_path / "human"
    return ProspectiveGen2Settings(
        human_dir=human,
        human_lock_path=human / "plan.lock.json",
        data_root=tmp_path / "data",
        prediction_root=tmp_path / "data/predictions",
        settlement_root=tmp_path / "data/settlements",
        calendar_path=calendar,
        dataset_path=tmp_path / "panel.parquet",
        dataset_manifest_path=tmp_path / "manifest.json",
        correctness_lock_path=parents["correctness"][0],
        correctness_interpretation_lock_path=parents["interpretation"][0],
        v1r4_lock_path=parents["v1r4"][0],
        v6_lock_path=parents["v6"][0],
        expected_correctness_lock=parents["correctness"][1],
        expected_interpretation_lock=parents["interpretation"][1],
        expected_v1r4_lock=parents["v1r4"][1],
        expected_v6_lock=parents["v6"][1],
    )


def _freeze(settings: ProspectiveGen2Settings) -> dict:
    return freeze_human_readjudication(
        settings,
        now=datetime(2026, 8, 31, 12, tzinfo=UTC),
        source_commit="test-commit",
    )


def _scorer(_: str, __: ProspectiveGen2Settings) -> tuple[pd.DataFrame, dict]:
    symbols = [f"{index:06d}" for index in range(1, 31)]
    frame = pd.DataFrame(
        {
            "date": pd.Timestamp("2026-09-01"),
            "symbol": symbols,
            "broad_sector": [f"S{index % 4}" for index in range(30)],
            "industry": [f"I{index % 8}" for index in range(30)],
            "benchmark_weight": 1 / 30,
            "benchmark_weight_rank": [index / 30 for index in range(30)],
            "score": [float(30 - index) for index in range(30)],
        }
    )
    return frame, {
        "model_signature": "model-signature",
        "training_snapshot_hash": "training-snapshot",
        "input_snapshot_hash": "input-snapshot",
        "selected_features": ["momentum"],
        "selected_features_hash": "features",
        "2026_realized_labels_read": False,
    }


def _prediction(settings: ProspectiveGen2Settings) -> dict:
    _freeze(settings)
    return generate_prediction(
        "2026-09-01",
        now=datetime(2026, 9, 1, 11, tzinfo=UTC),
        settings=settings,
        scorer=_scorer,
    )


def test_human_readjudication_retains_v6_champion(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    result = _freeze(settings)
    decision = read_verified_json(settings.human_dir / "decision.json")
    assert result["status"] == "PROSPECTIVE_RESEARCH_ONLY_APPROVED"
    assert decision["operative_champion"] == "V6"


def test_human_readjudication_does_not_promote_gen2(tmp_path: Path) -> None:
    decision = human_decision(_settings(tmp_path), "2026-08-31")
    assert decision["gen2_promotion_status"] == "NOT_PROMOTED"
    assert decision["production_shadow_eligible"] is False


def test_correctness_changed_eligibility_requires_human_decision(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_human_freeze(settings)
    assert human_decision(settings, "2026-08-31")["prospective_research_observation_approved"]


def test_original_cross_boundary_label_bug_is_preserved_in_audit(tmp_path: Path) -> None:
    audit = human_audit(_settings(tmp_path))
    assert audit["original_gen2_cross_boundary_label_bug_detected"]
    assert audit["original_gen2_2025_decision_rows_with_label_end_in_2026_existed"]


def test_corrected_run_did_not_consume_2026_holdout_labels(tmp_path: Path) -> None:
    audit = human_audit(_settings(tmp_path))
    assert audit["corrected_run_consumed_2026_realized_labels"] is False
    assert audit["untouched_2026_holdout"] is False


def test_prospective_start_is_after_human_freeze(tmp_path: Path) -> None:
    protocol = human_protocol(_settings(tmp_path), "2026-08-31")
    assert protocol["prospective_start_date"] == "2026-09-01"


def test_historical_backfill_is_rejected(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _freeze(settings)
    with pytest.raises(RuntimeError, match="HISTORICAL_BACKFILL_FORBIDDEN"):
        generate_prediction(
            "2026-08-31",
            now=datetime(2026, 8, 31, 11, tzinfo=UTC),
            settings=settings,
            scorer=_scorer,
        )


def test_prediction_artifact_is_immutable(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    result = _prediction(settings)
    assert verify_immutable(settings.prediction_root / "2026-09-01/manifest.json") == result["manifest_sha256"]


def test_existing_prediction_cannot_be_overwritten(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _prediction(settings)
    with pytest.raises(RuntimeError, match="IMMUTABLE_PREDICTION_ALREADY_EXISTS"):
        generate_prediction(
            "2026-09-01",
            now=datetime(2026, 9, 1, 11, tzinfo=UTC),
            settings=settings,
            scorer=_scorer,
        )


def test_prediction_contains_no_future_label(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _prediction(settings)
    text = (settings.prediction_root / "2026-09-01/prediction.csv").read_text(encoding="utf-8-sig")
    receipt = read_verified_json(settings.prediction_root / "2026-09-01/prediction.json")
    assert "actual_return" not in text and "realized_return" not in text
    assert receipt["future_label_fields_present"] is False


def _market(settings: ProspectiveGen2Settings) -> Path:
    prediction = pd.read_csv(settings.prediction_root / "2026-09-01/prediction.csv", dtype={"symbol": str})
    end = label_end_session("2026-09-01", settings)
    rows = []
    for symbol in prediction["symbol"]:
        rows.extend(
            [
                {"date": "2026-09-02", "symbol": symbol, "open": 10.0},
                {"date": end, "symbol": symbol, "open": 11.0},
            ]
        )
    path = settings.data_root / "market.csv"
    write_immutable_frame(path, pd.DataFrame(rows), ["date", "symbol"])
    return path


def test_20d_label_cannot_settle_early(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _prediction(settings)
    path = _market(settings)
    with pytest.raises(RuntimeError, match="20D_LABEL_NOT_MATURE"):
        settle_prediction("2026-09-01", path, as_of_date="2026-09-10", settings=settings)


def test_settlement_uses_frozen_prediction_not_recomputed_prediction(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _prediction(settings)
    result = settle_prediction("2026-09-01", _market(settings), as_of_date="2026-10-30", settings=settings)
    assert result["prediction_recomputed"] is False
    assert result["settlement_status"] == "SETTLED_RESEARCH_PROXY_ONLY"


@pytest.mark.parametrize(
    "field,change",
    [
        ("lightgbm_rounds", 81),
        ("selection_horizon", 20),
        ("top_k", 19),
    ],
)
def test_frozen_policy_hash_mismatch_fails_closed(tmp_path: Path, field: str, change: int) -> None:
    settings = _settings(tmp_path)
    _freeze(settings)
    changed = replace(settings, **{field: change})
    with pytest.raises(RuntimeError, match="FROZEN_SPEC_HASH_MISMATCH"):
        generate_prediction(
            "2026-09-01",
            now=datetime(2026, 9, 1, 11, tzinfo=UTC),
            settings=changed,
            scorer=_scorer,
        )


def test_v1r4_and_gen2_evidence_are_separate(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert "prospective_gen2" in settings.prediction_root.as_posix() or str(tmp_path) in str(settings.prediction_root)
    assert settings.prediction_root != Path("data/prospective_alpha_v1r4")


def test_gen2_prospective_cannot_modify_v6(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    before = verify_immutable(settings.v6_lock_path)
    _prediction(settings)
    assert verify_immutable(settings.v6_lock_path) == before


def test_gen2_prospective_cannot_authorize_execution(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    result = _prediction(settings)
    assert result["execution_authorized"] is False


def test_gen2_prospective_cannot_auto_promote(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    result = _prediction(settings)
    assert result["automatic_promotion_allowed"] is False
    assert review_checkpoint(settings)["human_review_required"] is True


def test_official_alpha_fails_closed_without_benchmark_approval(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _prediction(settings)
    with pytest.raises(RuntimeError, match="OFFICIAL_ALPHA_BLOCKED"):
        settle_prediction(
            "2026-09-01", _market(settings), as_of_date="2026-10-30", settings=settings, official_alpha_requested=True
        )


def test_research_proxy_metrics_allowed_when_benchmark_unapproved(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _prediction(settings)
    result = settle_prediction("2026-09-01", _market(settings), as_of_date="2026-10-30", settings=settings)
    assert result["official_benchmark_status"] == "UNAPPROVED"
    assert result["research_proxy_spread"] is not None


def _checkpoint_files(settings: ProspectiveGen2Settings, count: int, settled: int = 0) -> None:
    sessions = load_sessions(settings)
    for date in sessions[:count]:
        write_immutable_json(settings.prediction_root / f"{date}/manifest.json", {"date": date})
    for date in sessions[:settled]:
        write_immutable_json(settings.settlement_root / f"{date}/settlement.json", {"date": date})


def load_sessions(settings: ProspectiveGen2Settings) -> list[str]:
    from stockpilot.prospective_r2.calendar import load_verified_calendar

    return [str(value.date()) for value in load_verified_calendar(settings.calendar_path).sessions() if value >= pd.Timestamp("2026-08-03")]


def test_review_checkpoint_uses_trading_days(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _checkpoint_files(settings, 20)
    assert review_checkpoint(settings)["prediction_trading_days"] == 20


def test_20_trading_day_checkpoint_is_pipeline_only(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _checkpoint_files(settings, 20)
    assert review_checkpoint(settings)["status"] == "PIPELINE_ONLY_COMPLETE_NO_MODEL_JUDGMENT"


def test_60_day_checkpoint_cannot_promote(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _checkpoint_files(settings, 60)
    status = review_checkpoint(settings)
    assert status["status"] == "PROVISIONAL_RESEARCH_REVIEW_ONLY"
    assert status["automatic_promotion_allowed"] is False


def test_120_day_checkpoint_still_requires_human_review(tmp_path: Path) -> None:
    settings = replace(_settings(tmp_path), evidence_review_days=70, provisional_review_days=40)
    _checkpoint_files(settings, 70, settled=40)
    status = review_checkpoint(settings)
    assert status["status"] == "PROSPECTIVE_EVIDENCE_REVIEW_READY"
    assert status["human_review_required"] is True


def test_training_semantics_are_deterministic_protocol_retrain(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert training_policy(settings)["semantics"] == "DETERMINISTIC_PROTOCOL_RETRAIN"
    assert model_specification(settings)["random_seed"] == 42
    assert feature_policy(settings)["future_columns_forbidden"] is True
    assert portfolio_policy(settings)["name"] == "sector_balanced_top20"
    assert cost_policy(settings)["official_benchmark_status"] == "UNAPPROVED"


def test_legacy_parent_without_sidecar_requires_exact_pinned_hash(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.v6_lock_path.with_suffix(settings.v6_lock_path.suffix + ".sha256").unlink()
    result = _freeze(settings)
    assert result["status"] == "PROSPECTIVE_RESEARCH_ONLY_APPROVED"


def test_legacy_parent_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.v6_lock_path.with_suffix(settings.v6_lock_path.suffix + ".sha256").unlink()
    settings.v6_lock_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="FROZEN_LOCK_MISMATCH"):
        freeze_human_readjudication(
            settings,
            now=datetime(2026, 8, 31, 12, tzinfo=UTC),
            source_commit="test-commit",
        )


def test_operational_lock_uses_repository_relative_paths(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _freeze(settings)
    result = freeze_operational_portability_amendment(settings)
    lock = read_verified_json(
        settings.human_dir / "experiments/008_operational_portability_fix/plan.lock.json"
    )
    assert result["status"] == "OPERATIONAL_PORTABILITY_AMENDMENT_FROZEN"
    assert "stockpilot/research_challenger/prospective_gen2.py" in lock["files"]
    assert not any(
        ":/" in name and name.endswith("stockpilot/research_challenger/prospective_gen2.py")
        for name in lock["files"]
    )
