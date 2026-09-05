from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stockpilot.daily_pit.runtime import DailyRuntimeSettings
from stockpilot.daily_prediction.freshness import evaluate_daily_prediction_freshness
from stockpilot.daily_prediction.product import DailyPredictionSettings
from stockpilot.prospective_r2.integrity import (
    write_immutable_bytes,
    write_immutable_frame,
    write_immutable_json,
)


def _calendar(path: Path) -> Path:
    target = path / "calendar.json"
    target.write_text(
        json.dumps(
            {
                "market": "XSHG",
                "coverage_start": "2026-08-01",
                "coverage_end": "2026-09-30",
                "weekends_closed": True,
                "closed_weekdays": [],
                "source": "test verified calendar",
                "source_url": "https://example.invalid/calendar",
            }
        ),
        encoding="utf-8",
    )
    return target


def _settings(tmp_path: Path) -> DailyPredictionSettings:
    runtime = DailyRuntimeSettings(calendar_path=_calendar(tmp_path))
    return DailyPredictionSettings(
        root=tmp_path / "formal",
        runtime_settings=runtime,
        verify_git_boundary=False,
        require_product_protocol=False,
    )


def _publish_read_only_fixture(settings: DailyPredictionSettings, target_date: str) -> None:
    directory = settings.prediction_root / target_date
    prediction_id = f"DAILY-GEN2-{target_date}-fixture"
    ranking = pd.DataFrame(
        {
            "rank": range(1, 21),
            "symbol": [f"{value:06d}" for value in range(1, 21)],
        }
    )
    prediction = {
        "status": "PREDICTION_AVAILABLE",
        "prediction_id": prediction_id,
        "prediction_date": target_date,
        "model_id": settings.model_id,
        "predictions": [],
        "top10": ranking.head(10)["symbol"].tolist(),
        "top20": ranking["symbol"].tolist(),
        "universe_count": 20,
        "eligible_count": 20,
    }
    prediction_hash = write_immutable_json(directory / "prediction.json", prediction)
    ranking_hash = write_immutable_frame(directory / "ranking.csv", ranking, ["rank", "symbol"])
    top10_hash = write_immutable_frame(
        directory / "top10.csv", ranking.head(10), ["rank", "symbol"]
    )
    top20_hash = write_immutable_frame(directory / "top20.csv", ranking, ["rank", "symbol"])
    report_hash = write_immutable_bytes(
        directory / f"DAILY_STOCK_PREDICTION_REPORT_{target_date}.md", b"fixture"
    )
    manifest_hash = write_immutable_json(
        directory / "prediction_manifest.json",
        {
            "prediction.json": prediction_hash,
            "ranking.csv": ranking_hash,
            "top10.csv": top10_hash,
            "top20.csv": top20_hash,
            "prediction_report.md": report_hash,
        },
    )
    settings.latest_path.parent.mkdir(parents=True, exist_ok=True)
    settings.latest_path.write_text(
        json.dumps(
            {
                "prediction_date": target_date,
                "prediction_id": prediction_id,
                "artifact_path": str(directory),
                "prediction_manifest_hash": manifest_hash,
                "prediction_json_hash": prediction_hash,
            }
        ),
        encoding="utf-8",
    )


def test_weekend_without_product_is_explicit_and_points_to_next_session(tmp_path: Path) -> None:
    value = evaluate_daily_prediction_freshness(
        now=datetime(2026, 9, 5, 4, tzinfo=timezone.utc), settings=_settings(tmp_path)
    )
    assert value["freshness_status"] == "NO_FORMAL_PREDICTION"
    assert value["expected_latest_session"] == "2026-09-04"
    assert value["next_verified_prediction_date"] == "2026-09-07"
    assert value["earliest_legal_time"] == "2026-09-07T18:30:00+08:00"
    assert value["scheduled_time"] == "2026-09-07T18:45:00+08:00"


def test_valid_latest_completed_session_is_current(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _publish_read_only_fixture(settings, "2026-09-04")
    value = evaluate_daily_prediction_freshness(
        now=datetime(2026, 9, 4, 11, tzinfo=timezone.utc), settings=settings
    )
    assert value["freshness_status"] == "CURRENT"
    assert value["integrity_status"] == "VALID"
    assert value["lag_sessions"] == 0


def test_older_verified_product_is_stale_with_session_lag(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _publish_read_only_fixture(settings, "2026-09-02")
    value = evaluate_daily_prediction_freshness(
        now=datetime(2026, 9, 4, 11, tzinfo=timezone.utc), settings=settings
    )
    assert value["freshness_status"] == "STALE"
    assert value["lag_sessions"] == 2
    assert value["prediction_status"] == "PREDICTION_AVAILABLE"


def test_before_data_window_does_not_expect_same_day_prediction(tmp_path: Path) -> None:
    value = evaluate_daily_prediction_freshness(
        now=datetime(2026, 9, 7, 9, tzinfo=timezone.utc), settings=_settings(tmp_path)
    )
    assert value["expected_latest_session"] == "2026-09-04"
    assert value["next_verified_prediction_date"] == "2026-09-07"


def test_pointer_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _publish_read_only_fixture(settings, "2026-09-04")
    pointer = json.loads(settings.latest_path.read_text(encoding="utf-8"))
    pointer["prediction_json_hash"] = "0" * 64
    settings.latest_path.write_text(json.dumps(pointer), encoding="utf-8")
    value = evaluate_daily_prediction_freshness(
        now=datetime(2026, 9, 4, 11, tzinfo=timezone.utc), settings=settings
    )
    assert value["freshness_status"] == "INVALID"
    assert "LATEST_POINTER_PREDICTION_HASH_MISMATCH" in value["reason"]


def test_missing_product_has_no_operational_side_effects(tmp_path: Path) -> None:
    value = evaluate_daily_prediction_freshness(
        now=datetime(2026, 9, 5, 4, tzinfo=timezone.utc), settings=_settings(tmp_path)
    )
    assert value["read_only"] is True
    assert value["provider_requests"] == 0
    assert value["model_runs"] == 0
    assert value["broker_requests"] == 0
    assert value["execution_authorized"] is False
    assert {"2026-09-03", "2026-09-04"} <= set(value["known_no_backfill_dates"])


def test_dashboard_uses_formal_product_and_labels_research_snapshots() -> None:
    source = Path("dashboard.py").read_text(encoding="utf-8")
    assert "artifacts/daily_predictions/gen2/latest.json" in source
    assert '@st.fragment(run_every="60s")' in source
    assert "V6历史研究快照" in source
    assert "V30历史研究快照" in source
    assert "predict_daily(" not in source
    assert "acquire_lineage_aligned_market" not in source
    assert "run_demo" not in source
