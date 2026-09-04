from __future__ import annotations

import json
from pathlib import Path

from stockpilot.prediction_v2_readiness.audit import (
    AuditSettings,
    evaluate_joint_gate,
    run_and_write,
    sha256_file,
)


def _protocol() -> dict:
    path = Path("artifacts/prediction_v2/new_information_readiness/protocol.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _ready_sources() -> dict:
    return {
        "announcement_titles": {
            "rows": 250000,
            "symbols": 700,
            "date": {"distinct_years": 8},
            "duplicate_identity_rows": 0,
            "identity_verified": True,
        },
        "announcement_bodies": {
            "documents": 50000,
            "symbols": 500,
            "distinct_years": 5,
            "documents_with_content_hash": 50000,
            "historical_pit_verified": 50000,
        },
        "analyst_report_metadata": {
            "rows": 50000,
            "symbols": 400,
            "numeric_expectation_fields": ["eps"],
        },
        "analyst_consensus_vintages": {
            "symbols": 400,
            "distinct_snapshots": 24,
            "snapshot_span_days": 730,
            "numeric_forward_eps": True,
            "target_or_dispersion": True,
            "immutable_raw_hashes_complete": True,
            "replayable_revision_history": True,
            "strictly_before_actuals_verified": True,
        },
        "fundamental_actuals": {
            "symbols": 500,
            "available_date": {"invalid": 0},
            "replayable_revision_history": True,
            "revision_pollution_risk": False,
            "actual_release_time_verified": True,
            "distinct_years": 5,
            "novel_information_beyond_current_61_factors": True,
        },
        "prior_research": {"announcement_titles_are_novel": True},
    }


def test_protocol_sidecar_binds_exact_bytes() -> None:
    root = Path("artifacts/prediction_v2/new_information_readiness")
    expected = (root / "protocol.json.sha256").read_text(encoding="ascii").strip()
    assert sha256_file(root / "protocol.json") == expected


def test_joint_gate_requires_new_event_semantics_and_vintages() -> None:
    sources = _ready_sources()
    sources["announcement_bodies"]["documents"] = 12
    sources["announcement_consensus"] = {}
    sources["analyst_consensus_vintages"]["distinct_snapshots"] = 1
    result = evaluate_joint_gate(sources, _protocol())
    assert result["all_joint_gates_passed"] is False
    assert result["joint_gate"]["novel_event_semantics"] is False
    assert result["joint_gate"]["historical_consensus_vintages"] is False
    assert result["joint_gate"]["constructible_earnings_surprise"] is False


def test_title_volume_alone_cannot_authorize_challenger() -> None:
    sources = _ready_sources()
    sources["prior_research"]["announcement_titles_are_novel"] = False
    sources["announcement_bodies"]["historical_pit_verified"] = 0
    result = evaluate_joint_gate(sources, _protocol())
    assert result["announcement_title_gate"]["novelty"] is False
    assert result["all_joint_gates_passed"] is False


def test_all_pre_registered_gates_can_authorize_without_training() -> None:
    result = evaluate_joint_gate(_ready_sources(), _protocol())
    assert result["all_joint_gates_passed"] is True


def test_missing_sources_fail_closed_and_write_hashed_evidence(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    protocol_source = Path("artifacts/prediction_v2/new_information_readiness/protocol.json")
    (artifact_dir / "protocol.json").write_bytes(protocol_source.read_bytes())
    result = run_and_write(
        AuditSettings(
            source_root=tmp_path / "missing-data",
            repo_root=tmp_path,
            artifact_dir=artifact_dir,
        )
    )
    assert result["final_decision"]["status"] == "PREDICTION_V2_NEW_INFORMATION_NOT_READY"
    assert result["final_decision"]["bounded_challenger_experiment"] == "NOT_STARTED_GATE_FAILED"
    assert result["scope"]["labels_read"] is False
    assert result["scope"]["models_trained"] is False
    for name in (
        "source_inventory.json",
        "readiness_audit.json",
        "PREDICTION_V2_NEW_INFORMATION_READINESS_AUDIT_REPORT.md",
        "artifact_manifest.json",
    ):
        assert (artifact_dir / name).exists()
        assert sha256_file(artifact_dir / name) == (
            artifact_dir / f"{name}.sha256"
        ).read_text(encoding="ascii").strip()
