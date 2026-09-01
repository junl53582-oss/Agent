from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from stockpilot.intelligence.components import finish_score
from stockpilot.intelligence.confidence import CONFIDENCE_POLICY_HASH, confidence_score_v1
from stockpilot.intelligence.derived import build_intelligence_snapshot
from stockpilot.intelligence.evidence import ProductEvidence
from stockpilot.intelligence.lineage import ModelOutput, independent_model_agreement
from stockpilot.intelligence.policies import (
    CONFIDENCE_POLICY,
    LINEAGE_POLICY,
    RISK_POLICY,
    STOCK_SCORE_POLICY,
    policy_hash,
)
from stockpilot.intelligence.risk import RISK_POLICY_HASH, prediction_risk_score_v1
from stockpilot.intelligence.schema import CanonicalPrediction
from stockpilot.intelligence.scoring import STOCK_SCORE_POLICY_HASH, stock_score_v1
from stockpilot.intelligence.snapshot import build_daily_snapshot

POLICY_FILES = {
    "stock_score_v1.json": STOCK_SCORE_POLICY,
    "confidence_v1.json": CONFIDENCE_POLICY,
    "risk_v1.json": RISK_POLICY,
    "model_lineage_v1.json": LINEAGE_POLICY,
}
PROTECTED_LOCKS = (
    Path("artifacts/research_v6/plan.lock.json"),
    Path("artifacts/prospective_alpha_v1r4/plan.lock.json"),
    Path(
        "artifacts/research_challenger/gen02/experiments/007_human_readjudication/"
        "experiments/009_prospective_runtime_hardening/experiments/"
        "010r3_runtime_self_verification_activation/plan.lock.json"
    ),
)


def prediction(symbol: str = "000001", **overrides) -> CanonicalPrediction:
    values = {
        "symbol": symbol,
        "stock_name": f"Stock {symbol}",
        "industry": None,
        "prediction_date": "2026-08-28",
        "prediction_timestamp": "2026-08-28T10:00:00+00:00",
        "universe_id": "CSI300_VALIDATED_SCOPE",
        "eligibility_status": "ELIGIBLE",
        "data_coverage": None,
        "feature_coverage": None,
        "raw_rank_score": 0.8,
        "return_1d_pred": None,
        "return_5d_pred": 0.02,
        "return_20d_pred": 0.04,
        "up_prob_1d": 0.60,
        "up_prob_5d": 0.62,
        "up_prob_20d": 0.64,
        "rank_1d": 1,
        "rank_5d": 1,
        "rank_20d": 1,
        "market_rank": 1,
        "industry_rank": None,
        "stock_score": None,
        "ranking_score": None,
        "expected_return_score": None,
        "probability_score": None,
        "agreement_score": None,
        "industry_score": None,
        "regime_score": None,
        "confidence_score": 0.70,
        "risk_score": None,
        "positive_drivers": None,
        "negative_drivers": None,
        "risk_drivers": None,
        "model_version": "V30r1:test",
        "feature_version": None,
        "training_cutoff": "2026-08-21",
        "data_snapshot_hash": None,
        "model_manifest_hash": None,
        "production_prediction_ready": False,
        "execution_authorized": False,
        "source_kind": "V30R1_FORWARD_R2",
        "source_artifact_path": "fixture.csv",
        "source_artifact_hash": "a" * 64,
        "source_snapshot_hash": "a" * 64,
        "adapter_version": "test",
    }
    values.update(overrides)
    return CanonicalPrediction(**values)


def daily_snapshot():
    return build_daily_snapshot(
        (
            prediction("000001", market_rank=1, return_5d_pred=0.03, up_prob_5d=0.67),
            prediction("000002", market_rank=2, return_5d_pred=0.02, up_prob_5d=0.62),
            prediction("000003", market_rank=3, return_5d_pred=0.01, up_prob_5d=0.57),
        )
    )


def test_stock_score_is_deterministic() -> None:
    kwargs = {
        "ranking": 0.9,
        "expected_return": 0.8,
        "probability": 0.7,
        "agreement": None,
        "industry": None,
        "regime": None,
        "confidence_score": 70.0,
        "risk_score": 30.0,
    }
    assert stock_score_v1(**kwargs).to_dict() == stock_score_v1(**kwargs).to_dict()


def test_stock_score_decomposition_sums_to_score() -> None:
    result = stock_score_v1(
        ranking=0.9,
        expected_return=0.8,
        probability=0.7,
        agreement=None,
        industry=None,
        regime=None,
        confidence_score=70.0,
        risk_score=30.0,
    )
    assert result.status == "OK"
    assert sum(c.contribution or 0.0 for c in result.components) == pytest.approx(result.score)


def test_missing_components_are_null_not_neutral_defaults() -> None:
    result = stock_score_v1(
        ranking=0.9,
        expected_return=None,
        probability=None,
        agreement=None,
        industry=None,
        regime=None,
        confidence_score=None,
        risk_score=None,
    )
    missing = [component for component in result.components if not component.available]
    assert missing
    assert all(component.normalized_value is None for component in missing)
    assert all(component.contribution is None for component in missing)


def test_insufficient_stock_coverage_fails_closed() -> None:
    result = stock_score_v1(
        ranking=1.0,
        expected_return=None,
        probability=None,
        agreement=None,
        industry=None,
        regime=None,
        confidence_score=None,
        risk_score=None,
    )
    assert result.status == "INSUFFICIENT_COMPONENT_COVERAGE"
    assert result.score is None
    assert result.component_coverage == pytest.approx(0.30)


def test_high_probability_is_not_itself_confidence() -> None:
    high = prediction(up_prob_1d=0.90, up_prob_5d=0.91, up_prob_20d=0.92)
    moderate = prediction(up_prob_1d=0.60, up_prob_5d=0.61, up_prob_20d=0.62)
    evidence = ProductEvidence()
    assert confidence_score_v1(high, evidence).score == pytest.approx(
        confidence_score_v1(moderate, evidence).score
    )


def test_confidence_uses_only_available_evidence() -> None:
    result = confidence_score_v1(prediction(), ProductEvidence())
    names = {component.name for component in result.components if component.available}
    assert names == {"prediction_dispersion", "upstream_confidence_evidence"}
    assert result.component_coverage == pytest.approx(0.30)


def test_confidence_without_enough_evidence_is_null() -> None:
    item = prediction(confidence_score=None, up_prob_1d=None, up_prob_5d=None, up_prob_20d=None)
    result = confidence_score_v1(item, ProductEvidence())
    assert result.score is None
    assert result.status == "INSUFFICIENT_COMPONENT_COVERAGE"


def test_risk_direction_near_half_is_higher_uncertainty() -> None:
    uncertain = prediction(up_prob_1d=0.49, up_prob_5d=0.50, up_prob_20d=0.51)
    decisive = prediction(up_prob_1d=0.79, up_prob_5d=0.80, up_prob_20d=0.81)
    assert prediction_risk_score_v1(uncertain, ProductEvidence()).score > (
        prediction_risk_score_v1(decisive, ProductEvidence()).score
    )


def test_high_stock_score_can_coexist_with_high_risk() -> None:
    result = stock_score_v1(
        ranking=1.0,
        expected_return=1.0,
        probability=1.0,
        agreement=1.0,
        industry=1.0,
        regime=1.0,
        confidence_score=100.0,
        risk_score=90.0,
    )
    assert result.score > 90
    assert next(c for c in result.components if c.name == "risk_retention").raw_value == 90.0


def test_forward_and_parent_do_not_count_as_independent_models() -> None:
    evidence = independent_model_agreement(
        (ModelOutput("V30R1", 0.8), ModelOutput("V30R1_FORWARD_R2", 0.7))
    )
    assert evidence.independent_family_count == 1
    assert evidence.score is None
    assert evidence.selected_sources == ("V30R1_FORWARD_R2",)


def test_independent_families_can_form_agreement() -> None:
    evidence = independent_model_agreement(
        (ModelOutput("V30R1_FORWARD_R2", 0.7), ModelOutput("V6", 0.8))
    )
    assert evidence.independent_family_count == 2
    assert evidence.score == pytest.approx(0.9)


def test_product_rank_does_not_overwrite_model_rank() -> None:
    source = daily_snapshot()
    derived = build_intelligence_snapshot(source)
    assert {record.model_rank for record in derived.records} == {1, 2, 3}
    assert all(record.product_rank is not None for record in derived.records)
    assert [record.product_rank for record in derived.records] == [1, 2, 3]


def test_derived_snapshot_does_not_mutate_prediction_snapshot() -> None:
    source = daily_snapshot()
    before = source.to_dict()
    build_intelligence_snapshot(source)
    assert source.to_dict() == before


def test_prediction_hash_is_preserved_in_derived_record() -> None:
    source = daily_snapshot()
    derived = build_intelligence_snapshot(source)
    source_hashes = {record.symbol: record.prediction_hash for record in source.records}
    assert all(record.prediction_hash == source_hashes[record.symbol] for record in derived.records)


def test_derived_snapshot_is_deterministic() -> None:
    source = daily_snapshot()
    first = build_intelligence_snapshot(source)
    second = build_intelligence_snapshot(source)
    assert first.to_dict() == second.to_dict()


def test_single_cross_section_value_is_not_fabricated_percentile() -> None:
    source = build_daily_snapshot((prediction(),))
    derived = build_intelligence_snapshot(source)
    ranking = next(c for c in derived.records[0].stock_score.components if c.name == "ranking")
    assert not ranking.available
    assert ranking.normalized_value is None


def test_tied_cross_section_uses_average_percentile() -> None:
    source = build_daily_snapshot(
        (prediction("000001", market_rank=1), prediction("000002", market_rank=1))
    )
    derived = build_intelligence_snapshot(source)
    rankings = [
        next(c for c in record.stock_score.components if c.name == "ranking").normalized_value
        for record in derived.records
    ]
    assert rankings == [0.5, 0.5]


def test_policy_artifacts_match_code_and_sidecars() -> None:
    root = Path("artifacts/prediction_v1/policies")
    for filename, policy in POLICY_FILES.items():
        path = root / filename
        assert json.loads(path.read_text(encoding="utf-8")) == policy
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == Path(str(path) + ".sha256").read_text().strip()
        )
        assert len(policy_hash(policy)) == 64


def test_policy_hashes_are_bound_to_results() -> None:
    item = prediction()
    assert confidence_score_v1(item, ProductEvidence()).policy_hash == CONFIDENCE_POLICY_HASH
    assert prediction_risk_score_v1(item, ProductEvidence()).policy_hash == RISK_POLICY_HASH
    stock = stock_score_v1(
        ranking=1.0,
        expected_return=1.0,
        probability=1.0,
        agreement=None,
        industry=None,
        regime=None,
        confidence_score=None,
        risk_score=None,
    )
    assert stock.policy_hash == STOCK_SCORE_POLICY_HASH


def test_product_policy_is_not_model_promotion_or_execution_evidence() -> None:
    derived = build_intelligence_snapshot(daily_snapshot())
    assert derived.policy_type == "PRODUCT_PRESENTATION_FUSION"
    assert derived.not_alpha_model
    assert derived.not_model_promotion_evidence
    assert not derived.production_prediction_ready
    assert not derived.execution_authorized
    assert not any(record.execution_authorized for record in derived.records)


def test_protected_v6_v1r4_gen2_locks_unchanged() -> None:
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in PROTECTED_LOCKS}
    build_intelligence_snapshot(daily_snapshot())
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in PROTECTED_LOCKS}
    assert after == before


def test_api_enrichment_keeps_legacy_routes_and_canonical_confidence(monkeypatch) -> None:
    from stockpilot import api

    source = daily_snapshot()
    monkeypatch.setattr(api, "_canonical_prediction_snapshot", lambda: source)
    payload = api.canonical_latest_predictions()
    paths = {route.path for route in api.app.routes}
    assert "/predictions/latest" in paths
    assert "/predictions/{symbol}" in paths
    assert payload["records"][0]["confidence_score"] == 0.70
    assert "product_confidence_score" in payload["records"][0]
    assert "stock_score_status" in payload["records"][0]
    assert "score_coverage" in payload["records"][0]
    assert "confidence_band" in payload["records"][0]
    assert "risk_band" in payload["records"][0]
    assert payload["snapshot_hash"] == source.snapshot_hash
    assert payload["intelligence_snapshot_hash"]


def test_api_top_uses_product_rank_and_retains_model_rank(monkeypatch) -> None:
    from stockpilot import api

    monkeypatch.setattr(api, "_canonical_prediction_snapshot", daily_snapshot)
    payload = api.canonical_top_predictions(10)
    assert payload["rank_type"] == "product_rank"
    assert all("product_rank" in record and "model_rank" in record for record in payload["records"])


def test_api_symbol_exposes_full_decomposition(monkeypatch) -> None:
    from stockpilot import api

    monkeypatch.setattr(api, "_canonical_prediction_snapshot", daily_snapshot)
    payload = api.canonical_symbol_prediction("1")
    assert payload["symbol"] == "000001"
    assert payload["intelligence"]["stock_score"]["components"]
    assert payload["intelligence"]["confidence_score"]["components"]
    assert payload["intelligence"]["prediction_risk_score"]["components"]


def test_score_result_rejects_tampered_hash() -> None:
    result = stock_score_v1(
        ranking=1.0,
        expected_return=1.0,
        probability=1.0,
        agreement=None,
        industry=None,
        regime=None,
        confidence_score=None,
        risk_score=None,
    )
    with pytest.raises(ValueError, match="result_hash"):
        replace(result, result_hash="0" * 64)


def test_weight_sets_each_sum_to_one() -> None:
    for policy in (STOCK_SCORE_POLICY, CONFIDENCE_POLICY, RISK_POLICY):
        assert sum(policy["weights"].values()) == pytest.approx(1.0)


def test_generic_score_has_no_hidden_contribution() -> None:
    result = finish_score(
        score_type="TEST",
        policy_version="test-v1",
        policy_hash="a" * 64,
        minimum_coverage=0.0,
        band_thresholds=(20.0, 40.0, 60.0, 80.0),
        components=(),
    )
    assert result.score is None or result.score == 0.0
