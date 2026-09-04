from __future__ import annotations

from typing import Any

from .schema import canonical_json_bytes, sha256_bytes

PRODUCT_POLICY_SCOPE = {
    "policy_type": "PRODUCT_PRESENTATION_FUSION",
    "not_alpha_model": True,
    "not_model_promotion_evidence": True,
    "production_prediction_ready": False,
    "execution_authorized": False,
    "missing_value_policy": "null_not_fabricated",
}

STOCK_SCORE_POLICY: dict[str, Any] = PRODUCT_POLICY_SCOPE | {
    "policy_version": "stock-score-v1.0.0",
    "score_direction": "higher_is_more_attractive",
    "minimum_component_coverage": 0.60,
    "band_thresholds": [20.0, 40.0, 60.0, 80.0],
    "normalization": {
        "ranking": "inverse market rank; same-snapshot average-tie percentile",
        "expected_return": "mean available horizons; same-snapshot average-tie percentile",
        "probability": "mean available horizons; same-snapshot average-tie percentile",
        "independent_model_agreement": "model-lineage-v1 agreement in [0,1]",
        "industry": "pre-normalized real optional evidence in [0,1]",
        "regime": "pre-normalized real optional evidence in [0,1]",
        "confidence": "confidence-score-v1 divided by 100",
        "risk_retention": "one minus prediction-risk-v1 divided by 100",
        "cross_section_rule": "fewer than two available values is null",
    },
    "weights": {
        "ranking": 0.30,
        "expected_return": 0.20,
        "probability": 0.15,
        "independent_model_agreement": 0.10,
        "industry": 0.05,
        "regime": 0.05,
        "confidence": 0.10,
        "risk_retention": 0.05,
    },
}

CONFIDENCE_POLICY: dict[str, Any] = PRODUCT_POLICY_SCOPE | {
    "policy_version": "confidence-score-v1.0.0",
    "score_direction": "higher_is_more_confident",
    "minimum_component_coverage": 0.25,
    "band_thresholds": [20.0, 40.0, 60.0, 80.0],
    "weights": {
        "independent_model_agreement": 0.15,
        "historical_calibration": 0.15,
        "data_completeness": 0.10,
        "feature_completeness": 0.10,
        "drift_ood_quality": 0.10,
        "regime_familiarity": 0.05,
        "prediction_dispersion": 0.15,
        "historical_stability": 0.05,
        "upstream_confidence_evidence": 0.15,
    },
    "dispersion_scale": 0.25,
    "normalization": {
        "prediction_dispersion": "1 - min(1, horizon_probability_range / 0.25)",
        "upstream_confidence_evidence": "canonical upstream field in [0,1]",
        "other_components": "pre-normalized real evidence in [0,1]; absent is null",
    },
}

RISK_POLICY: dict[str, Any] = PRODUCT_POLICY_SCOPE | {
    "policy_version": "prediction-risk-v1.0.0",
    "score_direction": "higher_is_more_risky",
    "minimum_component_coverage": 0.25,
    "band_thresholds": [20.0, 40.0, 60.0, 80.0],
    "weights": {
        "volatility": 0.20,
        "drawdown": 0.10,
        "liquidity": 0.15,
        "model_uncertainty": 0.20,
        "horizon_dispersion": 0.10,
        "drift_ood": 0.10,
        "regime": 0.05,
        "data_quality": 0.10,
    },
    "dispersion_scale": 0.25,
    "normalization": {
        "model_uncertainty": "1 - min(1, mean_abs_probability_distance_from_0.5 / 0.5)",
        "horizon_dispersion": "min(1, horizon_probability_range / 0.25)",
        "other_components": "pre-normalized real risk evidence in [0,1]; absent is null",
    },
}

LINEAGE_POLICY: dict[str, Any] = PRODUCT_POLICY_SCOPE | {
    "policy_version": "model-lineage-v1.0.0",
    "family_by_source_kind": {
        "V6": "V6_RANKING_FAMILY",
        "V30": "V30_PROBABILITY_FAMILY",
        "V30R1": "V30_PROBABILITY_FAMILY",
        "V30R1_FORWARD_R2": "V30_PROBABILITY_FAMILY",
        "GEN2": "GEN2_RESEARCH_FAMILY",
    },
    "priority_by_source_kind": {
        "V6": 10,
        "V30": 20,
        "V30R1": 30,
        "V30R1_FORWARD_R2": 40,
        "GEN2": 10,
    },
    "minimum_independent_families": 2,
    "deduplication": "one_highest_priority_output_per_independent_family",
    "agreement_normalization": "1 - range of selected independent-family scores",
}


def policy_hash(policy: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(policy))
