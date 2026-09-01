from __future__ import annotations

from .components import ScoreResult, component, finish_score
from .evidence import ProductEvidence
from .lineage import independent_model_agreement
from .policies import CONFIDENCE_POLICY, policy_hash
from .schema import CanonicalPrediction

CONFIDENCE_POLICY_VERSION = str(CONFIDENCE_POLICY["policy_version"])
CONFIDENCE_POLICY_HASH = policy_hash(CONFIDENCE_POLICY)


def _probabilities(prediction: CanonicalPrediction) -> tuple[float, ...]:
    return tuple(
        value
        for value in (prediction.up_prob_1d, prediction.up_prob_5d, prediction.up_prob_20d)
        if value is not None
    )


def confidence_score_v1(prediction: CanonicalPrediction, evidence: ProductEvidence) -> ScoreResult:
    weights = CONFIDENCE_POLICY["weights"]
    agreement = independent_model_agreement(evidence.model_outputs)
    probabilities = _probabilities(prediction)
    dispersion_raw = max(probabilities) - min(probabilities) if len(probabilities) >= 2 else None
    dispersion = (
        1.0 - min(1.0, dispersion_raw / float(CONFIDENCE_POLICY["dispersion_scale"]))
        if dispersion_raw is not None
        else None
    )
    components = (
        component(
            "independent_model_agreement",
            {
                "independent_family_count": agreement.independent_family_count,
                "selected_sources": agreement.selected_sources,
                "selected_families": agreement.selected_families,
            },
            agreement.score,
            "model_lineage_v1",
            weights["independent_model_agreement"],
        ),
        component(
            "historical_calibration",
            evidence.historical_calibration,
            evidence.historical_calibration,
            "optional_product_evidence",
            weights["historical_calibration"],
        ),
        component(
            "data_completeness",
            evidence.data_completeness,
            evidence.data_completeness,
            "canonical_or_optional_product_evidence",
            weights["data_completeness"],
        ),
        component(
            "feature_completeness",
            evidence.feature_completeness,
            evidence.feature_completeness,
            "canonical_or_optional_product_evidence",
            weights["feature_completeness"],
        ),
        component(
            "drift_ood_quality",
            evidence.drift_ood_quality,
            evidence.drift_ood_quality,
            "optional_product_evidence",
            weights["drift_ood_quality"],
        ),
        component(
            "regime_familiarity",
            evidence.regime_familiarity,
            evidence.regime_familiarity,
            "optional_product_evidence",
            weights["regime_familiarity"],
        ),
        component(
            "prediction_dispersion",
            dispersion_raw,
            dispersion,
            "canonical_horizon_probabilities",
            weights["prediction_dispersion"],
        ),
        component(
            "historical_stability",
            evidence.historical_stability,
            evidence.historical_stability,
            "optional_product_evidence",
            weights["historical_stability"],
        ),
        component(
            "upstream_confidence_evidence",
            prediction.confidence_score,
            prediction.confidence_score,
            "canonical_upstream_confidence_field",
            weights["upstream_confidence_evidence"],
        ),
    )
    return finish_score(
        score_type="CONFIDENCE_SCORE",
        policy_version=CONFIDENCE_POLICY_VERSION,
        policy_hash=CONFIDENCE_POLICY_HASH,
        minimum_coverage=float(CONFIDENCE_POLICY["minimum_component_coverage"]),
        band_thresholds=tuple(CONFIDENCE_POLICY["band_thresholds"]),
        components=components,
    )
