from __future__ import annotations

from .components import ScoreResult, component, finish_score
from .evidence import ProductEvidence
from .policies import RISK_POLICY, policy_hash
from .schema import CanonicalPrediction

RISK_POLICY_VERSION = str(RISK_POLICY["policy_version"])
RISK_POLICY_HASH = policy_hash(RISK_POLICY)


def prediction_risk_score_v1(
    prediction: CanonicalPrediction, evidence: ProductEvidence
) -> ScoreResult:
    weights = RISK_POLICY["weights"]
    probabilities = tuple(
        value
        for value in (prediction.up_prob_1d, prediction.up_prob_5d, prediction.up_prob_20d)
        if value is not None
    )
    uncertainty_raw = (
        sum(abs(value - 0.5) for value in probabilities) / len(probabilities)
        if probabilities
        else None
    )
    uncertainty_risk = (
        1.0 - min(1.0, uncertainty_raw / 0.5) if uncertainty_raw is not None else None
    )
    dispersion_raw = max(probabilities) - min(probabilities) if len(probabilities) >= 2 else None
    dispersion_risk = (
        min(1.0, dispersion_raw / float(RISK_POLICY["dispersion_scale"]))
        if dispersion_raw is not None
        else None
    )
    components = (
        component(
            "volatility",
            evidence.volatility_risk,
            evidence.volatility_risk,
            "optional_product_evidence",
            weights["volatility"],
        ),
        component(
            "drawdown",
            evidence.drawdown_risk,
            evidence.drawdown_risk,
            "optional_product_evidence",
            weights["drawdown"],
        ),
        component(
            "liquidity",
            evidence.liquidity_risk,
            evidence.liquidity_risk,
            "optional_product_evidence",
            weights["liquidity"],
        ),
        component(
            "model_uncertainty",
            uncertainty_raw,
            uncertainty_risk,
            "canonical_horizon_probabilities",
            weights["model_uncertainty"],
        ),
        component(
            "horizon_dispersion",
            dispersion_raw,
            dispersion_risk,
            "canonical_horizon_probabilities",
            weights["horizon_dispersion"],
        ),
        component(
            "drift_ood",
            evidence.drift_ood_risk,
            evidence.drift_ood_risk,
            "optional_product_evidence",
            weights["drift_ood"],
        ),
        component(
            "regime",
            evidence.regime_risk,
            evidence.regime_risk,
            "optional_product_evidence",
            weights["regime"],
        ),
        component(
            "data_quality",
            evidence.data_quality_risk,
            evidence.data_quality_risk,
            "optional_product_evidence",
            weights["data_quality"],
        ),
    )
    return finish_score(
        score_type="PREDICTION_RISK_SCORE",
        policy_version=RISK_POLICY_VERSION,
        policy_hash=RISK_POLICY_HASH,
        minimum_coverage=float(RISK_POLICY["minimum_component_coverage"]),
        band_thresholds=tuple(RISK_POLICY["band_thresholds"]),
        components=components,
    )
