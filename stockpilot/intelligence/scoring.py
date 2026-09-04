from __future__ import annotations

from .components import ScoreResult, component, finish_score
from .policies import STOCK_SCORE_POLICY, policy_hash

STOCK_SCORE_POLICY_VERSION = str(STOCK_SCORE_POLICY["policy_version"])
STOCK_SCORE_POLICY_HASH = policy_hash(STOCK_SCORE_POLICY)


def stock_score_v1(
    *,
    ranking: float | None,
    expected_return: float | None,
    probability: float | None,
    agreement: float | None,
    industry: float | None,
    regime: float | None,
    confidence_score: float | None,
    risk_score: float | None,
) -> ScoreResult:
    weights = STOCK_SCORE_POLICY["weights"]
    risk_retention = 1.0 - risk_score / 100.0 if risk_score is not None else None
    components = (
        component(
            "ranking", ranking, ranking, "same_snapshot_market_rank_percentile", weights["ranking"]
        ),
        component(
            "expected_return",
            expected_return,
            expected_return,
            "same_snapshot_expected_return_percentile",
            weights["expected_return"],
        ),
        component(
            "probability",
            probability,
            probability,
            "same_snapshot_probability_percentile",
            weights["probability"],
        ),
        component(
            "independent_model_agreement",
            agreement,
            agreement,
            "model_lineage_v1",
            weights["independent_model_agreement"],
        ),
        component("industry", industry, industry, "optional_product_evidence", weights["industry"]),
        component("regime", regime, regime, "optional_product_evidence", weights["regime"]),
        component(
            "confidence",
            confidence_score,
            confidence_score / 100.0 if confidence_score is not None else None,
            "confidence_score_v1",
            weights["confidence"],
        ),
        component(
            "risk_retention",
            risk_score,
            risk_retention,
            "one_minus_prediction_risk_score_v1",
            weights["risk_retention"],
        ),
    )
    return finish_score(
        score_type="STOCK_SCORE",
        policy_version=STOCK_SCORE_POLICY_VERSION,
        policy_hash=STOCK_SCORE_POLICY_HASH,
        minimum_coverage=float(STOCK_SCORE_POLICY["minimum_component_coverage"]),
        band_thresholds=tuple(STOCK_SCORE_POLICY["band_thresholds"]),
        components=components,
    )
