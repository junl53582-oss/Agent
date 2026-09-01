from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .components import ScoreResult
from .confidence import confidence_score_v1
from .evidence import ProductEvidence
from .lineage import LINEAGE_POLICY_HASH, ModelOutput, independent_model_agreement
from .policies import PRODUCT_POLICY_SCOPE
from .risk import prediction_risk_score_v1
from .schema import CanonicalPrediction, _json_ready, canonical_json_bytes, sha256_bytes
from .scoring import stock_score_v1
from .snapshot import CanonicalDailySnapshot

INTELLIGENCE_SCHEMA_VERSION = "stockpilot-intelligence-v1.0.0"


def _average(values: tuple[float | None, ...]) -> float | None:
    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else None


def _percentiles(values: Mapping[str, float | None]) -> dict[str, float | None]:
    available = [(symbol, value) for symbol, value in values.items() if value is not None]
    if len(available) < 2:
        return {symbol: None for symbol in values}
    ordered = sorted(available, key=lambda item: (float(item[1]), item[0]))
    result: dict[str, float | None] = {symbol: None for symbol in values}
    index = 0
    denominator = len(ordered) - 1
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and ordered[end + 1][1] == ordered[index][1]:
            end += 1
        percentile = ((index + end) / 2.0) / denominator
        for position in range(index, end + 1):
            result[ordered[position][0]] = percentile
        index = end + 1
    return result


@dataclass(frozen=True)
class IntelligenceRecord:
    symbol: str
    prediction_hash: str
    model_rank: int | None
    product_rank: int | None
    stock_score: ScoreResult
    confidence_score: ScoreResult
    prediction_risk_score: ScoreResult
    schema_version: str = INTELLIGENCE_SCHEMA_VERSION
    production_prediction_ready: bool = False
    execution_authorized: bool = False
    derived_hash: str = field(default="")

    def __post_init__(self) -> None:
        if self.production_prediction_ready or self.execution_authorized:
            raise ValueError(
                "product intelligence cannot authorize production prediction or execution"
            )
        expected = sha256_bytes(canonical_json_bytes(self.hash_payload()))
        if self.derived_hash and self.derived_hash != expected:
            raise ValueError("derived_hash does not match intelligence record bytes")
        object.__setattr__(self, "derived_hash", expected)

    def hash_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("derived_hash", None)
        return _json_ready(payload)

    def to_dict(self) -> dict[str, Any]:
        return self.hash_payload() | {"derived_hash": self.derived_hash}


@dataclass(frozen=True)
class DailyIntelligenceSnapshot:
    prediction_date: str
    universe_id: str
    prediction_snapshot_hash: str
    records: tuple[IntelligenceRecord, ...]
    policy_hashes: dict[str, str]
    schema_version: str = INTELLIGENCE_SCHEMA_VERSION
    policy_type: str = str(PRODUCT_POLICY_SCOPE["policy_type"])
    not_alpha_model: bool = True
    not_model_promotion_evidence: bool = True
    production_prediction_ready: bool = False
    execution_authorized: bool = False
    snapshot_hash: str = field(default="")

    def __post_init__(self) -> None:
        if self.production_prediction_ready or self.execution_authorized:
            raise ValueError("product intelligence snapshot cannot authorize execution")
        expected = sha256_bytes(canonical_json_bytes(self.hash_payload()))
        if self.snapshot_hash and self.snapshot_hash != expected:
            raise ValueError("snapshot_hash does not match intelligence snapshot bytes")
        object.__setattr__(self, "snapshot_hash", expected)

    def hash_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("snapshot_hash", None)
        return _json_ready(payload)

    def to_dict(self) -> dict[str, Any]:
        return self.hash_payload() | {"snapshot_hash": self.snapshot_hash}


def _enrich_evidence(
    prediction: CanonicalPrediction,
    evidence: ProductEvidence,
    probability_mean: float | None,
    ranking_percentile: float | None,
) -> ProductEvidence:
    output_score = probability_mean if probability_mean is not None else ranking_percentile
    outputs = evidence.model_outputs
    if not outputs and output_score is not None:
        outputs = (ModelOutput(prediction.source_kind, output_score, prediction.prediction_hash),)
    return replace(
        evidence,
        model_outputs=outputs,
        data_completeness=(
            evidence.data_completeness
            if evidence.data_completeness is not None
            else prediction.data_coverage
        ),
        feature_completeness=(
            evidence.feature_completeness
            if evidence.feature_completeness is not None
            else prediction.feature_coverage
        ),
    )


def build_intelligence_snapshot(
    snapshot: CanonicalDailySnapshot,
    evidence_by_symbol: Mapping[str, ProductEvidence] | None = None,
) -> DailyIntelligenceSnapshot:
    evidence_by_symbol = evidence_by_symbol or {}
    predictions = {record.symbol: record for record in snapshot.records}
    ranking_values = {
        symbol: -float(record.market_rank) if record.market_rank is not None else None
        for symbol, record in predictions.items()
    }
    expected_values = {
        symbol: _average((record.return_1d_pred, record.return_5d_pred, record.return_20d_pred))
        for symbol, record in predictions.items()
    }
    probability_values = {
        symbol: _average((record.up_prob_1d, record.up_prob_5d, record.up_prob_20d))
        for symbol, record in predictions.items()
    }
    ranking_percentiles = _percentiles(ranking_values)
    expected_percentiles = _percentiles(expected_values)
    probability_percentiles = _percentiles(probability_values)

    provisional: list[IntelligenceRecord] = []
    for symbol, prediction in predictions.items():
        supplied = evidence_by_symbol.get(symbol, ProductEvidence())
        evidence = _enrich_evidence(
            prediction, supplied, probability_values[symbol], ranking_percentiles[symbol]
        )
        confidence = confidence_score_v1(prediction, evidence)
        risk = prediction_risk_score_v1(prediction, evidence)
        agreement = independent_model_agreement(evidence.model_outputs)
        stock = stock_score_v1(
            ranking=ranking_percentiles[symbol],
            expected_return=expected_percentiles[symbol],
            probability=probability_percentiles[symbol],
            agreement=agreement.score,
            industry=evidence.industry_score,
            regime=evidence.regime_score,
            confidence_score=confidence.score,
            risk_score=risk.score,
        )
        provisional.append(
            IntelligenceRecord(
                symbol=symbol,
                prediction_hash=prediction.prediction_hash,
                model_rank=prediction.market_rank,
                product_rank=None,
                stock_score=stock,
                confidence_score=confidence,
                prediction_risk_score=risk,
            )
        )
    rankable = sorted(
        (record for record in provisional if record.stock_score.score is not None),
        key=lambda record: (-float(record.stock_score.score), record.symbol),
    )
    rank_by_symbol = {record.symbol: rank for rank, record in enumerate(rankable, start=1)}
    records = tuple(
        replace(record, product_rank=rank_by_symbol.get(record.symbol), derived_hash="")
        for record in sorted(
            provisional,
            key=lambda record: (rank_by_symbol.get(record.symbol, 10**9), record.symbol),
        )
    )
    first = records[0]
    return DailyIntelligenceSnapshot(
        prediction_date=snapshot.prediction_date,
        universe_id=snapshot.universe_id,
        prediction_snapshot_hash=snapshot.snapshot_hash,
        records=records,
        policy_hashes={
            "stock_score": first.stock_score.policy_hash,
            "confidence_score": first.confidence_score.policy_hash,
            "prediction_risk_score": first.prediction_risk_score.policy_hash,
            "model_lineage": LINEAGE_POLICY_HASH,
        },
    )


def enrich_prediction_record(
    prediction: CanonicalPrediction, intelligence: IntelligenceRecord
) -> dict[str, Any]:
    result = prediction.to_dict()
    result.update(
        {
            "stock_score": intelligence.stock_score.score,
            "stock_score_status": intelligence.stock_score.status,
            "stock_score_version": intelligence.stock_score.policy_version,
            "score_coverage": intelligence.stock_score.component_coverage,
            "confidence": intelligence.confidence_score.score,
            "product_confidence_score": intelligence.confidence_score.score,
            "confidence_band": intelligence.confidence_score.band,
            "confidence_coverage": intelligence.confidence_score.component_coverage,
            "confidence_version": intelligence.confidence_score.policy_version,
            "risk": intelligence.prediction_risk_score.score,
            "prediction_risk_score": intelligence.prediction_risk_score.score,
            "risk_band": intelligence.prediction_risk_score.band,
            "risk_coverage": intelligence.prediction_risk_score.component_coverage,
            "risk_version": intelligence.prediction_risk_score.policy_version,
            "model_rank": intelligence.model_rank,
            "product_rank": intelligence.product_rank,
            "intelligence": intelligence.to_dict(),
        }
    )
    return result
