from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

PREDICTION_SCHEMA_VERSION = "stockpilot-prediction-v1.0.0"
PROHIBITED_UNIVERSE_IDS = frozenset({"ALL_A_SHARE", "ALL_A_SHARES"})


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _json_ready(value.item())
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize product evidence deterministically without permitting NaN."""
    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class CanonicalPrediction:
    # Identity
    symbol: str
    stock_name: str | None
    industry: str | None
    prediction_date: str
    prediction_timestamp: str

    # Coverage
    universe_id: str
    eligibility_status: str
    data_coverage: float | None
    feature_coverage: float | None

    # Raw prediction
    raw_rank_score: float | None
    return_1d_pred: float | None
    return_5d_pred: float | None
    return_20d_pred: float | None
    up_prob_1d: float | None
    up_prob_5d: float | None
    up_prob_20d: float | None
    rank_1d: int | None
    rank_5d: int | None
    rank_20d: int | None
    market_rank: int | None
    industry_rank: int | None

    # Fusion placeholders. Phase 1 never invents final product scores.
    stock_score: float | None
    ranking_score: float | None
    expected_return_score: float | None
    probability_score: float | None
    agreement_score: float | None
    industry_score: float | None
    regime_score: float | None
    confidence_score: float | None
    risk_score: float | None

    # Explanation placeholders
    positive_drivers: tuple[str, ...] | None
    negative_drivers: tuple[str, ...] | None
    risk_drivers: tuple[str, ...] | None

    # Governance
    model_version: str
    feature_version: str | None
    training_cutoff: str | None
    data_snapshot_hash: str | None
    model_manifest_hash: str | None
    production_prediction_ready: bool
    execution_authorized: bool

    # Product provenance
    source_kind: str
    source_artifact_path: str
    source_artifact_hash: str
    source_snapshot_hash: str
    adapter_version: str
    schema_version: str = PREDICTION_SCHEMA_VERSION
    prediction_hash: str = field(default="")

    def __post_init__(self) -> None:
        normalized_symbol = str(self.symbol).zfill(6)
        object.__setattr__(self, "symbol", normalized_symbol)
        if self.universe_id.upper() in PROHIBITED_UNIVERSE_IDS:
            raise ValueError("ALL_A_SHARE universe semantics are not yet supported by evidence")
        if self.execution_authorized:
            raise ValueError("Prediction V1 Phase 1 cannot authorize execution")
        expected = sha256_bytes(canonical_json_bytes(self.hash_payload()))
        if self.prediction_hash and self.prediction_hash != expected:
            raise ValueError("prediction_hash does not match canonical prediction bytes")
        object.__setattr__(self, "prediction_hash", expected)

    def hash_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("prediction_hash", None)
        return _json_ready(payload)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


def prediction_schema_definition() -> dict[str, Any]:
    """Machine-readable contract; this is product schema evidence, not model evidence."""
    nullable = {
        "stock_name",
        "industry",
        "data_coverage",
        "feature_coverage",
        "raw_rank_score",
        "return_1d_pred",
        "return_5d_pred",
        "return_20d_pred",
        "up_prob_1d",
        "up_prob_5d",
        "up_prob_20d",
        "rank_1d",
        "rank_5d",
        "rank_20d",
        "market_rank",
        "industry_rank",
        "stock_score",
        "ranking_score",
        "expected_return_score",
        "probability_score",
        "agreement_score",
        "industry_score",
        "regime_score",
        "confidence_score",
        "risk_score",
        "positive_drivers",
        "negative_drivers",
        "risk_drivers",
        "feature_version",
        "training_cutoff",
        "data_snapshot_hash",
        "model_manifest_hash",
    }
    fields = []
    for name in CanonicalPrediction.__dataclass_fields__:
        fields.append({"name": name, "nullable": name in nullable})
    return {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "scope": "PRODUCT_INTERFACE_ONLY_NOT_MODEL_PROMOTION_EVIDENCE",
        "fields": fields,
        "missing_value_policy": "null_not_fabricated",
        "universe_policy": "CSI300_VALIDATED_SCOPE_ONLY_UNTIL_NEW_PIT_UNIVERSE_EVIDENCE",
        "execution_authorized": False,
    }
