from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .schema import _json_ready, canonical_json_bytes, sha256_bytes


@dataclass(frozen=True)
class ScoreComponent:
    name: str
    raw_value: Any
    normalized_value: float | None
    available: bool
    source: str
    weight: float
    contribution: float | None = None

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError("component weight cannot be negative")
        if self.available != (self.normalized_value is not None):
            raise ValueError("component availability must match normalized-value availability")
        if self.normalized_value is not None and not 0.0 <= self.normalized_value <= 1.0:
            raise ValueError("normalized component value must be in [0,1]")
        if not self.available and self.contribution is not None:
            raise ValueError("unavailable component cannot have a contribution")

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass(frozen=True)
class ScoreResult:
    score_type: str
    policy_version: str
    policy_hash: str
    status: str
    score: float | None
    band: str | None
    component_coverage: float
    minimum_coverage: float
    components: tuple[ScoreComponent, ...]
    result_hash: str = field(default="")

    def __post_init__(self) -> None:
        expected = sha256_bytes(canonical_json_bytes(self.hash_payload()))
        if self.result_hash and self.result_hash != expected:
            raise ValueError("result_hash does not match score result bytes")
        object.__setattr__(self, "result_hash", expected)

    def hash_payload(self) -> dict[str, Any]:
        return {
            "score_type": self.score_type,
            "policy_version": self.policy_version,
            "policy_hash": self.policy_hash,
            "status": self.status,
            "score": self.score,
            "band": self.band,
            "component_coverage": self.component_coverage,
            "minimum_coverage": self.minimum_coverage,
            "components": [component.to_dict() for component in self.components],
        }

    def to_dict(self) -> dict[str, Any]:
        return self.hash_payload() | {"result_hash": self.result_hash}


def score_band(score: float, thresholds: tuple[float, float, float, float]) -> str:
    very_low, low, medium, high = thresholds
    if score < very_low:
        return "VERY_LOW"
    if score < low:
        return "LOW"
    if score < medium:
        return "MEDIUM"
    if score < high:
        return "HIGH"
    return "VERY_HIGH"


def finish_score(
    *,
    score_type: str,
    policy_version: str,
    policy_hash: str,
    minimum_coverage: float,
    band_thresholds: tuple[float, float, float, float],
    components: tuple[ScoreComponent, ...],
) -> ScoreResult:
    coverage = sum(component.weight for component in components if component.available)
    available = coverage > 0.0 and coverage + 1e-12 >= minimum_coverage
    score = None
    finalized = components
    if available:
        score = (
            100.0
            * sum(
                component.weight * float(component.normalized_value)
                for component in components
                if component.available and component.normalized_value is not None
            )
            / coverage
        )
        finalized = tuple(
            ScoreComponent(
                name=component.name,
                raw_value=component.raw_value,
                normalized_value=component.normalized_value,
                available=component.available,
                source=component.source,
                weight=component.weight,
                contribution=(
                    100.0 * component.weight * float(component.normalized_value) / coverage
                    if component.available and component.normalized_value is not None
                    else None
                ),
            )
            for component in components
        )
    return ScoreResult(
        score_type=score_type,
        policy_version=policy_version,
        policy_hash=policy_hash,
        status="OK" if available else "INSUFFICIENT_COMPONENT_COVERAGE",
        score=score,
        band=score_band(score, band_thresholds) if score is not None else None,
        component_coverage=coverage,
        minimum_coverage=minimum_coverage,
        components=finalized,
    )


def component(
    name: str,
    raw_value: Any,
    normalized_value: float | None,
    source: str,
    weight: float,
) -> ScoreComponent:
    return ScoreComponent(
        name=name,
        raw_value=raw_value,
        normalized_value=normalized_value,
        available=normalized_value is not None,
        source=source,
        weight=weight,
    )
