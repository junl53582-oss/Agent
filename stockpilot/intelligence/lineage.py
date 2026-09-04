from __future__ import annotations

from dataclasses import dataclass

from .policies import LINEAGE_POLICY, policy_hash

LINEAGE_POLICY_VERSION = str(LINEAGE_POLICY["policy_version"])
LINEAGE_POLICY_HASH = policy_hash(LINEAGE_POLICY)


@dataclass(frozen=True)
class ModelOutput:
    source_kind: str
    score: float
    source_prediction_hash: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("model output score must be in [0,1]")


@dataclass(frozen=True)
class AgreementEvidence:
    score: float | None
    independent_family_count: int
    selected_sources: tuple[str, ...]
    selected_families: tuple[str, ...]
    policy_version: str = LINEAGE_POLICY_VERSION
    policy_hash: str = LINEAGE_POLICY_HASH


def independent_model_agreement(outputs: tuple[ModelOutput, ...]) -> AgreementEvidence:
    families = LINEAGE_POLICY["family_by_source_kind"]
    priorities = LINEAGE_POLICY["priority_by_source_kind"]
    selected: dict[str, ModelOutput] = {}
    for output in outputs:
        family = families.get(output.source_kind)
        if family is None:
            continue
        current = selected.get(family)
        current_priority = priorities.get(current.source_kind, -1) if current else -1
        candidate_key = (priorities.get(output.source_kind, -1), output.source_kind)
        current_key = (current_priority, current.source_kind) if current else (-1, "")
        if current is None or candidate_key > current_key:
            selected[family] = output
    ordered = tuple(sorted(selected.items()))
    scores = [output.score for _, output in ordered]
    minimum = int(LINEAGE_POLICY["minimum_independent_families"])
    score = 1.0 - (max(scores) - min(scores)) if len(scores) >= minimum else None
    return AgreementEvidence(
        score=score,
        independent_family_count=len(scores),
        selected_sources=tuple(output.source_kind for _, output in ordered),
        selected_families=tuple(family for family, _ in ordered),
    )
