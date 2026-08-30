from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import ReadinessThresholds
from .integrity import verify_immutable
from .observation import verify_source_receipt


@dataclass(frozen=True)
class ReadinessStatus:
    source_observation_count: int
    pit_observation_count: int
    mature_1d_count: int
    mature_5d_count: int
    mature_20d_count: int
    observation_quality_ready: bool
    label_quality_ready: bool
    factor_validation_ready: bool
    model_training_ready: bool
    replacement_evaluation_ready: bool
    production_prediction_ready: bool
    execution_authorized: bool

    def to_dict(self) -> dict:
        return asdict(self)


def observation_qualifies(item: dict, thresholds: ReadinessThresholds) -> bool:
    source = item.get("sources", {}).get("earnings_expectations", {})
    hashes = (
        isinstance(item.get("universe_hash"), str) and len(item["universe_hash"]) == 64
        and isinstance(item.get("pit_membership_snapshot_hash"), str)
        and len(item["pit_membership_snapshot_hash"]) == 64
        and isinstance(item.get("pit_industry_mapping_hash"), str)
        and len(item["pit_industry_mapping_hash"]) == 64
        and isinstance(item.get("trading_calendar_hash"), str)
        and len(item["trading_calendar_hash"]) == 64
    )
    try:
        observation_integrity = (
            verify_immutable(item["observation_path"]) == item["observation_sha256"]
        )
    except (KeyError, OSError, RuntimeError):
        observation_integrity = False
    return bool(
        item.get("verified_shanghai_trading_date")
        and item.get("observation_immutable_verified")
        and observation_integrity
        and source.get("source_status") == "SUCCESS"
        and float(source.get("universe_coverage") or 0) >= thresholds.minimum_expectation_coverage
        and source.get("hashes_verified") is True
        and hashes
        and verify_source_receipt(source)
    )


def _qualified_mature_dates(labels: list[dict], horizon: int, thresholds: ReadinessThresholds) -> set[str]:
    grouped: dict[str, list[dict]] = {}
    for item in labels:
        if item.get("horizon") == horizon:
            grouped.setdefault(item["prediction_date"], []).append(item)
    qualified: set[str] = set()
    for date, rows in grouped.items():
        expected_sizes = {int(item.get("expected_universe_size") or 0) for item in rows}
        if len(expected_sizes) != 1:
            continue
        expected = next(iter(expected_sizes))
        valid = {
            item["symbol"] for item in rows
            if item.get("status") == "SETTLED"
            and item.get("label_fully_verified") is True
            and item.get("price_provenance_verified") is True
            and item.get("benchmark_provenance_verified") is True
            and item.get("corporate_action_verified") is True
        }
        coverage = len(valid) / expected if expected else 0.0
        if len(valid) >= thresholds.minimum_label_symbols and coverage >= thresholds.minimum_label_coverage:
            qualified.add(date)
    return qualified


def derive_readiness(
    observations: list[dict],
    labels: list[dict],
    *,
    thresholds: ReadinessThresholds | None = None,
) -> ReadinessStatus:
    thresholds = thresholds or ReadinessThresholds()
    qualifying = {
        item["target_date"] for item in observations if observation_qualifies(item, thresholds)
    }
    mature = {
        horizon: _qualified_mature_dates(labels, horizon, thresholds) for horizon in (1, 5, 20)
    }
    observation_ready = len(qualifying) >= thresholds.minimum_observation_dates
    label_ready = all(len(mature[horizon]) >= thresholds.minimum_label_dates for horizon in mature)
    factor_ready = observation_ready and label_ready
    return ReadinessStatus(
        source_observation_count=len({item["observation_id"] for item in observations}),
        pit_observation_count=len(qualifying),
        mature_1d_count=len(mature[1]),
        mature_5d_count=len(mature[5]),
        mature_20d_count=len(mature[20]),
        observation_quality_ready=observation_ready,
        label_quality_ready=label_ready,
        factor_validation_ready=factor_ready,
        # A later, separately frozen factor-validation decision is required.
        model_training_ready=False,
        replacement_evaluation_ready=False,
        production_prediction_ready=False,
        execution_authorized=False,
    )
