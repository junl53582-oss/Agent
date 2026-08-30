from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

from .certification import certify_observation
from .config import OperationalSettings
from .settlement import certify_label_record, load_approved_settlement_bundle


@dataclass(frozen=True)
class RuntimeStatus:
    active_version: str
    inherited_source_baseline_count: int
    runtime_source_observation_count: int
    qualified_pit_observation_count: int
    mature_1d_count: int
    mature_5d_count: int
    mature_20d_count: int
    observation_quality_ready: bool
    label_quality_ready: bool
    factor_validation_ready: bool
    model_training_ready: bool = False
    replacement_evaluation_ready: bool = False
    production_prediction_ready: bool = False
    execution_authorized: bool = False
    v31_trained: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _certification_dict(value: object) -> dict:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return value
    raise TypeError("certifier must return a dictionary-like result")


def _qualified_mature_dates(
    labels: list[dict],
    horizon: int,
    settings: OperationalSettings,
    label_certifier: Callable,
) -> set[str]:
    grouped: dict[str, list[tuple[dict, dict]]] = {}
    for record in labels:
        if int(record.get("horizon") or 0) != horizon:
            continue
        certification = _certification_dict(label_certifier(record, settings))
        grouped.setdefault(str(record.get("prediction_date")), []).append(
            (record, certification)
        )
    qualified: set[str] = set()
    for date, values in grouped.items():
        expected_sizes = {int(record.get("expected_universe_size") or 0) for record, _ in values}
        if len(expected_sizes) != 1:
            continue
        expected = next(iter(expected_sizes))
        symbols = {
            str(record.get("symbol")).zfill(6)
            for record, certification in values
            if certification.get("label_evidence_verified") is True
        }
        coverage = len(symbols) / expected if expected else 0.0
        if (
            len(symbols) >= settings.thresholds.minimum_label_symbols
            and coverage >= settings.thresholds.minimum_label_coverage
        ):
            qualified.add(date)
    return qualified


def build_runtime_status(
    settings: OperationalSettings,
    observations: list[dict],
    labels: list[dict],
    *,
    observation_certifier: Callable = certify_observation,
    label_certifier: Callable = certify_label_record,
) -> RuntimeStatus:
    certifications = [
        _certification_dict(observation_certifier(item, settings)) for item in observations
    ]
    qualifying_dates = {
        item["target_date"]
        for item in certifications
        if item.get("qualifying_observation") is True
    }
    effective_label_certifier = label_certifier
    if labels and label_certifier is certify_label_record:
        bundle = load_approved_settlement_bundle(settings)
        source_cache: dict = {}
        effective_label_certifier = lambda record, configured: certify_label_record(
            record, configured, bundle=bundle, source_cache=source_cache
        )
    mature = {
        horizon: _qualified_mature_dates(
            labels, horizon, settings, effective_label_certifier
        )
        for horizon in (1, 5, 20)
    }
    observation_ready = (
        len(qualifying_dates) >= settings.thresholds.minimum_observation_dates
    )
    label_ready = all(
        len(mature[horizon]) >= settings.thresholds.minimum_label_dates
        for horizon in mature
    )
    factor_ready = observation_ready and label_ready
    return RuntimeStatus(
        active_version=settings.version,
        inherited_source_baseline_count=settings.inherited_source_baseline_count,
        runtime_source_observation_count=len(
            {str(item.get("observation_id")) for item in observations}
        ),
        qualified_pit_observation_count=len(qualifying_dates),
        mature_1d_count=len(mature[1]),
        mature_5d_count=len(mature[5]),
        mature_20d_count=len(mature[20]),
        observation_quality_ready=observation_ready,
        label_quality_ready=label_ready,
        factor_validation_ready=factor_ready,
        # A separately frozen factor-validation decision is still mandatory.
        model_training_ready=False,
        replacement_evaluation_ready=False,
        production_prediction_ready=False,
        execution_authorized=False,
        v31_trained=False,
    )


def aggregate_daily_status(
    observation_status: str,
    prediction_status: str,
    settlement_status: str,
) -> str:
    if "INTERRUPTED" in {observation_status, prediction_status, settlement_status}:
        return "INTERRUPTED"
    prediction_ok = prediction_status in {"RECORDED", "ALREADY_RECORDED"}
    settlement_ok = settlement_status in {"SETTLED", "NO_MATURE_LABELS"}
    observation_ok = observation_status == "SUCCESS"
    if observation_ok and prediction_ok and settlement_ok:
        return "COMPLETE"
    if observation_ok and (
        prediction_status == "INPUT_NOT_AVAILABLE"
        or settlement_status.startswith("SETTLEMENT_BLOCKED_")
        or settlement_status == "SETTLEMENT_NOT_RUN"
    ):
        return "DERIVATIVES_PENDING"
    if observation_status in {"PARTIAL", "FAILED"} and (
        prediction_ok or settlement_ok
    ):
        return "PARTIAL"
    if observation_ok or prediction_ok or settlement_ok:
        return "PARTIAL"
    return "FAILED"
