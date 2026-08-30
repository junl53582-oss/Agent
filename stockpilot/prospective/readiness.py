from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ReadinessStatus:
    source_observation_count: int
    pit_observation_count: int
    mature_1d_count: int
    mature_5d_count: int
    mature_20d_count: int
    minimum_observations: int
    model_training_ready: bool
    factor_validation_ready: bool
    replacement_evaluation_ready: bool
    production_prediction_ready: bool
    execution_authorized: bool

    def to_dict(self) -> dict:
        return asdict(self)


def derive_readiness(
    observations: list[dict], labels: list[dict], *, minimum_observations: int = 20
) -> ReadinessStatus:
    qualifying = {
        item["target_date"]
        for item in observations
        if item.get("qualifying_trading_observation")
        and item.get("sources", {}).get("earnings_expectations", {}).get("source_status") == "SUCCESS"
    }
    mature = {
        horizon: {
            item["prediction_date"]
            for item in labels
            if item.get("horizon") == horizon and item.get("status") == "SETTLED"
        }
        for horizon in (1, 5, 20)
    }
    factor_ready = len(qualifying) >= minimum_observations and all(
        len(mature[horizon]) >= minimum_observations for horizon in mature
    )
    # This infrastructure has no model-training or production promotion path.
    return ReadinessStatus(
        source_observation_count=len({item["observation_id"] for item in observations}),
        pit_observation_count=len(qualifying),
        mature_1d_count=len(mature[1]),
        mature_5d_count=len(mature[5]),
        mature_20d_count=len(mature[20]),
        minimum_observations=minimum_observations,
        model_training_ready=factor_ready,
        factor_validation_ready=factor_ready,
        replacement_evaluation_ready=False,
        production_prediction_ready=False,
        execution_authorized=False,
    )
