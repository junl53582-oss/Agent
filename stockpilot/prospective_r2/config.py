from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ReadinessThresholds:
    minimum_observation_dates: int = 20
    minimum_expectation_coverage: float = 0.80
    minimum_label_dates: int = 20
    minimum_label_coverage: float = 0.80
    minimum_label_symbols: int = 240


@dataclass(frozen=True)
class OperationalSettings:
    version: str = "prospective-alpha-v1r2"
    data_root: Path = Path("data/prospective_alpha_v1r2")
    artifact_root: Path = Path("artifacts/prospective_alpha_v1r2")
    calendar_path: Path = Path("artifacts/prospective_alpha_v1r2/trading_calendar_2026.json")
    membership_path: Path = Path("data/universes/000300/history_v10.csv")
    industry_path: Path = Path("data/industry_history_v10.csv")
    announcement_org_metadata_path: Path = Path("data/announcements_pit_v14.csv")
    corporate_action_path: Path = Path("data/corporate_actions_v20r2.json")
    corporate_action_manifest_path: Path = Path("artifacts/research_v20r2/plan.lock.json")
    plan_lock_path: Path = Path("artifacts/prospective_alpha_v1r2/plan.lock.json")
    legacy_barrier_path: Path = Path(
        "data/pit_observations_v2/_LEGACY_LOW_LEVEL_ENTRYPOINT_DISABLED/manifest.json"
    )
    prediction_market_template: str = "data/prediction_forward/v30r1/hfq_union_{date}.csv"
    prediction_ranking_template: str = "artifacts/research_v6/live/predictions/{date}.csv"
    thresholds: ReadinessThresholds = field(default_factory=ReadinessThresholds)

    @property
    def attempts_root(self) -> Path:
        return self.data_root / "_attempts"

    @property
    def observations_root(self) -> Path:
        return self.data_root / "observations"

    @property
    def features_root(self) -> Path:
        return self.data_root / "feature_store"

    @property
    def labels_root(self) -> Path:
        return self.data_root / "mature_labels"

    @property
    def daily_receipts_root(self) -> Path:
        return self.artifact_root / "daily_receipts"
