from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PredictionSettings:
    version: str = "V30"
    market_path: Path = Path("data/market_history_v10_hfq.csv")
    membership_path: Path = Path("data/universes/000300/history_v10.csv")
    fundamental_path: Path = Path("data/fundamentals_pit_v10_extended.csv")
    industry_path: Path = Path("data/industry_history_v10.csv")
    names_path: Path = Path("data/stock_names.csv")
    artifact_dir: Path = Path("artifacts/prediction_v30")
    oos_years: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023, 2024, 2025)
    horizons: tuple[int, ...] = (1, 5, 20)
    return_horizons: tuple[int, ...] = (5, 20)
    training_window_years: int = 8
    purge_gaps: dict[int, int] = field(default_factory=lambda: {1: 2, 5: 6, 20: 21})
    direction_thresholds: dict[int, float] = field(default_factory=lambda: {1: 0.0021, 5: 0.0021, 20: 0.0021})
    retrain_every_trading_days: int = 20
    calibration_years: int = 1
    ridge_alpha: float = 10.0
    logistic_max_iter: int = 30
    training_row_cap: int = 400_000
    auc_minimum: float = 0.52
    calibration_error_maximum: float = 0.05
    minimum_positive_skill_years: int = 4
    minimum_regimes_passed: int = 4
    minimum_sectors_passed: int = 4
    low_confidence_upper: float = 0.40
    medium_confidence_upper: float = 0.70
    psi_warning: float = 0.10
    psi_severe: float = 0.25
    zscore_warning: float = 1.0
    zscore_severe: float = 2.0
    candidate_ranking_weight: float = 0.40
    candidate_probability_weight: float = 0.30
    candidate_return_weight: float = 0.30
    candidate_risk_penalty: float = 0.10
    execution_authorized: bool = False

    @property
    def validation_dir(self) -> Path:
        return self.artifact_dir / "validation"

    @property
    def models_dir(self) -> Path:
        return self.artifact_dir / "models"

    @property
    def prediction_dir(self) -> Path:
        return self.artifact_dir / "live" / "predictions"

    @property
    def certification_dir(self) -> Path:
        return self.artifact_dir / "certification"

    def ensure_dirs(self) -> None:
        for path in (self.validation_dir, self.models_dir, self.prediction_dir, self.certification_dir):
            path.mkdir(parents=True, exist_ok=True)
