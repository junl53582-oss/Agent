from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class V15Settings:
    test_years: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024, 2025)
    training_window_years: int = 8
    validation_years: int = 2
    embargo_calendar_days: int = 28
    target_horizons: tuple[int, ...] = (1, 5, 20)
    target_weights: tuple[float, ...] = (0.45, 0.35, 0.20)
    text_n_features: int = 32768
    text_alpha: float = 0.00003
    text_l1_ratio: float = 0.05
    text_max_iter: int = 30
    text_share: float = 0.25
    baseline_share: float = 0.75
    recent_event_lookback_days: int = 28
    recent_event_half_life_days: float = 7.0
    minimum_event_years_cap: int = 4
    confidence_z: float = 1.2815515655446004
    validation_year_floor: float = -0.0025
    active_top_n: int = 30
    technology_top_n: int = 5
    minimum_technology_rows: int = 5000
    minimum_technology_dates: int = 100
    maximum_active_budget: float = 0.15
    maximum_stock_active_weight: float = 0.0075
    maximum_ex_ante_tracking_error: float = 0.06
    holding_bonus: float = 0.02
    rebalance_every: int = 20
    fee_rate: float = 0.0003
    stamp_duty: float = 0.0005
    slippage: float = 0.0005
    artifact_dir: Path = Path("artifacts/research_v15")

    def ensure_dirs(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
