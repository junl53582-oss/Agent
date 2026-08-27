from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class V10Settings:
    test_years: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024, 2025)
    training_window_years: int = 8
    validation_years: int = 2
    ridge_alpha: float = 100.0
    ridge_share: float = 0.60
    lightgbm_share: float = 0.40
    horizon_5_share: float = 0.30
    horizon_20_share: float = 0.70
    v6_base_share: float = 0.40
    enhanced_share: float = 0.60
    technology_global_share: float = 0.60
    technology_specialist_share: float = 0.40
    minimum_technology_rows: int = 5000
    minimum_technology_dates: int = 100
    validation_ic_full_confidence: float = 0.03
    maximum_active_budget: float = 0.15
    maximum_stock_active_weight: float = 0.0075
    maximum_ex_ante_tracking_error: float = 0.06
    active_top_n: int = 30
    holding_bonus: float = 0.02
    rebalance_every: int = 20
    fee_rate: float = 0.0003
    stamp_duty: float = 0.0005
    slippage: float = 0.0005
    artifact_dir: Path = Path("artifacts/research_v10")
    plan_lock_path: Path = Path("artifacts/research_v10/plan.lock.json")

    def ensure_dirs(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

