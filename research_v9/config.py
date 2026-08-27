from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class V9Settings:
    test_years: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024, 2025)
    training_window_years: int = 5
    ridge_alpha: float = 75.0
    ridge_share: float = 0.60
    lightgbm_share: float = 0.40
    v6_base_share: float = 0.70
    enhanced_share: float = 0.30
    technology_global_share: float = 0.50
    technology_specialist_share: float = 0.50
    minimum_technology_rows: int = 5000
    minimum_technology_dates: int = 100
    core_share: float = 0.75
    active_share: float = 0.25
    top_n: int = 30
    min_positions: int = 20
    holding_bonus: float = 0.025
    rebalance_every: int = 5
    fee_rate: float = 0.0003
    stamp_duty: float = 0.0005
    slippage: float = 0.0005
    artifact_dir: Path = Path("artifacts/research_v9")
    plan_lock_path: Path = Path("artifacts/research_v9/plan.lock.json")

    def ensure_dirs(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
