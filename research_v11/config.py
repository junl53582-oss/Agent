from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class V11Settings:
    test_years: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024, 2025)
    training_window_years: int = 8
    validation_years: int = 2
    tail_share: float = 0.70
    v10_global_share: float = 0.30
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
    risk_on_exposure: float = 1.00
    weak_exposure: float = 0.80
    risk_off_exposure: float = 0.55
    risk_off_momentum: float = -0.04
    risk_off_breadth: float = 0.42
    weak_momentum: float = 0.00
    weak_breadth: float = 0.48
    artifact_dir: Path = Path("artifacts/research_v11")
    plan_lock_path: Path = Path("artifacts/research_v11/plan.lock.json")

    def ensure_dirs(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

