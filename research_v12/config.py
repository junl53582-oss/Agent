from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class V12Settings:
    test_years: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024, 2025)
    training_window_years: int = 8
    validation_years: int = 2
    embargo_calendar_days: int = 28
    portfolio_rank_share: float = 0.75
    v10_global_share: float = 0.25
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
    liquidity_impact_max: float = 0.003
    risk_target_annual_volatility: float = 0.12
    minimum_equity_exposure: float = 0.35
    maximum_equity_exposure: float = 1.00
    artifact_dir: Path = Path("artifacts/research_v12")
    plan_lock_path: Path = Path("artifacts/research_v12/plan.lock.json")

    def ensure_dirs(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

