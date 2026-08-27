from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PLAN_LOCK_SHA256 = "FB4907EDDAC2D0709053130430A242DB86132B7A44FCA888D03FB047D3BBB29E"


@dataclass(frozen=True)
class V4Settings:
    test_years: tuple[int, ...] = (2024, 2025, 2026)
    training_year_window: int = 3
    minimum_training_years: int = 2
    minimum_ic_days_per_year: int = 60
    minimum_absolute_mean_rank_ic: float = 0.005
    minimum_direction_consistency: float = 0.60
    horizon_days: int = 5
    rebalance_every: int = 5
    top_n: int = 20
    min_positions: int = 8
    hold_buffer: int = 5
    industry_cap: float = 0.30
    fee_rate: float = 0.0003
    stamp_duty: float = 0.0005
    slippage: float = 0.0005
    artifact_dir: Path = Path("artifacts/research_v4")
    plan_lock_path: Path = Path("artifacts/research_v4/plan.lock.json")

    def ensure_dirs(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
