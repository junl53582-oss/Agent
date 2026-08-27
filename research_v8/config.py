from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PLAN_LOCK_SHA256 = "8669522CC4E9B404579307530B149D3BFC1BC8972D125049143500E5A85CE932"


@dataclass(frozen=True)
class V8Settings:
    test_years: tuple[int, ...] = (2024, 2025, 2026)
    training_window_years: int = 3
    ridge_alpha: float = 50.0
    minimum_technology_rows: int = 5000
    minimum_technology_dates: int = 100
    v6_base_share: float = 0.75
    enhanced_global_share: float = 0.25
    technology_specialist_share: float = 0.70
    technology_global_share: float = 0.30
    risk_on_momentum_tilt: float = 0.05
    risk_off_quality_tilt: float = 0.05
    holding_bonus: float = 0.035
    top_n: int = 30
    min_positions: int = 20
    rebalance_every: int = 5
    fee_rate: float = 0.0003
    stamp_duty: float = 0.0005
    slippage: float = 0.0005
    artifact_dir: Path = Path("artifacts/research_v8")
    plan_lock_path: Path = Path("artifacts/research_v8/plan.lock.json")

    def ensure_dirs(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

