from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PLAN_LOCK_SHA256 = "1524D6EB86A2991D805A45BCBDFDA4B7499A5274ED80FFB19D7F6C06CEDDD754"


@dataclass(frozen=True)
class V5Settings:
    test_years: tuple[int, ...] = (2024, 2025, 2026)
    training_window_years: int = 3
    ridge_alpha: float = 50.0
    minimum_expert_rows: int = 5000
    minimum_expert_dates: int = 100
    rebalance_every: int = 5
    top_n: int = 20
    min_positions: int = 8
    hold_buffer: int = 5
    industry_cap: float = 0.30
    fee_rate: float = 0.0003
    stamp_duty: float = 0.0005
    slippage: float = 0.0005
    artifact_dir: Path = Path("artifacts/research_v5")
    plan_lock_path: Path = Path("artifacts/research_v5/plan.lock.json")

    def ensure_dirs(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
