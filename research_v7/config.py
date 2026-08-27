from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PLAN_LOCK_SHA256 = "03DE8D8D52A6B112A4E60761A741158932FBA647AD53F62D211A44F97FDFCD3D"


@dataclass(frozen=True)
class V7Settings:
    test_years: tuple[int, ...] = (2024, 2025, 2026)
    horizons: tuple[int, ...] = (5, 20, 60)
    horizon_weights: tuple[float, ...] = (0.50, 0.30, 0.20)
    training_window_years: int = 3
    ridge_alpha: float = 50.0
    global_share: float = 0.60
    expert_share: float = 0.40
    multihorizon_share: float = 0.65
    v6_share: float = 0.35
    uncertainty_penalty: float = 0.10
    holding_bonus: float = 0.05
    top_n: int = 30
    min_positions: int = 20
    rebalance_every: int = 5
    fee_rate: float = 0.0003
    stamp_duty: float = 0.0005
    slippage: float = 0.0005
    artifact_dir: Path = Path("artifacts/research_v7")
    plan_lock_path: Path = Path("artifacts/research_v7/plan.lock.json")

    def ensure_dirs(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
