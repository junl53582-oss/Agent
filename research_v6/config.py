from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PLAN_LOCK_SHA256 = "94EDFC9E05BD30A58A14E7E11A988A1B7FB0D5358E462DF1B20CB23DCA4C0F4D"


@dataclass(frozen=True)
class V6Settings:
    test_years: tuple[int, ...] = (2024, 2025, 2026)
    v5_weight: float = 0.65
    v4_weight: float = 0.20
    sector_rank_weight: float = 0.15
    top_n: int = 30
    min_positions: int = 20
    rebalance_every: int = 5
    fee_rate: float = 0.0003
    stamp_duty: float = 0.0005
    slippage: float = 0.0005
    artifact_dir: Path = Path("artifacts/research_v6")
    plan_lock_path: Path = Path("artifacts/research_v6/plan.lock.json")

    def ensure_dirs(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
