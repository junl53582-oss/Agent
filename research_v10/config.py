from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class V10AuditSettings:
    test_years: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024, 2025)
    horizon: int = 5
    rebalance_every: int = 5
    fee_rate: float = 0.0003
    stamp_duty: float = 0.0005
    slippage: float = 0.0005
    maximum_annualized_tracking_error: float = 0.02
    artifact_dir: Path = Path("artifacts/research_v10")

    def ensure_dirs(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

