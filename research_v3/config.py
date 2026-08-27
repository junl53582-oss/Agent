from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class V3Settings:
    horizons: tuple[int, ...] = (5, 10, 20)
    rebalance_every: int = 5
    retrain_every: int = 20
    min_train_days: int = 252
    train_window_days: int = 756
    top_n: int = 20
    min_positions: int = 8
    agreement_threshold: float = 2 / 3
    fee_rate: float = 0.0003
    stamp_duty: float = 0.0005
    slippage: float = 0.0005
    industry_cap: float = 0.30
    hold_buffer: int = 5
    artifact_dir: Path = Path("artifacts/research_v3")
    fundamental_path: Path = Path("data/fundamentals_pit.csv")

    def ensure_dirs(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.fundamental_path.parent.mkdir(parents=True, exist_ok=True)
