from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    horizon: int = 5
    rebalance_every: int = 5
    retrain_every: int = 20
    min_train_days: int = 252
    train_window_days: int = 756
    top_n: int = 5
    initial_cash: float = 1_000_000.0
    fee_rate: float = 0.0003
    stamp_duty: float = 0.0005
    slippage: float = 0.0005
    ridge_alpha: float = 10.0
    model_name: str = "ridge"
    label_mode: str = "neutral"
    weighting: str = "equal"
    hold_buffer: int = 0
    industry_cap: float = 1.0
    evaluation_start: str | None = None
    evaluation_end: str | None = None

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            data_dir=Path(os.getenv("STOCKPILOT_DATA_DIR", "data")),
            artifact_dir=Path(os.getenv("STOCKPILOT_ARTIFACT_DIR", "artifacts")),
            initial_cash=float(os.getenv("STOCKPILOT_INITIAL_CASH", "1000000")),
            fee_rate=float(os.getenv("STOCKPILOT_FEE_RATE", "0.0003")),
            stamp_duty=float(os.getenv("STOCKPILOT_STAMP_DUTY", "0.0005")),
            slippage=float(os.getenv("STOCKPILOT_SLIPPAGE", "0.0005")),
        )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "cache").mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def public_dict(self) -> dict:
        result = asdict(self)
        result["data_dir"] = str(self.data_dir)
        result["artifact_dir"] = str(self.artifact_dir)
        return result
