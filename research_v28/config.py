from dataclasses import dataclass
from pathlib import Path

from research_v22.config import V22Settings


@dataclass(frozen=True)
class V28Settings(V22Settings):
    artifact_dir: Path = Path("artifacts/research_v28")
    training_window_years: int = 8
    validation_years: int = 2
    tail_quantile: float = 0.80
    direction_share: float = 0.50
    tail_share: float = 0.50
    lightgbm_share: float = 0.40
    enhanced_share: float = 0.60
    horizon_5_share: float = 0.30
    horizon_20_share: float = 0.70
    validation_rebalance_every: int = 20
    active_drawdown_floor: float = -0.10
    maximum_tracking_error: float = 0.06

