from dataclasses import dataclass
from pathlib import Path

from research_v22.config import V22Settings


@dataclass(frozen=True)
class V25R1Settings(V22Settings):
    artifact_dir: Path = Path("artifacts/research_v25r1")
    training_windows_years: tuple[int, ...] = (8, 5, 3)
    ridge_share: float = 0.60
    lightgbm_share: float = 0.40
    horizon_5_share: float = 0.30
    horizon_20_share: float = 0.70
    v6_base_share: float = 0.40
    enhanced_share: float = 0.60

