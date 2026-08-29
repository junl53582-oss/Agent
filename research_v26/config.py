from dataclasses import dataclass
from pathlib import Path

from research_v22.config import V22Settings


@dataclass(frozen=True)
class V26Settings(V22Settings):
    artifact_dir: Path = Path("artifacts/research_v26")
    training_window_years: int = 8
    lightgbm_share: float = 0.40
    horizon_5_share: float = 0.30
    horizon_20_share: float = 0.70
    enhanced_share: float = 0.60

