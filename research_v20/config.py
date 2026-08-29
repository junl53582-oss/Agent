from dataclasses import dataclass
from pathlib import Path

from research_v16.config import V16Settings


@dataclass(frozen=True)
class V20Settings(V16Settings):
    # Inherit the complete scoring contract, including baseline_share/text_share.
    artifact_dir: Path = Path("artifacts/research_v20")
    timing_window_days: int = 20
    timing_threshold: float = 0.0
    weight_bull: float = 0.50
    weight_neutral: float = 0.65
    weight_bear: float = 0.80
    bull_threshold: float = 0.02
    bear_threshold: float = -0.02
    minimum_market_coverage: float = 0.95
