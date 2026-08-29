from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stockpilot.prediction.config import PredictionSettings


@dataclass(frozen=True)
class V30R1Settings(PredictionSettings):
    version: str = "V30r1"
    artifact_dir: Path = Path("artifacts/prediction_v30r1")
    calibration_years: int = 3

    @property
    def parent_dir(self) -> Path:
        return Path("artifacts/prediction_v30")
