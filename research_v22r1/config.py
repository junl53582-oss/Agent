from dataclasses import dataclass
from pathlib import Path

from research_v22.config import V22Settings


@dataclass(frozen=True)
class V22R1Settings(V22Settings):
    artifact_dir: Path = Path("artifacts/research_v22r1")

