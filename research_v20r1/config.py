from dataclasses import dataclass
from pathlib import Path

from research_v20.config import V20Settings


@dataclass(frozen=True)
class V20R1Settings(V20Settings):
    artifact_dir: Path = Path("artifacts/research_v20r1")
