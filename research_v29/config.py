from dataclasses import dataclass
from pathlib import Path

from research_v28.config import V28Settings


@dataclass(frozen=True)
class V29Settings(V28Settings):
    artifact_dir: Path = Path("artifacts/research_v29")
