from dataclasses import dataclass
from pathlib import Path

from research_v20r2.config import V20R2Settings


@dataclass(frozen=True)
class V22Settings(V20R2Settings):
    artifact_dir: Path = Path("artifacts/research_v22")

