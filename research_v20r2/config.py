from dataclasses import dataclass
from pathlib import Path

from research_v20.config import V20Settings


@dataclass(frozen=True)
class V20R2Settings(V20Settings):
    artifact_dir: Path = Path("artifacts/research_v20r2")
    action_path: Path = Path("data/corporate_actions_v20r2.json")
