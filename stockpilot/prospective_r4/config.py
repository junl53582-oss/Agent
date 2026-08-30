from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from pathlib import Path

from stockpilot.prospective_r3.config import OperationalSettings as V1R3Settings


@dataclass(frozen=True)
class OperationalSettings(V1R3Settings):
    version: str = "prospective-alpha-v1r4"
    data_root: Path = Path("data/prospective_alpha_v1r4")
    artifact_root: Path = Path("artifacts/prospective_alpha_v1r4")
    settlement_manifest_path: Path = Path(
        "artifacts/prospective_alpha_v1r4/settlement_source_manifest.json"
    )
    plan_lock_path: Path = Path("artifacts/prospective_alpha_v1r4/plan.lock.json")
    v1r3_lock_path: Path = Path("artifacts/prospective_alpha_v1r3/plan.lock.json")
    v1r3_barrier_path: Path = Path(
        "data/prospective_alpha_v1r3/observations/_V1R4_ACTIVE/observation.json"
    )
    earliest_daily_run_time: time = time(18, 30)
    minimum_forward_market_symbols: int = 300
    minimum_v6_ranking_symbols: int = 240
    minimum_v6_ranking_coverage: float = 0.80

    @property
    def input_evidence_root(self) -> Path:
        return self.data_root / "prediction_input_evidence"

    @property
    def operational_receipts_root(self) -> Path:
        return self.data_root / "operational_receipts"
