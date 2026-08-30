from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from research_v10.features import V10_FEATURES


@dataclass(frozen=True)
class ChallengerSettings:
    model_id: str = "V31"
    role: str = "RANKING_CHALLENGER"
    status: str = "RESEARCH_ONLY"
    champion: str = "V6"
    dataset_path: Path = Path("artifacts/prediction_v30/cache/eligible_panel.parquet")
    dataset_manifest_path: Path = Path("artifacts/prediction_v30/cache/manifest.json")
    artifact_dir: Path = Path("artifacts/research_v31")
    protocol_path: Path = Path("artifacts/research_v31/protocol.json")
    oos_years: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024, 2025)
    final_oos_years: tuple[int, ...] = (2024, 2025)
    horizons: tuple[int, ...] = (1, 5, 20)
    training_window_years: int = 8
    validation_years: int = 1
    purge_gaps: dict[int, int] = field(default_factory=lambda: {1: 2, 5: 6, 20: 21})
    rebalance_every: dict[int, int] = field(default_factory=lambda: {1: 1, 5: 5, 20: 20})
    top_ks: tuple[int, ...] = (10, 20, 30)
    factor_columns: tuple[str, ...] = tuple(V10_FEATURES)
    selection_horizon: int = 5
    minimum_abs_rank_ic: float = 0.005
    minimum_positive_ratio: float = 0.52
    minimum_year_direction_consistency: float = 0.60
    fdr_q: float = 0.10
    correlation_threshold: float = 0.90
    maximum_selected_factors: int = 20
    minimum_ic_dates: int = 120
    training_row_cap: int = 200_000
    ridge_alpha: float = 10.0
    lightgbm_rounds: int = 80
    random_seed: int = 42
    fee_rate: float = 0.0003
    stamp_duty: float = 0.0005
    slippage: float = 0.0005
    bootstrap_replications: int = 1_000
    bootstrap_block_length: int = 20
    minimum_rank_ic_improvement: float = 0.005
    minimum_positive_years: int = 4
    maximum_drawdown_worsening: float = 0.03
    pre_registered_challenger: str = "lightgbm_lambdarank"

    @property
    def buy_rate(self) -> float:
        return self.fee_rate + self.slippage

    @property
    def sell_rate(self) -> float:
        return self.fee_rate + self.slippage + self.stamp_duty

    def ensure_dirs(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


FORBIDDEN_FEATURE_TOKENS = (
    "future_",
    "label",
    "entry_",
    "exit_",
    "execution_",
    "raw_up",
    "tradable_up",
    "target",
)

MODEL_NAMES = ("v6", "ridge", "lightgbm_regression", "lightgbm_lambdarank")
