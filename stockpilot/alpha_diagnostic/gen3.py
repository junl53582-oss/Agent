from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stockpilot.alpha_diagnostic.gen2 import BASELINE_SHA as ARCHITECTURE_SHA
from stockpilot.research_challenger.config import ChallengerSettings
from stockpilot.research_challenger.gen02 import _selected_factors
from stockpilot.research_challenger.gen02_correctness import (
    _load_verified_price_book,
    evaluate_stateful_portfolio_policy,
    load_maturity_safe_development_dataset,
    summarize_stateful_portfolio,
)
from stockpilot.research_challenger.gen02_portfolio import PortfolioPolicy
from stockpilot.research_challenger.metrics import (
    daily_rank_metrics,
    moving_block_bootstrap_delta,
    quantile_returns,
    summarize_ic,
)
from stockpilot.research_challenger.models import (
    RidgeModel,
    TrainOnlyPreprocessor,
    deterministic_full_date_sample,
)
from stockpilot.research_challenger.split import build_fold, fold_receipt

ARTIFACT_DIR = Path("artifacts/research_challenger/gen03_alpha_improvement")
DIAGNOSTIC_DIR = Path("artifacts/research_challenger/gen03_alpha_diagnostic")
MERGED_DIAGNOSTIC_SHA = "e91a7b2cd97b088bb613977867417b3fa3aaa2d1"
GEN2_MODEL_ID = "GEN2-LGBM-20D-SECTOR-BALANCED-TOP20"

STABLE_CORE = (
    "volatility_60_rank",
    "liquidity",
    "revenue_growth_change_rank",
    "profit_growth_change_rank",
    "gross_margin_change_rank",
    "momentum",
    "gross_margin_yoy_change_rank",
    "deducted_profit_growth_change_rank",
    "intraday_strength_20_rank",
    "low_volatility",
    "industry_momentum",
)

DEAD_UNSTABLE = (
    "roe_rank",
    "momentum_120_rank",
    "operating_cash_margin_change_rank",
    "volume_attention",
    "debt_ratio_change_rank",
    "book_to_price_rank",
    "fundamental_coverage",
    "gross_margin_rank",
    "earnings_yield_rank",
    "debt_ratio_rank",
    "technology_growth_rank",
    "technology_valuation_rank",
    "technology_momentum_rank",
    "price_position_120_rank",
    "overnight_gap_20_rank",
    "drawdown_120_rank",
    "downside_volatility_60_rank",
    "cash_ratio_rank",
    "fixed_asset_turnover_rank",
    "inventory_turnover_rank",
    "receivables_turnover_rank",
    "technology_quality_rank",
    "staff_average_revenue_rank",
    "staff_average_profit_rank",
    "interest_debt_ratio_rank",
    "receivables_turnover_change_rank",
    "operating_cycle_rank",
    "fixed_asset_turnover_change_rank",
    "inventory_turnover_change_rank",
    "interest_debt_ratio_change_rank",
    "staff_average_revenue_change_rank",
    "staff_average_profit_change_rank",
    "fcff_back_change_rank",
)

REDUNDANT_REMOVALS = (
    "downside_volatility_60_rank",
    "revenue_growth_rank",
    "growth",
)

NEW_FEATURES = (
    "gen3_vol_adjusted_momentum_60",
    "gen3_downside_asymmetry",
    "gen3_sector_relative_momentum_60",
    "gen3_extreme_reversal_5",
    "gen3_liquidity_shock",
    "gen3_trend_consistency",
)

REGIME_INTERACTIONS = (
    "gen3_risk_off_x_vol_adjusted_momentum",
    "gen3_risk_off_x_liquidity_shock",
    "gen3_large_cap_x_sector_relative_momentum",
    "gen3_large_cap_x_downside_asymmetry",
    "gen3_low_vol_x_extreme_reversal",
    "gen3_technology_x_trend_consistency",
)


@dataclass(frozen=True)
class Gen3Settings:
    artifact_dir: Path = ARTIFACT_DIR
    years: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024, 2025)
    research_development_years: tuple[int, ...] = (2020, 2021, 2022, 2023)
    comparison_years: tuple[int, ...] = (2024, 2025)
    horizon: int = 20
    random_seed: int = 42
    bootstrap_replications: int = 1_000
    bootstrap_block_length: int = 20
    top_ks: tuple[int, ...] = (10, 20, 30, 40, 50)
    cost_bps: tuple[int, ...] = (0, 10, 20, 30, 50)
    quantiles: int = 5
    maximum_research_candidates: int = 2


@dataclass(frozen=True)
class LGBMConfig:
    learning_rate: float
    num_leaves: int
    max_depth: int
    min_data_in_leaf: int
    feature_fraction: float
    bagging_fraction: float
    bagging_freq: int
    lambda_l1: float
    lambda_l2: float
    rounds: int
    early_stopping_rounds: int = 0


BASELINE_CONFIG = LGBMConfig(0.04, 15, 5, 200, 0.8, 1.0, 0, 1.0, 5.0, 80)
SHALLOW_CONFIG = LGBMConfig(0.035, 7, 3, 400, 0.75, 0.8, 1, 2.0, 10.0, 60)
STRONG_CONFIG = LGBMConfig(0.03, 7, 3, 800, 0.70, 0.75, 1, 5.0, 20.0, 50)
EARLY_STOP_CONFIG = LGBMConfig(0.035, 7, 3, 500, 0.75, 0.8, 1, 3.0, 12.0, 120, 15)


EXPERIMENTS = (
    {
        "id": "A_GEN2_BASELINE",
        "score": "gen2_baseline",
        "family": "baseline",
        "model": "lightgbm",
        "feature_policy": "yearly_frozen_gen2",
        "config": "baseline",
        "hypothesis": "Exact Gen2 reproduction anchors all paired comparisons.",
        "expected_mechanism": "None; frozen comparative baseline.",
    },
    {
        "id": "B1_SHALLOW_REG",
        "score": "lgbm_shallow",
        "family": "regularization",
        "model": "lightgbm",
        "feature_policy": "yearly_frozen_gen2",
        "config": "shallow",
        "hypothesis": "Lower tree capacity and row/feature subsampling reduce overfit.",
        "expected_mechanism": "Lower train/OOS gap and fold variance.",
    },
    {
        "id": "B2_STRONG_REG",
        "score": "lgbm_strong_reg",
        "family": "regularization",
        "model": "lightgbm",
        "feature_policy": "yearly_frozen_gen2",
        "config": "strong",
        "hypothesis": "Stronger shrinkage improves weak-year robustness.",
        "expected_mechanism": "Higher worst-year IC and ICIR at lower complexity.",
    },
    {
        "id": "B3_EARLY_STOP_REG",
        "score": "lgbm_early_stop",
        "family": "regularization",
        "model": "lightgbm",
        "feature_policy": "yearly_frozen_gen2",
        "config": "early_stop",
        "hypothesis": "Past-only validation can choose boosting length without OOS tuning.",
        "expected_mechanism": "Adaptive rounds reduce unnecessary boosting.",
    },
    {
        "id": "C1_FULL61_REG",
        "score": "lgbm_full61",
        "family": "feature_selection",
        "model": "lightgbm",
        "feature_policy": "full61",
        "config": "shallow",
        "hypothesis": "Regularization may safely use the complete frozen information set.",
        "expected_mechanism": "Recover conditional features without deep trees.",
    },
    {
        "id": "C2_STABLE_CORE_REG",
        "score": "lgbm_stable_core",
        "family": "feature_selection",
        "model": "lightgbm",
        "feature_policy": "stable_core",
        "config": "shallow",
        "hypothesis": "Stable high-activity features improve sample efficiency.",
        "expected_mechanism": "Lower variance with preserved risk/liquidity/price alpha.",
    },
    {
        "id": "C3_DE_REDUNDANT_REG",
        "score": "lgbm_dedup",
        "family": "feature_selection",
        "model": "lightgbm",
        "feature_policy": "de_redundant",
        "config": "shallow",
        "hypothesis": "Removing one representative from each >0.90 cluster reduces noise.",
        "expected_mechanism": "Lower correlation burden without losing information sources.",
    },
    {
        "id": "C4_REMOVE_DEAD_REG",
        "score": "lgbm_dead_removed",
        "family": "feature_selection",
        "model": "lightgbm",
        "feature_policy": "remove_dead_unstable",
        "config": "shallow",
        "hypothesis": "Features inactive in all folds and weak in diagnostics add variance.",
        "expected_mechanism": "Reduce dimensionality while retaining selected evidence.",
    },
    {
        "id": "E_NEW_INDEPENDENT_FEATURES",
        "score": "lgbm_new_features",
        "family": "new_features",
        "model": "lightgbm",
        "feature_policy": "stable_core_plus_new",
        "config": "shallow",
        "hypothesis": "Six PIT-safe ratios/residual ranks add independent risk/liquidity/price information.",
        "expected_mechanism": "Improve residual IC and quantile ordering.",
    },
    {
        "id": "F_REGIME_INTERACTIONS",
        "score": "lgbm_regime_interactions",
        "family": "regime_conditioning",
        "model": "lightgbm",
        "feature_policy": "stable_core_new_interactions",
        "config": "shallow",
        "hypothesis": "Simple PIT interactions address known weak slices without separate models.",
        "expected_mechanism": "Improve risk-off/large-cap/technology/low-vol IC.",
    },
    {
        "id": "G1_RIDGE_GEN2",
        "score": "ridge_gen2",
        "family": "linear",
        "model": "ridge",
        "feature_policy": "yearly_frozen_gen2",
        "config": "ridge_alpha_10",
        "hypothesis": "A linear model may generalize as well as trees with a smaller gap.",
        "expected_mechanism": "Stable additive signal and low variance.",
    },
    {
        "id": "G2_RIDGE_STABLE_CORE",
        "score": "ridge_stable_core",
        "family": "linear",
        "model": "ridge",
        "feature_policy": "stable_core",
        "config": "ridge_alpha_10",
        "hypothesis": "Stable-core linear structure may improve worst-year behavior.",
        "expected_mechanism": "Reduced feature and functional complexity.",
    },
)

DERIVED_EXPERIMENTS = (
    {
        "id": "H1_ENSEMBLE_50_50",
        "score": "ensemble_50_50",
        "family": "ensemble",
        "hypothesis": "Fixed rank averaging diversifies tree and linear errors.",
    },
    {
        "id": "H2_ENSEMBLE_PAST_IC",
        "score": "ensemble_past_ic",
        "family": "ensemble",
        "hypothesis": "Past validation IC weights adapt without reading test labels.",
    },
    {
        "id": "H3_ENSEMBLE_REGIME",
        "score": "ensemble_regime",
        "family": "ensemble",
        "hypothesis": "Shrunk past-regime weights improve weak regimes without separate models.",
    },
    {
        "id": "I_RESIDUALIZED_SCORE",
        "score": "residualized_shallow",
        "family": "residualization",
        "hypothesis": "Current-date exposure neutralization preserves more independent alpha.",
    },
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _cross_rank(data: pd.DataFrame, values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return numeric.groupby(data["date"]).rank(pct=True, method="average").sub(0.5).fillna(0.0)


def add_gen3_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add only decision-time transforms of existing trailing/PIT inputs."""

    result = data.copy()
    eps = 1e-6
    volatility_60 = pd.to_numeric(result["volatility_60"], errors="coerce").abs().clip(lower=eps)
    volatility_20 = pd.to_numeric(result["volatility_20"], errors="coerce").abs().clip(lower=eps)
    momentum_60 = pd.to_numeric(result["momentum_60"], errors="coerce")
    downside_60 = pd.to_numeric(result["downside_volatility_60"], errors="coerce").abs()
    ret_5 = pd.to_numeric(result["ret_5"], errors="coerce")
    ret_20 = pd.to_numeric(result["ret_20"], errors="coerce")
    volume_ratio = pd.to_numeric(result["volume_ratio_20"], errors="coerce")
    volume_trend = pd.to_numeric(result["volume_trend_60"], errors="coerce")

    result[NEW_FEATURES[0]] = _cross_rank(result, momentum_60 / volatility_60)
    result[NEW_FEATURES[1]] = _cross_rank(result, -(downside_60 / volatility_60))
    result[NEW_FEATURES[2]] = (
        momentum_60.groupby([result["date"], result["industry"].fillna("UNKNOWN")])
        .rank(pct=True, method="average")
        .sub(0.5)
        .fillna(0.0)
    )
    result[NEW_FEATURES[3]] = _cross_rank(result, -ret_5 / volatility_20)
    result[NEW_FEATURES[4]] = _cross_rank(result, volume_ratio / (1 + volume_trend).clip(lower=0.1))
    direction = np.sign(ret_5) + np.sign(ret_20) + np.sign(momentum_60)
    result[NEW_FEATURES[5]] = _cross_rank(result, direction * ret_20.abs() / volatility_20)

    risk_off = result["regime"].astype(str).eq("risk_off").astype(float)
    large_cap = pd.to_numeric(result["benchmark_weight_rank"], errors="coerce").clip(lower=0)
    low_vol = (
        result.groupby("date")["volatility_20"].rank(pct=True, method="average").le(1 / 3)
    ).astype(float)
    technology = result["broad_sector"].astype(str).eq("technology").astype(float)
    result[REGIME_INTERACTIONS[0]] = risk_off * result[NEW_FEATURES[0]]
    result[REGIME_INTERACTIONS[1]] = risk_off * result[NEW_FEATURES[4]]
    result[REGIME_INTERACTIONS[2]] = large_cap * result[NEW_FEATURES[2]]
    result[REGIME_INTERACTIONS[3]] = large_cap * result[NEW_FEATURES[1]]
    result[REGIME_INTERACTIONS[4]] = low_vol * result[NEW_FEATURES[3]]
    result[REGIME_INTERACTIONS[5]] = technology * result[NEW_FEATURES[5]]
    result[[*NEW_FEATURES, *REGIME_INTERACTIONS]] = (
        result[[*NEW_FEATURES, *REGIME_INTERACTIONS]].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    return result


def feature_registry() -> list[dict]:
    common = {
        "pit_available": True,
        "effective_timestamp_semantics": "computed only from inputs available at decision date",
        "missing_policy": "replace non-finite result with cross-sectional neutral 0.0",
        "diagnostic_evidence_source": "PR #5 Gen2 alpha diagnostic",
        "frozen_contract_modified": False,
    }
    definitions = [
        (
            NEW_FEATURES[0],
            "price_behavior",
            "rank(momentum_60 / abs(volatility_60))",
            "60 sessions",
            ["momentum_60", "volatility_60"],
            "Separate trend from volatility loading.",
        ),
        (
            NEW_FEATURES[1],
            "risk",
            "rank(-downside_volatility_60 / abs(volatility_60))",
            "60 sessions",
            ["downside_volatility_60", "volatility_60"],
            "Measure downside asymmetry beyond total volatility.",
        ),
        (
            NEW_FEATURES[2],
            "price_behavior",
            "within-industry rank(momentum_60)",
            "60 sessions",
            ["momentum_60", "industry"],
            "Remove broad sector momentum contamination.",
        ),
        (
            NEW_FEATURES[3],
            "price_behavior",
            "rank(-ret_5 / abs(volatility_20))",
            "5/20 sessions",
            ["ret_5", "volatility_20"],
            "Test reversal after volatility-scaled extremes.",
        ),
        (
            NEW_FEATURES[4],
            "liquidity",
            "rank(volume_ratio_20 / max(1+volume_trend_60,0.1))",
            "20/60 sessions",
            ["volume_ratio_20", "volume_trend_60"],
            "Distinguish immediate volume shock from medium trend.",
        ),
        (
            NEW_FEATURES[5],
            "price_behavior",
            "rank((sign(ret_5)+sign(ret_20)+sign(momentum_60))*abs(ret_20)/volatility_20)",
            "5/20/60 sessions",
            ["ret_5", "ret_20", "momentum_60", "volatility_20"],
            "Reward direction consistency rather than raw momentum.",
        ),
    ]
    rows = [
        {
            "feature_name": name,
            "category": category,
            "formula": formula,
            "lookback": lookback,
            "inputs": inputs,
            "motivation": motivation,
            **common,
        }
        for name, category, formula, lookback, inputs, motivation in definitions
    ]
    interaction_inputs = [
        (REGIME_INTERACTIONS[0], ["regime", NEW_FEATURES[0]], "risk-off trend interaction"),
        (REGIME_INTERACTIONS[1], ["regime", NEW_FEATURES[4]], "risk-off liquidity interaction"),
        (
            REGIME_INTERACTIONS[2],
            ["benchmark_weight_rank", NEW_FEATURES[2]],
            "large-cap residual momentum interaction",
        ),
        (
            REGIME_INTERACTIONS[3],
            ["benchmark_weight_rank", NEW_FEATURES[1]],
            "large-cap downside-risk interaction",
        ),
        (
            REGIME_INTERACTIONS[4],
            ["volatility_20", NEW_FEATURES[3]],
            "low-vol reversal interaction",
        ),
        (
            REGIME_INTERACTIONS[5],
            ["broad_sector", NEW_FEATURES[5]],
            "technology trend-consistency interaction",
        ),
    ]
    rows.extend(
        {
            "feature_name": name,
            "category": "regime_interaction",
            "formula": description,
            "lookback": "inherits source lookback; regime/bucket observed at decision date",
            "inputs": inputs,
            "motivation": "Address a pre-registered weak slice without a separate model.",
            **common,
        }
        for name, inputs, description in interaction_inputs
    )
    return rows


def _feature_set(policy: str, year: int, base: ChallengerSettings) -> tuple[str, ...]:
    if policy == "yearly_frozen_gen2":
        return tuple(_selected_factors()[year])
    if policy == "full61":
        return tuple(base.factor_columns)
    if policy == "stable_core":
        return STABLE_CORE
    if policy == "de_redundant":
        return tuple(
            feature for feature in base.factor_columns if feature not in REDUNDANT_REMOVALS
        )
    if policy == "remove_dead_unstable":
        return tuple(feature for feature in base.factor_columns if feature not in DEAD_UNSTABLE)
    if policy == "stable_core_plus_new":
        return (*STABLE_CORE, *NEW_FEATURES)
    if policy == "stable_core_new_interactions":
        return (*STABLE_CORE, *NEW_FEATURES, *REGIME_INTERACTIONS)
    raise ValueError(f"unknown feature policy: {policy}")


def _config(name: str) -> LGBMConfig:
    return {
        "baseline": BASELINE_CONFIG,
        "shallow": SHALLOW_CONFIG,
        "strong": STRONG_CONFIG,
        "early_stop": EARLY_STOP_CONFIG,
    }[name]


def _lgb_params(config: LGBMConfig, seed: int) -> dict:
    return {
        "objective": "regression_l1",
        "metric": "l1",
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "max_depth": config.max_depth,
        "min_data_in_leaf": config.min_data_in_leaf,
        "feature_fraction": config.feature_fraction,
        "bagging_fraction": config.bagging_fraction,
        "bagging_freq": config.bagging_freq,
        "lambda_l1": config.lambda_l1,
        "lambda_l2": config.lambda_l2,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "num_threads": 4,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }


def _fit_lgbm(
    x: np.ndarray,
    y: np.ndarray,
    config: LGBMConfig,
    seed: int,
    *,
    valid: tuple[np.ndarray, np.ndarray] | None = None,
    rounds: int | None = None,
):
    import lightgbm as lgb

    train_set = lgb.Dataset(x, label=y, free_raw_data=True)
    callbacks = []
    valid_sets = None
    if valid is not None and config.early_stopping_rounds:
        valid_sets = [lgb.Dataset(valid[0], label=valid[1], reference=train_set)]
        callbacks = [lgb.early_stopping(config.early_stopping_rounds, verbose=False)]
    model = lgb.train(
        _lgb_params(config, seed),
        train_set,
        num_boost_round=rounds or config.rounds,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )
    return model


def _model_hash(model) -> str:
    return hashlib.sha256(model.model_to_string().encode("utf-8")).hexdigest()


def _residualize_scores(frame: pd.DataFrame, score: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    controls = ["benchmark_weight_rank", "volatility_60_rank", "momentum", "liquidity"]
    for indexes in frame.groupby("date", sort=False).groups.values():
        current = frame.loc[indexes]
        y = pd.to_numeric(score.loc[indexes], errors="coerce")
        valid = y.notna() & current[controls].notna().all(axis=1)
        if valid.sum() < 40:
            continue
        idx = current.index[valid]
        sectors = pd.get_dummies(current.loc[idx, "broad_sector"], drop_first=True, dtype=float)
        design = np.column_stack(
            [np.ones(len(idx)), current.loc[idx, controls].to_numpy(float), sectors.to_numpy(float)]
        )
        values = y.loc[idx].to_numpy(float)
        result.loc[idx] = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
    return result


def _rank_average(
    frame: pd.DataFrame, left: str, right: str, weight: float | pd.Series
) -> pd.Series:
    left_rank = frame.groupby("date")[left].rank(pct=True)
    right_rank = frame.groupby("date")[right].rank(pct=True)
    return weight * left_rank + (1 - weight) * right_rank


def _validation_weights(validation: pd.DataFrame) -> tuple[float, dict[str, float]]:
    lgb_daily = daily_rank_metrics(validation, "validation_lgbm", "future_return_20d")
    ridge_daily = daily_rank_metrics(validation, "validation_ridge", "future_return_20d")
    lgb_ic = max(0.0, float(lgb_daily["rank_ic"].mean()))
    ridge_ic = max(0.0, float(ridge_daily["rank_ic"].mean()))
    global_weight = lgb_ic / (lgb_ic + ridge_ic) if lgb_ic + ridge_ic > 0 else 0.5
    global_weight = float(np.clip(global_weight, 0.2, 0.8))
    regime_weights = {}
    date_regime = validation.groupby("date")["regime"].first()
    for regime in ("risk_on", "risk_off", "neutral"):
        dates = date_regime[date_regime.eq(regime)].index
        lgb_value = float(lgb_daily[lgb_daily["date"].isin(dates)]["rank_ic"].mean())
        ridge_value = float(ridge_daily[ridge_daily["date"].isin(dates)]["rank_ic"].mean())
        raw = (
            max(0.0, lgb_value) / (max(0.0, lgb_value) + max(0.0, ridge_value))
            if max(0.0, lgb_value) + max(0.0, ridge_value) > 0
            else global_weight
        )
        shrink = len(dates) / (len(dates) + 60)
        regime_weights[regime] = float(
            np.clip(shrink * raw + (1 - shrink) * global_weight, 0.2, 0.8)
        )
    return global_weight, regime_weights


def freeze_protocol(settings: Gen3Settings | None = None) -> dict:
    settings = settings or Gen3Settings()
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((DIAGNOSTIC_DIR / "diagnostic_summary.json").read_text(encoding="utf-8"))
    protocol = {
        "protocol_id": "GEN3_ALPHA_IMPROVEMENT_EXPERIMENT",
        "classification": "DEVELOPMENT_COMPARATIVE_RESEARCH_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": _git("rev-parse", "HEAD"),
        "architecture_sha": ARCHITECTURE_SHA,
        "merged_diagnostic_sha": MERGED_DIAGNOSTIC_SHA,
        "diagnostic_summary_sha256": _sha256(DIAGNOSTIC_DIR / "diagnostic_summary.json"),
        "settings": asdict(settings),
        "nested_discipline": {
            "research_development_years": list(settings.research_development_years),
            "comparison_years": list(settings.comparison_years),
            "2025_untouched": False,
            "manual_retuning_against_2025_after_protocol": False,
            "untouched_holdout_available": False,
        },
        "evaluation": {
            "target": "cross-sectional rank of 20D T+1-open to T+21-open return",
            "training_window_years": 8,
            "validation_years": 1,
            "purge_gap_trading_days": 21,
            "random_split": False,
            "future_standardization": False,
            "selection_semantics": "all families fixed before full evaluation run",
        },
        "success_gates": {
            "rank_ic_baseline": baseline["overall"]["rank_ic_mean"],
            "icir_baseline": baseline["overall"]["icir"],
            "residual_ic_floor": baseline["residual_alpha"]["mean_residual_rank_ic"] - 0.002,
            "paired_rank_ic_bootstrap_lower_gt": 0.0,
            "20bps_research_proxy_alpha_gte": 0.0,
            "train_oos_gap_not_worse": True,
            "quantile_monotonicity_better": True,
        },
        "forbidden": [
            "Gen2 modification",
            "007-012 modification",
            "DAILY PIT modification",
            "sandbox modification",
            "execution/settlement modification",
            "Gen3 activation",
            "champion promotion",
            "live trading authorization",
        ],
    }
    _write_json(settings.artifact_dir / "research_protocol.json", protocol)
    _write_json(
        settings.artifact_dir / "experiment_registry.json", [*EXPERIMENTS, *DERIVED_EXPERIMENTS]
    )
    _write_json(settings.artifact_dir / "gen3_feature_registry.json", feature_registry())
    return protocol


def _score_summary(daily: pd.DataFrame) -> dict:
    if daily.empty or "rank_ic" not in daily:
        return {
            "dates": 0,
            "rank_ic_mean": np.nan,
            "rank_ic_median": np.nan,
            "rank_ic_std": np.nan,
            "icir": np.nan,
            "positive_ic_ratio": np.nan,
            "pearson_ic_mean": np.nan,
            "rank_ic_t_stat": np.nan,
        }
    summary = summarize_ic(daily)
    values = pd.to_numeric(daily.get("rank_ic"), errors="coerce").dropna()
    return {
        "dates": summary["dates"],
        "rank_ic_mean": summary["mean_rank_ic"],
        "rank_ic_median": float(values.median()) if len(values) else np.nan,
        "rank_ic_std": summary["rank_ic_std"],
        "icir": summary["rank_ic_ir"],
        "positive_ic_ratio": summary["positive_rank_ic_ratio"],
        "pearson_ic_mean": summary["mean_pearson_ic"],
        "rank_ic_t_stat": (
            float(values.mean() / values.std(ddof=1) * math.sqrt(len(values)))
            if len(values) > 1 and values.std(ddof=1) > 0
            else np.nan
        ),
    }


def _fit_one_experiment(
    experiment: dict,
    data: pd.DataFrame,
    fold,
    base: ChallengerSettings,
    year: int,
) -> tuple[np.ndarray, dict, dict]:
    features = _feature_set(experiment["feature_policy"], year, base)
    target = "return_rank_20d"
    refit = data.loc[fold.refit_index].copy()
    finite = pd.to_numeric(refit[target], errors="coerce")
    refit = refit[finite.notna() & np.isfinite(finite)].copy()
    sample = deterministic_full_date_sample(refit, base.training_row_cap)
    test = data.loc[fold.test_index]
    processor = TrainOnlyPreprocessor().fit(sample, features)
    x_train = processor.transform(sample, features)
    x_test = processor.transform(test, features)
    y_train = pd.to_numeric(sample[target], errors="raise").to_numpy(float)
    rounds = None
    if experiment["model"] == "ridge":
        model = RidgeModel(base.ridge_alpha).fit(x_train, y_train)
        predicted = model.predict(x_test)
        train_predicted = model.predict(x_train)
        signature = model.signature()
    else:
        config = _config(experiment["config"])
        if config.early_stopping_rounds:
            inner = data.loc[fold.train_index].copy()
            valid = data.loc[fold.validation_index].copy()
            inner_finite = pd.to_numeric(inner[target], errors="coerce")
            inner = inner[inner_finite.notna() & np.isfinite(inner_finite)].copy()
            inner_sample = deterministic_full_date_sample(inner, base.training_row_cap)
            inner_processor = TrainOnlyPreprocessor().fit(inner_sample, features)
            early_model = _fit_lgbm(
                inner_processor.transform(inner_sample, features),
                pd.to_numeric(inner_sample[target], errors="raise").to_numpy(float),
                config,
                base.random_seed,
                valid=(
                    inner_processor.transform(valid, features),
                    pd.to_numeric(valid[target], errors="raise").to_numpy(float),
                ),
            )
            rounds = max(1, int(early_model.best_iteration))
        model = _fit_lgbm(x_train, y_train, config, base.random_seed, rounds=rounds)
        predicted = np.asarray(model.predict(x_test), dtype=float)
        train_predicted = np.asarray(model.predict(x_train), dtype=float)
        signature = _model_hash(model)
    train_frame = sample[["date", "future_return_20d"]].copy()
    train_frame["score"] = train_predicted
    train_metrics = _score_summary(daily_rank_metrics(train_frame, "score", "future_return_20d"))
    receipt = {
        "experiment_id": experiment["id"],
        "test_year": year,
        "features": list(features),
        "feature_count": len(features),
        "training_rows": len(sample),
        "preprocessor_fit_rows": len(processor.fit_row_ids_),
        "model_hash": signature,
        "early_stop_rounds": rounds,
        "config": asdict(_config(experiment["config"]))
        if experiment["model"] == "lightgbm"
        else {"ridge_alpha": base.ridge_alpha},
    }
    return predicted, train_metrics, receipt


def _past_validation_scores(
    data: pd.DataFrame, fold, base: ChallengerSettings, year: int
) -> tuple[float, dict[str, float], dict]:
    features = tuple(_selected_factors()[year])
    target = "return_rank_20d"
    train = data.loc[fold.train_index].copy()
    finite = pd.to_numeric(train[target], errors="coerce")
    train = train[finite.notna() & np.isfinite(finite)].copy()
    sample = deterministic_full_date_sample(train, base.training_row_cap)
    valid = data.loc[fold.validation_index].copy()
    processor = TrainOnlyPreprocessor().fit(sample, features)
    x_train = processor.transform(sample, features)
    x_valid = processor.transform(valid, features)
    y = pd.to_numeric(sample[target], errors="raise").to_numpy(float)
    lgbm = _fit_lgbm(x_train, y, BASELINE_CONFIG, base.random_seed)
    ridge = RidgeModel(base.ridge_alpha).fit(x_train, y)
    validation = valid[["date", "regime", "future_return_20d"]].copy()
    validation["validation_lgbm"] = lgbm.predict(x_valid)
    validation["validation_ridge"] = ridge.predict(x_valid)
    global_weight, regime_weights = _validation_weights(validation)
    return (
        global_weight,
        regime_weights,
        {
            "test_year": year,
            "source": "past_validation_only",
            "global_lgbm_weight": global_weight,
            "regime_lgbm_weights": regime_weights,
        },
    )


def train_oos_scores(
    data: pd.DataFrame, settings: Gen3Settings, base: ChallengerSettings
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], list[dict]]:
    identity = [
        "date",
        "symbol",
        "industry",
        "broad_sector",
        "benchmark_weight",
        "benchmark_weight_rank",
        "amount_rank",
        "volatility_20",
        "volatility_60_rank",
        "momentum",
        "liquidity",
        "regime",
        "future_return_20d",
        "entry_tradable_20",
        "execution_return_20",
    ]
    pieces: list[pd.DataFrame] = []
    train_rows: list[dict] = []
    fold_rows: list[dict] = []
    model_rows: list[dict] = []
    weight_rows: list[dict] = []
    for year in settings.years:
        fold = build_fold(
            data,
            year,
            settings.horizon,
            training_window_years=base.training_window_years,
            validation_years=base.validation_years,
            purge_gap_trading_days=base.purge_gaps[settings.horizon],
        )
        fold_rows.append(fold_receipt(data, fold))
        piece = data.loc[fold.test_index, identity].copy()
        piece["test_year"] = year
        for experiment in EXPERIMENTS:
            predicted, metrics, receipt = _fit_one_experiment(experiment, data, fold, base, year)
            piece[experiment["score"]] = predicted
            train_rows.append(
                {
                    "experiment_id": experiment["id"],
                    "score": experiment["score"],
                    "test_year": year,
                    **metrics,
                }
            )
            model_rows.append(receipt)
        global_weight, regime_weights, weight_receipt = _past_validation_scores(
            data, fold, base, year
        )
        weight_rows.append(weight_receipt)
        piece["ensemble_50_50"] = _rank_average(piece, "gen2_baseline", "ridge_gen2", 0.5)
        piece["ensemble_past_ic"] = _rank_average(
            piece, "gen2_baseline", "ridge_gen2", global_weight
        )
        weights = piece["regime"].map(regime_weights).fillna(global_weight)
        piece["ensemble_regime"] = _rank_average(piece, "gen2_baseline", "ridge_gen2", weights)
        piece["residualized_shallow"] = _residualize_scores(piece, piece["lgbm_shallow"]).fillna(
            piece["lgbm_shallow"]
        )
        pieces.append(piece)
    scores = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"])
    return scores, pd.DataFrame(train_rows), fold_rows, [*model_rows, *weight_rows]


def _all_experiments() -> list[dict]:
    return [*EXPERIMENTS, *DERIVED_EXPERIMENTS]


def rank_metrics(
    scores: pd.DataFrame, train: pd.DataFrame, settings: Gen3Settings
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    rows, yearly_rows, daily_map = [], [], {}
    train_grouped = train.groupby("score")["rank_ic_mean"].mean().to_dict()
    for experiment in _all_experiments():
        score = experiment["score"]
        daily = daily_rank_metrics(scores, score, "future_return_20d")
        daily_map[score] = daily.set_index("date")
        overall = _score_summary(daily)
        train_ic = float(train_grouped.get(score, np.nan))
        rows.append(
            {
                "experiment_id": experiment["id"],
                "score": score,
                "family": experiment["family"],
                **overall,
                "train_rank_ic": train_ic,
                "train_oos_gap": train_ic - overall["rank_ic_mean"]
                if np.isfinite(train_ic)
                else np.nan,
            }
        )
        for year in settings.years:
            subset = daily[daily["date"].dt.year.eq(year)]
            yearly_rows.append(
                {
                    "experiment_id": experiment["id"],
                    "score": score,
                    "year": year,
                    **_score_summary(subset),
                }
            )
    metrics = pd.DataFrame(rows)
    yearly = pd.DataFrame(yearly_rows)
    worst = yearly.groupby("score")["rank_ic_mean"].min().rename("worst_year_rank_ic")
    metrics = metrics.merge(worst, on="score", how="left")
    paired_rows = []
    baseline = daily_map["gen2_baseline"]["rank_ic"]
    for experiment in _all_experiments():
        score = experiment["score"]
        candidate = daily_map[score]["rank_ic"]
        paired = pd.concat(
            [candidate.rename("candidate"), baseline.rename("baseline")], axis=1
        ).dropna()
        boot = moving_block_bootstrap_delta(
            candidate,
            baseline,
            replications=settings.bootstrap_replications,
            block_length=settings.bootstrap_block_length,
            seed=settings.random_seed,
        )
        fold_delta = (
            yearly[yearly["score"].eq(score)].set_index("year")["rank_ic_mean"]
            - yearly[yearly["score"].eq("gen2_baseline")].set_index("year")["rank_ic_mean"]
        )
        paired_rows.append(
            {
                "experiment_id": experiment["id"],
                "score": score,
                **boot,
                "probability_bootstrap_delta_gt_zero": np.nan,
                "positive_daily_difference_ratio": float(
                    (paired["candidate"] > paired["baseline"]).mean()
                ),
                "fold_wins": int((fold_delta > 0).sum()),
                "folds": len(fold_delta),
            }
        )
    return metrics, yearly, pd.DataFrame(paired_rows), daily_map


def sliced_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    bucketed = scores.copy()
    bucketed["cap_bucket"] = bucketed.groupby("date")["benchmark_weight_rank"].transform(
        lambda values: pd.qcut(values.rank(method="first"), 3, labels=["small", "mid", "large"])
    )
    bucketed["vol_bucket"] = bucketed.groupby("date")["volatility_20"].transform(
        lambda values: pd.qcut(values.rank(method="first"), 3, labels=["low", "mid", "high"])
    )
    rows = []
    dimensions = {
        "market_regime": "regime",
        "sector": "broad_sector",
        "market_cap": "cap_bucket",
        "volatility": "vol_bucket",
    }
    for experiment in _all_experiments():
        for dimension, column in dimensions.items():
            for bucket, part in bucketed.groupby(column, observed=True):
                daily = daily_rank_metrics(part, experiment["score"], "future_return_20d")
                rows.append(
                    {
                        "experiment_id": experiment["id"],
                        "score": experiment["score"],
                        "dimension": dimension,
                        "bucket": str(bucket),
                        "rows": len(part),
                        **_score_summary(daily),
                    }
                )
    return pd.DataFrame(rows)


def quantile_metrics(scores: pd.DataFrame, settings: Gen3Settings) -> pd.DataFrame:
    rows = []
    for experiment in _all_experiments():
        values = quantile_returns(
            scores, experiment["score"], "future_return_20d", settings.quantiles
        )
        means = values.groupby("quantile")["actual_return"].mean()
        monotonic = float(pd.Series(means.index, dtype=float).corr(means, method="spearman"))
        adjacent = float((np.diff(means.to_numpy()) > 0).mean()) if len(means) > 1 else np.nan
        for quantile, value in means.items():
            rows.append(
                {
                    "experiment_id": experiment["id"],
                    "score": experiment["score"],
                    "quantile": int(quantile),
                    "mean_return": float(value),
                    "monotonic_correlation": monotonic,
                    "q5_minus_q1": float(means.iloc[-1] - means.iloc[0]),
                    "adjacent_consistency": adjacent,
                }
            )
    return pd.DataFrame(rows)


def residual_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for experiment in _all_experiments():
        residual = _residualize_scores(scores, scores[experiment["score"]])
        frame = scores[["date", "future_return_20d"]].copy()
        frame["score"] = residual
        rows.append(
            {
                "experiment_id": experiment["id"],
                "score": experiment["score"],
                **_score_summary(daily_rank_metrics(frame, "score", "future_return_20d")),
                "controls": "sector,size,volatility,momentum,liquidity",
            }
        )
    return pd.DataFrame(rows)


def portfolio_metrics(
    scores: pd.DataFrame,
    settings: Gen3Settings,
    base: ChallengerSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    book, price_evidence = _load_verified_price_book(scores, base)
    period_cache: dict[tuple[str, str, int], pd.DataFrame] = {}
    turnover_rows, cost_rows = [], []
    for experiment in _all_experiments():
        score = experiment["score"]
        policy = PortfolioPolicy(name=f"{score}_sector_top20", top_k=20, sector_balanced=True)
        periods, _ = evaluate_stateful_portfolio_policy(scores, score, 20, policy, book)
        period_cache[(score, "sector_balanced", 20)] = periods
        summary = summarize_stateful_portfolio(periods, 20)
        turnover_rows.append(
            {
                "experiment_id": experiment["id"],
                "score": score,
                "policy": policy.name,
                "top_k": 20,
                "buffer_exit_rank": np.nan,
                **summary,
            }
        )
        for bps in settings.cost_bps:
            adjusted = (
                periods["gross_return"]
                - (periods["buy_turnover"] + periods["sell_turnover"]) * bps / 10_000
            )
            total = float((1 + adjusted).prod() - 1)
            benchmark = float((1 + periods["research_benchmark_proxy_return"]).prod() - 1)
            cost_rows.append(
                {
                    "experiment_id": experiment["id"],
                    "score": score,
                    "policy": policy.name,
                    "top_k": 20,
                    "cost_bps": bps,
                    "gross_total_return": float((1 + periods["gross_return"]).prod() - 1),
                    "net_total_return": total,
                    "net_research_proxy_alpha": total - benchmark,
                    "average_one_way_turnover": summary["average_one_way_turnover"],
                }
            )
    provisional = pd.DataFrame(cost_rows)
    at_20 = provisional[provisional["cost_bps"].eq(20)].sort_values(
        "net_research_proxy_alpha", ascending=False
    )
    finalists = list(dict.fromkeys(["gen2_baseline", *at_20["score"].head(2).tolist()]))
    for score in finalists:
        experiment_id = next(item["id"] for item in _all_experiments() if item["score"] == score)
        for top_k in settings.top_ks:
            if top_k == 20:
                continue
            policy = PortfolioPolicy(
                name=f"{score}_sector_top{top_k}", top_k=top_k, sector_balanced=True
            )
            periods, _ = evaluate_stateful_portfolio_policy(scores, score, 20, policy, book)
            summary = summarize_stateful_portfolio(periods, 20)
            turnover_rows.append(
                {
                    "experiment_id": experiment_id,
                    "score": score,
                    "policy": policy.name,
                    "top_k": top_k,
                    "buffer_exit_rank": np.nan,
                    **summary,
                }
            )
            for bps in settings.cost_bps:
                adjusted = (
                    periods["gross_return"]
                    - (periods["buy_turnover"] + periods["sell_turnover"]) * bps / 10_000
                )
                total = float((1 + adjusted).prod() - 1)
                benchmark = float((1 + periods["research_benchmark_proxy_return"]).prod() - 1)
                cost_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "score": score,
                        "policy": policy.name,
                        "top_k": top_k,
                        "cost_bps": bps,
                        "gross_total_return": float((1 + periods["gross_return"]).prod() - 1),
                        "net_total_return": total,
                        "net_research_proxy_alpha": total - benchmark,
                        "average_one_way_turnover": summary["average_one_way_turnover"],
                    }
                )
        buffer_policy = PortfolioPolicy(
            name=f"{score}_buffer20_30",
            top_k=20,
            buffer_exit_rank=30,
            sector_balanced=False,
        )
        periods, _ = evaluate_stateful_portfolio_policy(scores, score, 20, buffer_policy, book)
        summary = summarize_stateful_portfolio(periods, 20)
        turnover_rows.append(
            {
                "experiment_id": experiment_id,
                "score": score,
                "policy": buffer_policy.name,
                "top_k": 20,
                "buffer_exit_rank": 30,
                **summary,
            }
        )
        for bps in settings.cost_bps:
            adjusted = (
                periods["gross_return"]
                - (periods["buy_turnover"] + periods["sell_turnover"]) * bps / 10_000
            )
            total = float((1 + adjusted).prod() - 1)
            benchmark = float((1 + periods["research_benchmark_proxy_return"]).prod() - 1)
            cost_rows.append(
                {
                    "experiment_id": experiment_id,
                    "score": score,
                    "policy": buffer_policy.name,
                    "top_k": 20,
                    "cost_bps": bps,
                    "gross_total_return": float((1 + periods["gross_return"]).prod() - 1),
                    "net_total_return": total,
                    "net_research_proxy_alpha": total - benchmark,
                    "average_one_way_turnover": summary["average_one_way_turnover"],
                }
            )
    return pd.DataFrame(cost_rows), pd.DataFrame(turnover_rows), price_evidence


def feature_evidence(
    data: pd.DataFrame, metrics: pd.DataFrame, residual: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    baseline_ic = float(metrics.loc[metrics["score"].eq("gen2_baseline"), "rank_ic_mean"].iloc[0])
    residual_map = residual.set_index("score")["rank_ic_mean"].to_dict()
    for experiment in _all_experiments():
        current = metrics[metrics["score"].eq(experiment["score"])].iloc[0]
        rows.append(
            {
                "item": experiment["id"],
                "kind": "experiment",
                "family": experiment["family"],
                "rank_ic": current["rank_ic_mean"],
                "incremental_rank_ic": current["rank_ic_mean"] - baseline_ic,
                "residual_rank_ic": residual_map.get(experiment["score"], np.nan),
                "evidence_basis": experiment["hypothesis"],
            }
        )
    for feature in NEW_FEATURES:
        daily = daily_rank_metrics(data, feature, "future_return_20d")
        summary = _score_summary(daily)
        yearly = [
            _score_summary(daily[daily["date"].dt.year.eq(year)])["rank_ic_mean"]
            for year in range(2020, 2026)
        ]
        rows.append(
            {
                "item": feature,
                "kind": "new_feature_standalone",
                "family": next(
                    row["category"] for row in feature_registry() if row["feature_name"] == feature
                ),
                "rank_ic": summary["rank_ic_mean"],
                "incremental_rank_ic": np.nan,
                "residual_rank_ic": np.nan,
                "evidence_basis": f"positive_years={sum(value > 0 for value in yearly if np.isfinite(value))}/6",
            }
        )
    return pd.DataFrame(rows)


def classify_candidates(
    metrics: pd.DataFrame,
    yearly: pd.DataFrame,
    quantiles: pd.DataFrame,
    residual: pd.DataFrame,
    paired: pd.DataFrame,
    costs: pd.DataFrame,
    settings: Gen3Settings,
) -> tuple[pd.DataFrame, dict]:
    baseline = metrics[metrics["score"].eq("gen2_baseline")].iloc[0]
    baseline_quantile = quantiles[quantiles["score"].eq("gen2_baseline")].iloc[0]
    baseline_residual = float(
        residual.loc[residual["score"].eq("gen2_baseline"), "rank_ic_mean"].iloc[0]
    )
    cost20 = (
        costs[costs["cost_bps"].eq(20) & costs["policy"].str.endswith("sector_top20")]
        .drop_duplicates("score")
        .set_index("score")["net_research_proxy_alpha"]
        .to_dict()
    )
    paired_map = paired.set_index("score").to_dict("index")
    residual_map = residual.set_index("score")["rank_ic_mean"].to_dict()
    quantile_map = quantiles.drop_duplicates("score").set_index("score").to_dict("index")
    rows = []
    for _, current in metrics.iterrows():
        score = current["score"]
        year_values = yearly[yearly["score"].eq(score)]
        pair = paired_map[score]
        quantile = quantile_map[score]
        gates = {
            "rank_ic": bool(current["rank_ic_mean"] > baseline["rank_ic_mean"]),
            "paired_ci": bool(pair["ci_lower"] > 0),
            "icir": bool(current["icir"] > baseline["icir"]),
            "worst_year": bool(current["worst_year_rank_ic"] >= baseline["worst_year_rank_ic"]),
            "positive_years": bool((year_values["rank_ic_mean"] > 0).sum() >= 5),
            "monotonicity": bool(
                quantile["monotonic_correlation"] > baseline_quantile["monotonic_correlation"]
                and quantile["adjacent_consistency"] >= baseline_quantile["adjacent_consistency"]
            ),
            "cost20": bool(cost20.get(score, -np.inf) >= 0),
            "overfit_gap": bool(
                np.isfinite(current["train_oos_gap"])
                and current["train_oos_gap"] <= baseline["train_oos_gap"]
            ),
            "residual_alpha": bool(residual_map[score] >= baseline_residual - 0.002),
        }
        strong = (
            gates["rank_ic"]
            and gates["icir"]
            and gates["paired_ci"]
            and gates["residual_alpha"]
            and sum(gates.values()) >= 7
        )
        modest = gates["rank_ic"] and sum(gates.values()) >= 5
        status = "PROMISING_RESEARCH_ONLY" if strong else "INCONCLUSIVE" if modest else "REJECTED"
        rows.append(
            {
                "experiment_id": current["experiment_id"],
                "score": score,
                "rank_ic": current["rank_ic_mean"],
                "icir": current["icir"],
                "worst_year": current["worst_year_rank_ic"],
                "residual_ic": residual_map[score],
                "20bps_alpha": cost20.get(score, np.nan),
                "train_oos_gap": current["train_oos_gap"],
                "paired_ci_lower": pair["ci_lower"],
                "paired_ci_upper": pair["ci_upper"],
                "fold_wins": pair["fold_wins"],
                "gates_passed": sum(gates.values()),
                "gate_detail": json.dumps(gates, sort_keys=True),
                "status": status,
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["status", "gates_passed", "rank_ic", "icir"],
        ascending=[False, False, False, False],
    )
    promising = ranking[ranking["status"].eq("PROMISING_RESEARCH_ONLY")].head(
        settings.maximum_research_candidates
    )
    marginal = ranking[ranking["status"].eq("INCONCLUSIVE")].head(
        settings.maximum_research_candidates
    )
    selected = promising if len(promising) else marginal
    if len(promising):
        final_status, assessment = "GEN3_ALPHA_IMPROVEMENT_COMPLETE", "YES — RESEARCH EVIDENCE"
    elif len(marginal):
        final_status, assessment = "GEN3_ALPHA_IMPROVEMENT_INCONCLUSIVE", "MARGINAL"
    else:
        final_status, assessment = "GEN3_ALPHA_IMPROVEMENT_INCONCLUSIVE", "NO"
    summary = {
        "final_status": final_status,
        "assessment": assessment,
        "candidate_role": "GEN3_RESEARCH_CANDIDATE",
        "selected": selected[["experiment_id", "score", "status"]].to_dict("records"),
        "champion_promoted": False,
        "untouched_confirmatory_holdout": False,
        "interpretation": "development/comparative research evidence only",
    }
    return ranking, summary


def _markdown_table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    shown = frame[columns].head(limit) if limit else frame[columns]
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, divider]
    for _, row in shown.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(
                f"{value:.6f}"
                if isinstance(value, (float, np.floating)) and np.isfinite(value)
                else str(value)
            )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    settings: Gen3Settings,
    protocol: dict,
    data_evidence: dict,
    metrics: pd.DataFrame,
    yearly: pd.DataFrame,
    regimes: pd.DataFrame,
    quantiles: pd.DataFrame,
    residual: pd.DataFrame,
    paired: pd.DataFrame,
    costs: pd.DataFrame,
    turnover: pd.DataFrame,
    feature_rows: pd.DataFrame,
    ranking: pd.DataFrame,
    summary: dict,
    price_evidence: dict,
) -> None:
    baseline = metrics[metrics["score"].eq("gen2_baseline")].iloc[0]
    best = ranking.iloc[0]
    baseline_cost = costs[
        costs["score"].eq("gen2_baseline")
        & costs["cost_bps"].eq(20)
        & costs["policy"].str.endswith("sector_top20")
    ].iloc[0]
    regularization = metrics[metrics["family"].isin(["baseline", "regularization"])]
    selection = metrics[metrics["family"].eq("feature_selection")]
    linear = metrics[metrics["family"].eq("linear")]
    ensemble = metrics[metrics["family"].eq("ensemble")]
    rejected = ranking[ranking["status"].eq("REJECTED")]
    selected = summary["selected"]
    topk = costs[
        costs["score"].eq(best["score"])
        & costs["cost_bps"].eq(20)
        & ~costs["policy"].str.contains("buffer")
    ]
    weak = regimes[
        regimes["score"].isin(["gen2_baseline", best["score"]])
        & regimes["dimension"].isin(["market_regime", "market_cap", "volatility", "sector"])
        & regimes["bucket"].isin(["risk_off", "large", "low", "technology"])
    ]
    quantile_best = quantiles[quantiles["score"].isin(["gen2_baseline", best["score"]])]
    paired_best = paired[paired["score"].isin(ranking.head(4)["score"])]
    report = f"""# GEN3_ALPHA_IMPROVEMENT_REPORT

## 1. Final Status

`{summary["final_status"]}`. Assessment: `{summary["assessment"]}`. No champion promotion was performed.

## 2. Baseline

Merged SHA `{MERGED_DIAGNOSTIC_SHA}`; model `{GEN2_MODEL_ID}`. Reproduced Rank IC {baseline["rank_ic_mean"]:.6f}, ICIR {baseline["icir"]:.6f}, residual IC {float(residual.loc[residual["score"].eq("gen2_baseline"), "rank_ic_mean"].iloc[0]):.6f}; Top20 20 bps proxy alpha {baseline_cost["net_research_proxy_alpha"]:.6f}.

## 3. Experimental Integrity

PIT checks all passed: `{data_evidence["pit_checks"]}`. Six annual walk-forward folds use 8-year training, one past validation year and 21-trading-day purge. No random split, future normalization, future regime or 2026 label was used. Deterministic seed is {settings.random_seed}. This is reused development/comparative evidence; no untouched holdout exists.

## 4. Experiment Registry

{_markdown_table(pd.DataFrame(_all_experiments()), ["id", "family", "score", "hypothesis"])}

## 5. Gen2 Baseline Reproduction

The exact frozen feature-by-year and LightGBM configuration reproduced Rank IC {baseline["rank_ic_mean"]:.6f} versus diagnostic 0.049877, a difference of {baseline["rank_ic_mean"] - 0.04987666852835187:.8f}. This anchors paired tests.

## 6. Regularization Results

{_markdown_table(regularization, ["experiment_id", "train_rank_ic", "rank_ic_mean", "icir", "train_oos_gap", "worst_year_rank_ic"])}

## 7. Feature Selection Results

{_markdown_table(selection, ["experiment_id", "rank_ic_mean", "icir", "train_oos_gap", "worst_year_rank_ic"])}

## 8. New Feature Results

{_markdown_table(feature_rows[feature_rows["kind"].eq("new_feature_standalone")], ["item", "family", "rank_ic", "evidence_basis"])}

The group-level incremental result is recorded for `E_NEW_INDEPENDENT_FEATURES` in `feature_ablation.csv`; all definitions, lookbacks, missing behavior and provenance are in both feature registries.

## 9. Regime Results

{_markdown_table(weak[weak["dimension"].isin(["market_regime", "volatility"])], ["score", "dimension", "bucket", "rank_ic_mean", "icir", "positive_ic_ratio"])}

Repository regimes map current-date market state to risk-on, risk-off and neutral; bull/bear/sideways are not separately relabeled because doing so after observing returns would violate the frozen PIT semantics.

## 10. Sector / Cap Results

{_markdown_table(weak[weak["dimension"].isin(["sector", "market_cap"])], ["score", "dimension", "bucket", "rank_ic_mean", "icir", "positive_ic_ratio"])}

## 11. Ridge / Linear Results

{_markdown_table(linear, ["experiment_id", "rank_ic_mean", "icir", "train_oos_gap", "worst_year_rank_ic"])}

## 12. Ensemble Results

{_markdown_table(ensemble, ["experiment_id", "rank_ic_mean", "icir", "worst_year_rank_ic"])}

Weights for adaptive ensembles came exclusively from each fold's past validation year; no full-period weight fitting occurred.

## 13. Quantile Monotonicity

{_markdown_table(quantile_best, ["score", "quantile", "mean_return", "monotonic_correlation", "q5_minus_q1", "adjacent_consistency"])}

## 14. Top-K

{_markdown_table(topk, ["score", "top_k", "net_research_proxy_alpha", "average_one_way_turnover"])}

The best historical K remains development evidence and is exposed to multiple-comparison/data-mining risk; production Top20 was not changed.

## 15. Turnover

{_markdown_table(turnover[turnover["score"].isin(["gen2_baseline", best["score"]])], ["score", "policy", "top_k", "buffer_exit_rank", "average_one_way_turnover", "annualized_turnover"])}

## 16. Cost Sensitivity

{_markdown_table(costs[costs["score"].isin(["gen2_baseline", best["score"]]) & costs["top_k"].eq(20)], ["score", "policy", "cost_bps", "gross_total_return", "net_research_proxy_alpha", "average_one_way_turnover"])}

## 17. Residual Alpha

Scores were cross-sectionally residualized against sector, size, volatility, momentum and liquidity using only same-date exposures.

{_markdown_table(residual.sort_values("rank_ic_mean", ascending=False), ["experiment_id", "rank_ic_mean", "icir", "positive_ic_ratio"], 8)}

## 18. Overfitting

{_markdown_table(metrics.sort_values("train_oos_gap"), ["experiment_id", "train_rank_ic", "rank_ic_mean", "train_oos_gap", "rank_ic_std"], 12)}

Derived ensembles have no single fitted training score and are therefore reported with a missing train/OOS gap rather than an invented value.

## 19. Statistical Comparison

{_markdown_table(paired_best, ["experiment_id", "mean_delta", "ci_lower", "ci_upper", "positive_daily_difference_ratio", "fold_wins"])}

The confidence intervals use paired 20-session moving-block bootstrap with {settings.bootstrap_replications} replications.

## 20. Candidate Ranking

{_markdown_table(ranking, ["experiment_id", "rank_ic", "icir", "worst_year", "residual_ic", "20bps_alpha", "gates_passed", "status"])}

## 21. Best Research Candidate

Selected (maximum two): `{selected}`. The leading candidate is `{best["experiment_id"]}` because it passed {best["gates_passed"]} jointly pre-registered gates. Weaknesses remain its {best["paired_ci_lower"]:.6f} paired CI lower bound and lack of untouched confirmation. It remains `GEN3_RESEARCH_CANDIDATE`, not champion.

## 22. Rejected Ideas

{_markdown_table(rejected, ["experiment_id", "rank_ic", "icir", "gates_passed", "status"]) if len(rejected) else "No experiment was automatically hidden; none met the REJECTED rule."}

## 23. Gen3 Assessment

Did Gen3 materially improve predictive quality over Gen2? **{summary["assessment"]}**. This label is limited to development/comparative research evidence and is not confirmatory proof.

## 24. Next Phase

If a `PROMISING_RESEARCH_ONLY` candidate exists, freeze it for `GEN3_CONFIRMATORY_FORWARD_VALIDATION` on genuinely future matured observations. Otherwise continue `ALPHA_RESEARCH`; do not activate or retune against 2025. 10D and 40D remain `NOT EVALUATED` because no same-semantics frozen labels were introduced.

## 25. Git / PR

Branch `codex/gen3-alpha-improvement`. Baseline `{MERGED_DIAGNOSTIC_SHA}`. Research code, tests and isolated artifacts only; frozen/007–012 modified count must remain zero. PR and CI fields are finalized in delivery metadata after push; the PR must not be auto-merged.
"""
    (settings.artifact_dir / "GEN3_ALPHA_IMPROVEMENT_REPORT.md").write_text(
        report, encoding="utf-8", newline="\n"
    )


def _write_manifest(directory: Path) -> dict:
    manifest = {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    _write_json(directory / "artifact_manifest.json", manifest)
    return manifest


def run_experiments(settings: Gen3Settings | None = None) -> dict:
    settings = settings or Gen3Settings()
    base = ChallengerSettings()
    protocol_path = settings.artifact_dir / "research_protocol.json"
    registry_path = settings.artifact_dir / "experiment_registry.json"
    if not protocol_path.is_file() or not registry_path.is_file():
        raise RuntimeError(
            "GEN3_EXPERIMENT_BLOCKED: freeze the protocol before full evaluation"
        )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    frozen_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if frozen_registry != _json_value([*EXPERIMENTS, *DERIVED_EXPERIMENTS]):
        raise RuntimeError("GEN3_EXPERIMENT_INVALID: experiment registry drift")
    data, data_evidence = load_maturity_safe_development_dataset(base)
    data = add_gen3_features(data)
    if data["date"].max() >= pd.Timestamp("2026-01-01"):
        raise RuntimeError("GEN3_EXPERIMENT_INVALID: 2026 decision data entered evaluation")
    scores, train, folds, model_receipts = train_oos_scores(data, settings, base)
    metrics, yearly, paired, _ = rank_metrics(scores, train, settings)
    regimes = sliced_metrics(scores)
    quantiles = quantile_metrics(scores, settings)
    residual = residual_metrics(scores)
    costs, turnover, price_evidence = portfolio_metrics(scores, settings, base)
    features = feature_evidence(data, metrics, residual)
    ranking, summary = classify_candidates(
        metrics, yearly, quantiles, residual, paired, costs, settings
    )
    baseline = metrics[metrics["score"].eq("gen2_baseline")].iloc[0].to_dict()
    baseline["diagnostic_reference_rank_ic"] = 0.04987666852835187
    baseline["reproduction_difference"] = baseline["rank_ic_mean"] - 0.04987666852835187
    _write_json(settings.artifact_dir / "baseline_metrics.json", baseline)
    _write_csv(settings.artifact_dir / "challenger_metrics.csv", metrics)
    _write_csv(settings.artifact_dir / "yearly_metrics.csv", yearly)
    _write_csv(settings.artifact_dir / "regime_metrics.csv", regimes)
    _write_csv(settings.artifact_dir / "quantile_metrics.csv", quantiles)
    _write_csv(settings.artifact_dir / "cost_metrics.csv", costs)
    _write_csv(settings.artifact_dir / "turnover_metrics.csv", turnover)
    _write_json(settings.artifact_dir / "feature_registry.json", feature_registry())
    _write_csv(settings.artifact_dir / "feature_ablation.csv", features)
    _write_csv(settings.artifact_dir / "paired_comparisons.csv", paired)
    _write_csv(settings.artifact_dir / "candidate_ranking.csv", ranking)
    _write_json(settings.artifact_dir / "candidate_summary.json", summary)
    _write_json(settings.artifact_dir / "fold_receipts.json", folds)
    _write_json(settings.artifact_dir / "model_receipts.json", model_receipts)
    _write_json(settings.artifact_dir / "data_evidence.json", data_evidence)
    _write_json(settings.artifact_dir / "price_evidence.json", price_evidence)
    _write_json(
        settings.artifact_dir / "reproducibility.json",
        {
            "git_sha": _git("rev-parse", "HEAD"),
            "input_data_hash": data_evidence["dataset_sha256"],
            "diagnostic_baseline": MERGED_DIAGNOSTIC_SHA,
            "folds": folds,
            "purge": 21,
            "seed": settings.random_seed,
            "evaluation_dates": [
                str(scores["date"].min().date()),
                str(scores["date"].max().date()),
            ],
            "model_receipt_hash": hashlib.sha256(
                json.dumps(_json_value(model_receipts), sort_keys=True).encode()
            ).hexdigest(),
            "deterministic": True,
            "2026_labels_read": False,
        },
    )
    write_report(
        settings,
        protocol,
        data_evidence,
        metrics,
        yearly,
        regimes,
        quantiles,
        residual,
        paired,
        costs,
        turnover,
        features,
        ranking,
        summary,
        price_evidence,
    )
    manifest = _write_manifest(settings.artifact_dir)
    summary["artifact_count"] = len(manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated Gen3 alpha improvement research")
    parser.add_argument("--freeze-protocol", action="store_true")
    args = parser.parse_args()
    if args.freeze_protocol:
        result = freeze_protocol()
        print(json.dumps(_json_value(result), ensure_ascii=False, indent=2))
    else:
        result = run_experiments()
        print(json.dumps(_json_value(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
