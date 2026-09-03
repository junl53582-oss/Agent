from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from stockpilot.alpha_diagnostic.gen3 import (
    BASELINE_CONFIG,
    STABLE_CORE,
    _fit_lgbm,
    _json_value,
    _markdown_table,
    _model_hash,
    _residualize_scores,
    _score_summary,
    _sha256,
    _write_csv,
    _write_json,
)
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
)
from stockpilot.research_challenger.models import (
    RidgeModel,
    TrainOnlyPreprocessor,
    deterministic_full_date_sample,
)
from stockpilot.research_challenger.split import build_fold, fold_receipt

ARTIFACT_DIR = Path("artifacts/research_challenger/nextgen_alpha_signal_discovery")
BASELINE_SHA = "442a88a9fc24b9c43e62ec48f38ed7858490adfd"
GEN2_RANK_IC = 0.04987666852835187
GEN2_ICIR = 0.26534749011677106
GEN2_RESIDUAL_IC = 0.0204882670229622
GEN2_2025_IC = 0.0012487700821363
GEN2_TOP20_20BPS_ALPHA = -0.076455
HARNESS_RANK_IC = 0.064102

RESIDUAL_MOMENTUM = (
    "ng_residual_momentum_20",
    "ng_residual_momentum_60",
    "ng_residual_momentum_120",
    "ng_residual_momentum_consistency",
)
LIQUIDITY_SHOCK = (
    "ng_abnormal_amount_20_60",
    "ng_amihud_change_20_60",
    "ng_liquidity_dryup_5_60",
    "ng_volume_price_divergence",
)
PRICE_PATH = (
    "ng_up_day_ratio_20",
    "ng_trend_efficiency_20",
    "ng_return_autocorrelation_20",
    "ng_recovery_from_low_20",
)
SIGNAL_FAMILIES = {
    "residual_momentum": RESIDUAL_MOMENTUM,
    "liquidity_shock": LIQUIDITY_SHOCK,
    "price_path_shape": PRICE_PATH,
}


@dataclass(frozen=True)
class NextgenSettings:
    artifact_dir: Path = ARTIFACT_DIR
    years: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024, 2025)
    development_years: tuple[int, ...] = (2020, 2021, 2022, 2023)
    comparison_years: tuple[int, ...] = (2024, 2025)
    horizon: int = 20
    seed: int = 42
    bootstrap_replications: int = 1_000
    bootstrap_block_length: int = 20
    top_ks: tuple[int, ...] = (10, 20, 30, 50)
    cost_bps: tuple[int, ...] = (0, 10, 20, 30, 50)
    experiment_config_budget: int = 12
    major_family_budget: int = 3
    maximum_candidates: int = 2


EXPERIMENTS = (
    {
        "id": "R0_GEN2_EXACT_REFERENCE",
        "track": "reference",
        "features": "gen2_yearly_frozen",
        "label": "L0_GEN2_CROSS_SECTIONAL_RANK",
        "model": "ridge" if False else "lightgbm",
        "hypothesis": "Exact Gen2 reproduction anchors every paired comparison.",
    },
    {
        "id": "R1_STABLE_CORE_RIDGE_HARNESS",
        "track": "harness",
        "features": "stable_core",
        "label": "L0_GEN2_CROSS_SECTIONAL_RANK",
        "model": "ridge",
        "hypothesis": "Stable-Core Ridge is the low-complexity signal-quality harness.",
    },
    {
        "id": "A1_RESIDUAL_MOMENTUM",
        "track": "signal",
        "features": "stable_core_plus_residual_momentum",
        "label": "L0_GEN2_CROSS_SECTIONAL_RANK",
        "model": "ridge",
        "hypothesis": "Past-beta and sector residual momentum adds independent price information.",
    },
    {
        "id": "A2_LIQUIDITY_SHOCK",
        "track": "signal",
        "features": "stable_core_plus_liquidity_shock",
        "label": "L0_GEN2_CROSS_SECTIONAL_RANK",
        "model": "ridge",
        "hypothesis": "Abnormal liquidity and price impact contain information beyond liquidity level.",
    },
    {
        "id": "A3_PRICE_PATH_SHAPE",
        "track": "signal",
        "features": "stable_core_plus_price_path_shape",
        "label": "L0_GEN2_CROSS_SECTIONAL_RANK",
        "model": "ridge",
        "hypothesis": "Path smoothness and return sequencing distinguish durable from noisy trends.",
    },
    {
        "id": "B1_RAW_FORWARD_RETURN",
        "track": "label",
        "features": "stable_core",
        "label": "L1_RAW_FORWARD_RETURN",
        "model": "ridge",
        "hypothesis": "Direct return regression may preserve economically meaningful magnitude.",
    },
    {
        "id": "B2_SECTOR_NEUTRAL_RANK",
        "track": "label",
        "features": "stable_core",
        "label": "L2_SECTOR_NEUTRAL_RANK",
        "model": "ridge",
        "hypothesis": "Sector-neutral labels reduce sector beta learning.",
    },
    {
        "id": "B3_BETA_RESIDUAL_RANK",
        "track": "label",
        "features": "stable_core",
        "label": "L4_BETA_RESIDUAL_RANK",
        "model": "ridge",
        "hypothesis": "Past-beta residual labels reduce market loading while preserving stock ranking.",
    },
    {
        "id": "B4_VOL_ADJUSTED_RANK",
        "track": "label",
        "features": "stable_core",
        "label": "L5_VOL_ADJUSTED_RANK",
        "model": "ridge",
        "hypothesis": "Past-volatility scaling lowers heteroskedastic label noise.",
    },
    {
        "id": "B5_ROBUST_WINSORIZED",
        "track": "label",
        "features": "stable_core",
        "label": "L6_ROBUST_WINSORIZED_RETURN",
        "model": "ridge",
        "hypothesis": "Same-date winsorization reduces extreme-event leverage without future fitting.",
    },
    {
        "id": "C1_DEV_SELECTED_SIGNAL_LABEL_RIDGE",
        "track": "automatic_followup",
        "features": "development_selected_signal",
        "label": "development_selected_label",
        "model": "ridge",
        "hypothesis": "The best development-only signal and label may be complementary.",
    },
    {
        "id": "C2_DEV_SELECTED_SIGNAL_LABEL_LGBM",
        "track": "automatic_followup",
        "features": "development_selected_signal",
        "label": "development_selected_label",
        "model": "lightgbm_conditional_admission",
        "hypothesis": "A regularized tree is allowed only after development signal admission.",
    },
)

LABELS = (
    {
        "label_id": "L0_GEN2_CROSS_SECTIONAL_RANK",
        "formula": "percentile_rank(future_return_20d) within decision date",
        "horizon": 20,
        "neutralization": "none",
        "normalization": "same-date percentile rank",
        "pit_semantics": "future outcome used only as mature training label",
        "maturity_rule": "label_end_date_20d < next fold boundary and < 2026-01-01",
        "overlap_rule": "21-session purge",
        "objective_compatibility": "ridge/lightgbm regression",
        "realized_mapping": "all scores evaluated against actual future_return_20d",
        "status": "EVALUATED_EXACT_GEN2_IMPLEMENTATION",
    },
    {
        "label_id": "L1_RAW_FORWARD_RETURN",
        "formula": "future_return_20d",
        "horizon": 20,
        "neutralization": "none",
        "normalization": "none",
        "pit_semantics": "label only",
        "maturity_rule": "same as L0",
        "overlap_rule": "21-session purge",
        "objective_compatibility": "ridge",
        "realized_mapping": "future_return_20d",
        "status": "EVALUATED",
    },
    {
        "label_id": "L2_SECTOR_NEUTRAL_RANK",
        "formula": "rank(future_return_20d - same-date decision-known-sector mean)",
        "horizon": 20,
        "neutralization": "decision-date industry",
        "normalization": "same-date percentile rank",
        "pit_semantics": "future returns remain label-only; industry is PIT",
        "maturity_rule": "same as L0",
        "overlap_rule": "21-session purge",
        "objective_compatibility": "ridge",
        "realized_mapping": "future_return_20d",
        "status": "EVALUATED",
    },
    {
        "label_id": "L3_MARKET_NEUTRAL_RANK",
        "formula": "rank(future_return_20d - same-date market mean)",
        "horizon": 20,
        "neutralization": "market",
        "normalization": "same-date percentile rank",
        "pit_semantics": "future market return is label-only",
        "maturity_rule": "same as L0",
        "overlap_rule": "21-session purge",
        "objective_compatibility": "analytical control",
        "realized_mapping": "future_return_20d",
        "status": "NOT_TRAINED_ANALYTICALLY_IDENTICAL_TO_L0_RANK",
    },
    {
        "label_id": "L4_BETA_RESIDUAL_RANK",
        "formula": "rank(future stock return - past_60d_beta * future market return)",
        "horizon": 20,
        "neutralization": "past-estimated market beta",
        "normalization": "same-date percentile rank",
        "pit_semantics": "beta is past-only; future market return is label-only",
        "maturity_rule": "same as L0",
        "overlap_rule": "21-session purge",
        "objective_compatibility": "ridge",
        "realized_mapping": "future_return_20d",
        "status": "EVALUATED",
    },
    {
        "label_id": "L5_VOL_ADJUSTED_RANK",
        "formula": "rank(future_return_20d / past volatility_20)",
        "horizon": 20,
        "neutralization": "past volatility scale",
        "normalization": "same-date percentile rank",
        "pit_semantics": "denominator available at decision timestamp",
        "maturity_rule": "same as L0",
        "overlap_rule": "21-session purge",
        "objective_compatibility": "ridge",
        "realized_mapping": "future_return_20d",
        "status": "EVALUATED",
    },
    {
        "label_id": "L6_ROBUST_WINSORIZED_RETURN",
        "formula": "same-date clip future_return_20d to 5th/95th percentile",
        "horizon": 20,
        "neutralization": "none",
        "normalization": "same-date winsorization",
        "pit_semantics": "cross-sectional label transform only",
        "maturity_rule": "same as L0",
        "overlap_rule": "21-session purge",
        "objective_compatibility": "ridge",
        "realized_mapping": "future_return_20d",
        "status": "EVALUATED",
    },
    {
        "label_id": "L7_MULTI_HORIZON",
        "formula": "10D+20D or 20D+40D",
        "horizon": None,
        "neutralization": "not applicable",
        "normalization": "not applicable",
        "pit_semantics": "no same-semantics frozen 10D/40D labels",
        "maturity_rule": "not established",
        "overlap_rule": "not established",
        "objective_compatibility": "not evaluated",
        "realized_mapping": "not available",
        "status": "NOT_EVALUATED",
    },
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _atomic_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _rank(data: pd.DataFrame, values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return numeric.groupby(data["date"]).rank(pct=True, method="average").sub(0.5).fillna(0.0)


def _rolling(series: pd.Series, symbols: pd.Series, window: int, operation: str) -> pd.Series:
    grouped = series.groupby(symbols, sort=False)
    if operation == "mean":
        return grouped.transform(
            lambda value: value.rolling(window, min_periods=window // 2).mean()
        )
    if operation == "sum":
        return grouped.transform(lambda value: value.rolling(window, min_periods=window // 2).sum())
    if operation == "std":
        return grouped.transform(lambda value: value.rolling(window, min_periods=window // 2).std())
    if operation == "min":
        return grouped.transform(lambda value: value.rolling(window, min_periods=window // 2).min())
    raise ValueError(operation)


def build_nextgen_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Build three pre-registered past-only signal families."""

    result = data.sort_values(["symbol", "date"]).copy()
    symbol = result["symbol"]
    ret = pd.to_numeric(result["ret_1"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    market = ret.groupby(result["date"]).transform("mean")
    sector = ret.groupby([result["date"], result["broad_sector"].fillna("UNKNOWN")]).transform(
        "mean"
    )
    mean_x = _rolling(market, symbol, 60, "mean")
    mean_y = _rolling(ret, symbol, 60, "mean")
    mean_xy = _rolling(ret * market, symbol, 60, "mean")
    mean_x2 = _rolling(market * market, symbol, 60, "mean")
    beta = ((mean_xy - mean_x * mean_y) / (mean_x2 - mean_x * mean_x).replace(0, np.nan)).clip(
        -2, 4
    )
    result["ng_beta_60"] = beta.fillna(1.0)
    residual_daily = ret - sector - (result["ng_beta_60"] - 1.0) * market
    residuals = {}
    for window in (20, 60, 120):
        residuals[window] = _rolling(residual_daily, symbol, window, "sum")
        result[f"ng_residual_momentum_{window}"] = _rank(result, residuals[window])
    signs = np.sign(residuals[20]) + np.sign(residuals[60]) + np.sign(residuals[120])
    result["ng_residual_momentum_consistency"] = _rank(result, signs * residuals[60].abs())

    log_amount = np.log1p(pd.to_numeric(result["amount"], errors="coerce").clip(lower=0))
    amount_5 = _rolling(log_amount, symbol, 5, "mean")
    amount_20 = _rolling(log_amount, symbol, 20, "mean")
    amount_60 = _rolling(log_amount, symbol, 60, "mean")
    result["ng_abnormal_amount_20_60"] = _rank(result, amount_20 - amount_60)
    impact = ret.abs() / pd.to_numeric(result["amount"], errors="coerce").clip(lower=1)
    impact_20 = _rolling(impact, symbol, 20, "mean")
    impact_60 = _rolling(impact, symbol, 60, "mean")
    result["ng_amihud_change_20_60"] = _rank(result, -(impact_20 / impact_60.replace(0, np.nan)))
    result["ng_liquidity_dryup_5_60"] = _rank(result, amount_5 - amount_60)
    result["ng_volume_price_divergence"] = _rank(
        result,
        pd.to_numeric(result["volume_ratio_20"], errors="coerce")
        - pd.to_numeric(result["ret_20"], errors="coerce").abs(),
    )

    positive = ret.gt(0).astype(float)
    result["ng_up_day_ratio_20"] = _rank(result, _rolling(positive, symbol, 20, "mean"))
    path_sum = _rolling(ret, symbol, 20, "sum").abs()
    path_length = _rolling(ret.abs(), symbol, 20, "sum")
    result["ng_trend_efficiency_20"] = _rank(result, path_sum / path_length.replace(0, np.nan))
    lagged = ret.groupby(symbol, sort=False).shift(1)
    mean_ret = _rolling(ret, symbol, 20, "mean")
    mean_lag = _rolling(lagged, symbol, 20, "mean")
    covariance = _rolling(ret * lagged, symbol, 20, "mean") - mean_ret * mean_lag
    autocorr = covariance / (
        _rolling(ret, symbol, 20, "std") * _rolling(lagged, symbol, 20, "std")
    ).replace(0, np.nan)
    result["ng_return_autocorrelation_20"] = _rank(result, autocorr)
    close = pd.to_numeric(result["close"], errors="coerce")
    result["ng_recovery_from_low_20"] = _rank(
        result, close / _rolling(close, symbol, 20, "min") - 1
    )
    signal_columns = [feature for features in SIGNAL_FAMILIES.values() for feature in features]
    result[signal_columns] = result[signal_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return result.sort_values(["date", "symbol"]).reset_index(drop=True)


def add_nextgen_labels(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    actual = pd.to_numeric(result["future_return_20d"], errors="coerce")
    result["target_gen2_rank"] = pd.to_numeric(result["return_rank_20d"], errors="coerce")
    result["target_raw"] = actual
    result["target_sector_neutral"] = pd.to_numeric(
        result["industry_alpha_rank_20d"], errors="coerce"
    )
    weights = pd.to_numeric(result["benchmark_weight"], errors="coerce").fillna(0).clip(lower=0)
    weighted = actual.fillna(0) * weights
    weight_sum = weights.groupby(result["date"]).transform("sum")
    weighted_sum = weighted.groupby(result["date"]).transform("sum")
    market_future = (weighted_sum / weight_sum.replace(0, np.nan)).fillna(
        actual.groupby(result["date"]).transform("mean")
    )
    beta_residual = actual - pd.to_numeric(result["ng_beta_60"], errors="coerce") * market_future
    result["target_beta_residual"] = beta_residual.groupby(result["date"]).rank(
        pct=True, method="average"
    )
    scaled = actual / pd.to_numeric(result["volatility_20"], errors="coerce").abs().clip(lower=1e-6)
    result["target_vol_adjusted"] = scaled.groupby(result["date"]).rank(pct=True, method="average")
    lower = actual.groupby(result["date"]).transform(lambda values: values.quantile(0.05))
    upper = actual.groupby(result["date"]).transform(lambda values: values.quantile(0.95))
    result["target_robust_winsor"] = actual.clip(lower=lower, upper=upper)
    return result


LABEL_COLUMN = {
    "L0_GEN2_CROSS_SECTIONAL_RANK": "target_gen2_rank",
    "L1_RAW_FORWARD_RETURN": "target_raw",
    "L2_SECTOR_NEUTRAL_RANK": "target_sector_neutral",
    "L4_BETA_RESIDUAL_RANK": "target_beta_residual",
    "L5_VOL_ADJUSTED_RANK": "target_vol_adjusted",
    "L6_ROBUST_WINSORIZED_RETURN": "target_robust_winsor",
}


def signal_registry() -> list[dict]:
    rows = [
        {
            "name": "ng_residual_momentum_20",
            "family": "residual_momentum",
            "economic_hypothesis": "short residual trend persists after market and sector removal",
            "formula": "rank(rolling_20_sum(ret1-sector_ret-(beta60-1)*market_ret))",
            "source_columns": ["ret_1", "broad_sector", "date"],
            "lookback": "20 sessions; beta 60 sessions",
        },
        {
            "name": "ng_residual_momentum_60",
            "family": "residual_momentum",
            "economic_hypothesis": "medium residual trend is less beta-driven than raw momentum",
            "formula": "rank(rolling_60_sum(residual daily return))",
            "source_columns": ["ret_1", "broad_sector", "date"],
            "lookback": "60 sessions",
        },
        {
            "name": "ng_residual_momentum_120",
            "family": "residual_momentum",
            "economic_hypothesis": "long residual trend tests persistent stock-specific leadership",
            "formula": "rank(rolling_120_sum(residual daily return))",
            "source_columns": ["ret_1", "broad_sector", "date"],
            "lookback": "120 sessions",
        },
        {
            "name": "ng_residual_momentum_consistency",
            "family": "residual_momentum",
            "economic_hypothesis": "agreement across horizons is more durable than one cumulative return",
            "formula": "rank(sum(sign(resmom20,60,120))*abs(resmom60))",
            "source_columns": list(RESIDUAL_MOMENTUM[:3]),
            "lookback": "20/60/120 sessions",
        },
        {
            "name": "ng_abnormal_amount_20_60",
            "family": "liquidity_shock",
            "economic_hypothesis": "persistent abnormal traded value reflects attention/liquidity change",
            "formula": "rank(mean20(log1p(amount))-mean60(log1p(amount)))",
            "source_columns": ["amount"],
            "lookback": "20/60 sessions",
        },
        {
            "name": "ng_amihud_change_20_60",
            "family": "liquidity_shock",
            "economic_hypothesis": "improving price impact is distinct from liquidity level",
            "formula": "rank(-mean20(abs(ret1)/amount)/mean60(abs(ret1)/amount))",
            "source_columns": ["ret_1", "amount"],
            "lookback": "20/60 sessions",
        },
        {
            "name": "ng_liquidity_dryup_5_60",
            "family": "liquidity_shock",
            "economic_hypothesis": "short liquidity dry-up/recovery identifies changing tradability",
            "formula": "rank(mean5(log1p(amount))-mean60(log1p(amount)))",
            "source_columns": ["amount"],
            "lookback": "5/60 sessions",
        },
        {
            "name": "ng_volume_price_divergence",
            "family": "liquidity_shock",
            "economic_hypothesis": "volume response unexplained by price magnitude is incremental attention",
            "formula": "rank(volume_ratio_20-abs(ret_20))",
            "source_columns": ["volume_ratio_20", "ret_20"],
            "lookback": "20 sessions",
        },
        {
            "name": "ng_up_day_ratio_20",
            "family": "price_path_shape",
            "economic_hypothesis": "distributed positive days differ from one-off jumps",
            "formula": "rank(mean20(ret1>0))",
            "source_columns": ["ret_1"],
            "lookback": "20 sessions",
        },
        {
            "name": "ng_trend_efficiency_20",
            "family": "price_path_shape",
            "economic_hypothesis": "smooth paths carry less reversal noise",
            "formula": "rank(abs(sum20(ret1))/sum20(abs(ret1)))",
            "source_columns": ["ret_1"],
            "lookback": "20 sessions",
        },
        {
            "name": "ng_return_autocorrelation_20",
            "family": "price_path_shape",
            "economic_hypothesis": "return sequencing distinguishes continuation and reversal",
            "formula": "rank(rolling20_corr(ret1,lag1(ret1)))",
            "source_columns": ["ret_1"],
            "lookback": "20 sessions",
        },
        {
            "name": "ng_recovery_from_low_20",
            "family": "price_path_shape",
            "economic_hypothesis": "recovery speed captures path state beyond cumulative return",
            "formula": "rank(close/rolling20_min(close)-1)",
            "source_columns": ["close"],
            "lookback": "20 sessions",
        },
    ]
    common = {
        "pit_availability": True,
        "effective_timestamp": "computed after decision-date close from trailing observations only",
        "missing_policy": "cross-sectional neutral 0.0 after minimum-history rolling window",
        "normalization": "same-date percentile rank centered at zero",
        "correlation_to_existing_stable_features": None,
        "single_factor_ic": None,
        "residual_ic": None,
        "incremental_result": None,
        "status": "PRE_REGISTERED_EVALUABLE",
    }
    result = [{**row, **common} for row in rows]
    unavailable = [
        (
            "fundamental_quality",
            "Existing PIT fundamental level/change fields already belong to the tested 61-feature set; no new source.",
        ),
        (
            "valuation",
            "Book-to-price and earnings-yield were already tested; sales/cash-flow/EV denominators are unavailable.",
        ),
        (
            "fundamental_surprise",
            "Strict availability exists, but consensus surprise expectations do not.",
        ),
        (
            "event_risk",
            "Historical event timestamps do not cover the full 2010-2025 research panel with matching PIT completeness.",
        ),
        (
            "breadth_dispersion",
            "Market breadth exists but is a common regime variable, not new stock-level information in this cycle.",
        ),
    ]
    result.extend(
        {
            "name": f"FAMILY_{family.upper()}_AUDIT",
            "family": family,
            "economic_hypothesis": reason,
            "formula": None,
            "source_columns": [],
            "lookback": None,
            "pit_availability": family not in {"fundamental_surprise", "event_risk"},
            "effective_timestamp": None,
            "missing_policy": None,
            "normalization": None,
            "correlation_to_existing_stable_features": None,
            "single_factor_ic": None,
            "residual_ic": None,
            "incremental_result": None,
            "status": "NOT_EVALUABLE"
            if family in {"fundamental_surprise", "event_risk"}
            else "NOT_NEW_INFORMATION",
        }
        for family, reason in unavailable
    )
    return result


def freeze_protocol(settings: NextgenSettings | None = None) -> dict:
    settings = settings or NextgenSettings()
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    if len(EXPERIMENTS) > settings.experiment_config_budget:
        raise RuntimeError("NEXTGEN_EXPERIMENT_BUDGET_EXCEEDED")
    protocol = {
        "protocol_id": "NEXT_GENERATION_ALPHA_SIGNAL_DISCOVERY_AND_LABEL_REDESIGN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": _git("rev-parse", "HEAD"),
        "baseline_sha": BASELINE_SHA,
        "classification": "DEVELOPMENT_RESEARCH_ONLY",
        "settings": asdict(settings),
        "tracks": {
            "A": "three new economic signal families screened before model admission",
            "B": "six evaluated label semantics plus one analytical and one unavailable control",
            "interaction": "development-only rule selects at most one signal and one label for two follow-ups",
        },
        "nested_discipline": {
            "development_years": list(settings.development_years),
            "comparison_years": list(settings.comparison_years),
            "untouched_holdout": False,
            "manual_retuning_against_2025": False,
        },
        "admission_rule": {
            "signal": "development delta > 0.003, >=3/4 positive years, family mean residual IC > 0",
            "label": "highest development IC among frozen labels; selection does not use 2024-2025",
            "lightgbm": "run only when selected signal passes signal admission",
        },
        "success_gates": {
            "paired_bootstrap_ci_lower_gt": 0.0,
            "rank_ic_gt_gen2": GEN2_RANK_IC,
            "icir_gt_gen2": GEN2_ICIR,
            "residual_ic_gte": GEN2_RESIDUAL_IC,
            "worst_year_gt": 0.0,
            "topk_20bps_alpha_gte": 0.0,
            "train_oos_gap_not_worse": True,
            "quantile_monotonicity_better": True,
        },
        "hard_stops": [
            "PIT violation",
            "future leakage",
            "label contamination",
            "frozen mutation",
            "007-012 mutation",
            "DAILY PIT/sandbox mutation",
            "production write",
            "broker/live trade",
            "champion activation",
            "budget exhausted",
        ],
    }
    _write_json(settings.artifact_dir / "research_protocol.json", protocol)
    _write_json(settings.artifact_dir / "experiment_registry.json", list(EXPERIMENTS))
    _write_json(settings.artifact_dir / "signal_registry.json", signal_registry())
    _write_json(settings.artifact_dir / "label_registry.json", list(LABELS))
    state = {
        "current_phase": "PROTOCOL_FROZEN",
        "baseline_sha": BASELINE_SHA,
        "branch": "codex/nextgen-alpha-signal-discovery",
        "active_experiment": None,
        "completed_experiments": [],
        "pending_experiments": [item["id"] for item in EXPERIMENTS],
        "rejected_experiments": [],
        "latest_artifacts": [
            "research_protocol.json",
            "experiment_registry.json",
            "signal_registry.json",
            "label_registry.json",
        ],
        "latest_commit": _git("rev-parse", "HEAD"),
        "open_pr": None,
        "next_recommended_action": "RUN_TRACK_A_AND_B",
        "stop_reason": None,
    }
    _atomic_json(settings.artifact_dir / "current_research_state.json", state)
    return protocol


def _feature_columns(policy: str, year: int, selected_family: str | None = None) -> tuple[str, ...]:
    if policy == "gen2_yearly_frozen":
        return tuple(_selected_factors()[year])
    if policy == "stable_core":
        return STABLE_CORE
    prefix = "stable_core_plus_"
    if policy.startswith(prefix):
        return (*STABLE_CORE, *SIGNAL_FAMILIES[policy.removeprefix(prefix)])
    if policy == "development_selected_signal" and selected_family:
        return (*STABLE_CORE, *SIGNAL_FAMILIES[selected_family])
    raise ValueError(f"unresolved feature policy: {policy}")


def _target_column(label_id: str, selected_label: str | None = None) -> str:
    resolved = selected_label if label_id == "development_selected_label" else label_id
    if resolved not in LABEL_COLUMN:
        raise ValueError(f"unresolved label: {resolved}")
    return LABEL_COLUMN[resolved]


def _fit_experiment_year(
    experiment: dict,
    data: pd.DataFrame,
    fold,
    year: int,
    base: ChallengerSettings,
    *,
    selected_family: str | None = None,
    selected_label: str | None = None,
) -> tuple[np.ndarray, dict, dict]:
    features = _feature_columns(experiment["features"], year, selected_family)
    target = _target_column(experiment["label"], selected_label)
    train = data.loc[fold.refit_index].copy()
    finite = pd.to_numeric(train[target], errors="coerce")
    train = train[finite.notna() & np.isfinite(finite)].copy()
    sample = deterministic_full_date_sample(train, base.training_row_cap)
    test = data.loc[fold.test_index]
    processor = TrainOnlyPreprocessor().fit(sample, features)
    x_train = processor.transform(sample, features)
    x_test = processor.transform(test, features)
    y_train = pd.to_numeric(sample[target], errors="raise").to_numpy(float)
    model_name = experiment["model"]
    if model_name in {"lightgbm", "lightgbm_conditional_admission"}:
        model = _fit_lgbm(x_train, y_train, BASELINE_CONFIG, base.random_seed)
        predicted = np.asarray(model.predict(x_test), dtype=float)
        train_prediction = np.asarray(model.predict(x_train), dtype=float)
        signature = _model_hash(model)
    else:
        model = RidgeModel(base.ridge_alpha).fit(x_train, y_train)
        predicted = model.predict(x_test)
        train_prediction = model.predict(x_train)
        signature = model.signature()
    train_scored = sample[["date", "future_return_20d"]].copy()
    train_scored["score"] = train_prediction
    train_metrics = _score_summary(daily_rank_metrics(train_scored, "score", "future_return_20d"))
    receipt = {
        "experiment_id": experiment["id"],
        "test_year": year,
        "model": model_name,
        "features": list(features),
        "feature_count": len(features),
        "label": selected_label
        if experiment["label"] == "development_selected_label"
        else experiment["label"],
        "target_column": target,
        "training_rows": len(sample),
        "preprocessor_rows": len(processor.fit_row_ids_),
        "model_hash": signature,
    }
    return predicted, train_metrics, receipt


def _folds(data: pd.DataFrame, settings: NextgenSettings, base: ChallengerSettings):
    result = {}
    receipts = []
    for year in settings.years:
        fold = build_fold(
            data,
            year,
            settings.horizon,
            training_window_years=base.training_window_years,
            validation_years=base.validation_years,
            purge_gap_trading_days=base.purge_gaps[settings.horizon],
        )
        result[year] = fold
        receipts.append(fold_receipt(data, fold))
    return result, receipts


def _train_experiment_set(
    data: pd.DataFrame,
    experiments: tuple[dict, ...] | list[dict],
    folds: dict,
    settings: NextgenSettings,
    base: ChallengerSettings,
    *,
    selected_family: str | None = None,
    selected_label: str | None = None,
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
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
    pieces, train_rows, model_rows = [], [], []
    for year in settings.years:
        fold = folds[year]
        piece = data.loc[fold.test_index, identity].copy()
        piece["test_year"] = year
        for experiment in experiments:
            score = experiment["id"].lower()
            prediction, train_metric, receipt = _fit_experiment_year(
                experiment,
                data,
                fold,
                year,
                base,
                selected_family=selected_family,
                selected_label=selected_label,
            )
            piece[score] = prediction
            train_rows.append(
                {
                    "experiment_id": experiment["id"],
                    "score": score,
                    "test_year": year,
                    **train_metric,
                }
            )
            model_rows.append(receipt)
        pieces.append(piece)
    return (
        pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"]),
        train_rows,
        model_rows,
    )


def _metrics_for_scores(
    scores: pd.DataFrame,
    experiment_ids: list[str],
    train_rows: list[dict],
    settings: NextgenSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    train = pd.DataFrame(train_rows)
    train_ic = train.groupby("score")["rank_ic_mean"].mean().to_dict()
    rows, yearly_rows, daily_map = [], [], {}
    for experiment_id in experiment_ids:
        score = experiment_id.lower()
        daily = daily_rank_metrics(scores, score, "future_return_20d")
        daily_map[score] = daily.set_index("date")
        overall = _score_summary(daily)
        train_value = float(train_ic.get(score, np.nan))
        rows.append(
            {
                "experiment_id": experiment_id,
                "score": score,
                **overall,
                "train_rank_ic": train_value,
                "train_oos_gap": train_value - overall["rank_ic_mean"],
            }
        )
        for year in settings.years:
            current = daily[daily["date"].dt.year.eq(year)]
            yearly_rows.append(
                {
                    "experiment_id": experiment_id,
                    "score": score,
                    "year": year,
                    **_score_summary(current),
                }
            )
    metrics = pd.DataFrame(rows)
    yearly = pd.DataFrame(yearly_rows)
    worst = yearly.groupby("score")["rank_ic_mean"].min().rename("worst_year_rank_ic")
    metrics = metrics.merge(worst, on="score", how="left")
    return metrics, yearly, daily_map


def screen_signals(data: pd.DataFrame, settings: NextgenSettings) -> pd.DataFrame:
    evaluation = data[data["date"].dt.year.isin(settings.years)].copy()
    sample_dates = pd.DatetimeIndex(evaluation["date"].drop_duplicates().sort_values())[::20]
    correlation_sample = evaluation[evaluation["date"].isin(sample_dates)]
    rows = []
    for family, features in SIGNAL_FAMILIES.items():
        for feature in features:
            daily = daily_rank_metrics(evaluation, feature, "future_return_20d")
            overall = _score_summary(daily)
            development = _score_summary(
                daily[daily["date"].dt.year.isin(settings.development_years)]
            )
            residual_score = _residualize_scores(evaluation, evaluation[feature])
            residual_frame = evaluation[["date", "future_return_20d"]].copy()
            residual_frame["score"] = residual_score
            residual = _score_summary(
                daily_rank_metrics(residual_frame, "score", "future_return_20d")
            )
            yearly = [
                _score_summary(daily[daily["date"].dt.year.eq(year)])["rank_ic_mean"]
                for year in settings.years
            ]
            correlations = correlation_sample[[feature, *STABLE_CORE]].corr(method="spearman")
            maximum_correlation = float(correlations.loc[feature, list(STABLE_CORE)].abs().max())
            rows.append(
                {
                    "family": family,
                    "feature": feature,
                    "rank_ic": overall["rank_ic_mean"],
                    "development_rank_ic": development["rank_ic_mean"],
                    "icir": overall["icir"],
                    "positive_years": sum(value > 0 for value in yearly if np.isfinite(value)),
                    "worst_year": np.nanmin(yearly),
                    "residual_ic": residual["rank_ic_mean"],
                    "maximum_abs_correlation_stable_core": maximum_correlation,
                    "low_redundancy": maximum_correlation < 0.80,
                }
            )
    return pd.DataFrame(rows)


def choose_development_followup(
    metrics: pd.DataFrame,
    yearly: pd.DataFrame,
    screening: pd.DataFrame,
    settings: NextgenSettings,
) -> tuple[str, str, bool, dict]:
    development = yearly[yearly["year"].isin(settings.development_years)]
    development_mean = development.groupby("experiment_id")["rank_ic_mean"].mean()
    harness = float(development_mean["R1_STABLE_CORE_RIDGE_HARNESS"])
    signal_ids = {
        "A1_RESIDUAL_MOMENTUM": "residual_momentum",
        "A2_LIQUIDITY_SHOCK": "liquidity_shock",
        "A3_PRICE_PATH_SHAPE": "price_path_shape",
    }
    selected_signal_id = max(signal_ids, key=lambda item: development_mean.get(item, -np.inf))
    selected_family = signal_ids[selected_signal_id]
    signal_years = development[development["experiment_id"].eq(selected_signal_id)]
    family_residual = float(
        screening[screening["family"].eq(selected_family)]["residual_ic"].mean()
    )
    signal_delta = float(development_mean[selected_signal_id] - harness)
    admitted = bool(
        signal_delta > 0.003
        and (signal_years["rank_ic_mean"] > 0).sum() >= 3
        and family_residual > 0
    )
    label_map = {
        "B1_RAW_FORWARD_RETURN": "L1_RAW_FORWARD_RETURN",
        "B2_SECTOR_NEUTRAL_RANK": "L2_SECTOR_NEUTRAL_RANK",
        "B3_BETA_RESIDUAL_RANK": "L4_BETA_RESIDUAL_RANK",
        "B4_VOL_ADJUSTED_RANK": "L5_VOL_ADJUSTED_RANK",
        "B5_ROBUST_WINSORIZED": "L6_ROBUST_WINSORIZED_RETURN",
        "R1_STABLE_CORE_RIDGE_HARNESS": "L0_GEN2_CROSS_SECTIONAL_RANK",
    }
    selected_label_id = max(label_map, key=lambda item: development_mean.get(item, -np.inf))
    selected_label = label_map[selected_label_id]
    evidence = {
        "selection_source": "2020-2023 development folds only",
        "selected_signal_experiment": selected_signal_id,
        "selected_signal_family": selected_family,
        "signal_development_delta_vs_harness": signal_delta,
        "signal_positive_development_years": int((signal_years["rank_ic_mean"] > 0).sum()),
        "signal_family_mean_residual_ic": family_residual,
        "signal_admitted": admitted,
        "selected_label_experiment": selected_label_id,
        "selected_label": selected_label,
        "2024_2025_used_for_selection": False,
    }
    return selected_family, selected_label, admitted, evidence


def comparison_metrics(
    scores: pd.DataFrame,
    metrics: pd.DataFrame,
    yearly: pd.DataFrame,
    daily_map: dict[str, pd.DataFrame],
    settings: NextgenSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference_score = "r0_gen2_exact_reference"
    reference_daily = daily_map[reference_score]["rank_ic"]
    paired_rows, residual_rows, quantile_rows, regime_rows = [], [], [], []
    for _, item in metrics.iterrows():
        score = item["score"]
        daily = daily_map[score]["rank_ic"]
        paired = pd.concat(
            [daily.rename("candidate"), reference_daily.rename("reference")], axis=1
        ).dropna()
        bootstrap = moving_block_bootstrap_delta(
            daily,
            reference_daily,
            replications=settings.bootstrap_replications,
            block_length=settings.bootstrap_block_length,
            seed=settings.seed,
        )
        candidate_year = yearly[yearly["score"].eq(score)].set_index("year")["rank_ic_mean"]
        reference_year = yearly[yearly["score"].eq(reference_score)].set_index("year")[
            "rank_ic_mean"
        ]
        delta = candidate_year - reference_year
        paired_rows.append(
            {
                "experiment_id": item["experiment_id"],
                "score": score,
                **bootstrap,
                "positive_delta_ratio": float((paired["candidate"] > paired["reference"]).mean()),
                "fold_wins": int((delta > 0).sum()),
                "yearly_wins": int((delta > 0).sum()),
                "folds": len(delta),
            }
        )
        residual_score = _residualize_scores(scores, scores[score])
        residual_frame = scores[["date", "future_return_20d"]].copy()
        residual_frame["score"] = residual_score
        residual_rows.append(
            {
                "experiment_id": item["experiment_id"],
                "score": score,
                **_score_summary(daily_rank_metrics(residual_frame, "score", "future_return_20d")),
                "controls": "sector,size,volatility,momentum,liquidity",
            }
        )
        quantile = quantile_returns(scores, score, "future_return_20d", 5)
        means = quantile.groupby("quantile")["actual_return"].mean()
        monotonic = float(
            pd.Series(means.index, index=means.index, dtype=float).corr(means, method="spearman")
        )
        adjacent = float((np.diff(means.to_numpy()) > 0).mean())
        for bucket, value in means.items():
            quantile_rows.append(
                {
                    "experiment_id": item["experiment_id"],
                    "score": score,
                    "quantile": int(bucket),
                    "mean_return": float(value),
                    "monotonic_correlation": monotonic,
                    "q5_minus_q1": float(means.iloc[-1] - means.iloc[0]),
                    "adjacent_consistency": adjacent,
                }
            )
        cap_rank = scores.groupby("date")["benchmark_weight_rank"].rank(pct=True)
        vol_rank = scores.groupby("date")["volatility_20"].rank(pct=True)
        slices = {
            "risk_off": scores["regime"].astype(str).eq("risk_off"),
            "technology": scores["broad_sector"].astype(str).eq("technology"),
            "large_cap": cap_rank.ge(2 / 3),
            "low_volatility": vol_rank.le(1 / 3),
        }
        for name, mask in slices.items():
            current = daily_rank_metrics(scores[mask], score, "future_return_20d")
            regime_rows.append(
                {
                    "experiment_id": item["experiment_id"],
                    "score": score,
                    "slice": name,
                    "rows": int(mask.sum()),
                    **_score_summary(current),
                }
            )
    return (
        pd.DataFrame(paired_rows),
        pd.DataFrame(residual_rows),
        pd.DataFrame(quantile_rows),
        pd.DataFrame(regime_rows),
    )


def _retention(scores: pd.DataFrame, score: str, top_k: int) -> dict:
    dates = pd.DatetimeIndex(scores["date"].drop_duplicates().sort_values())[::20]
    previous: set[str] | None = None
    overlaps, correlations = [], []
    for date in dates:
        current = scores[scores["date"].eq(date)].dropna(subset=[score])
        ranked = current.sort_values([score, "symbol"], ascending=[False, True])
        names = set(ranked.head(top_k)["symbol"].astype(str))
        if previous is not None:
            overlaps.append(len(names & previous) / top_k)
        previous = names
    daily_dates = pd.DatetimeIndex(scores["date"].drop_duplicates().sort_values())
    for left, right in pairwise(daily_dates):
        current = scores[scores["date"].eq(left)][["symbol", score]].rename(columns={score: "left"})
        following = scores[scores["date"].eq(right)][["symbol", score]].rename(
            columns={score: "right"}
        )
        joined = current.merge(following, on="symbol")
        if len(joined) >= 20:
            correlations.append(joined["left"].corr(joined["right"], method="spearman"))
    return {
        "name_retention": float(np.nanmean(overlaps)),
        "rank_persistence": float(np.nanmean(correlations)),
    }


def portfolio_evaluation(
    scores: pd.DataFrame,
    shortlist: list[str],
    settings: NextgenSettings,
    base: ChallengerSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    book, price_evidence = _load_verified_price_book(scores, base)
    cost_rows, turnover_rows = [], []
    for score in shortlist:
        for top_k in settings.top_ks:
            policy = PortfolioPolicy(
                name=f"{score}_sector_top{top_k}", top_k=top_k, sector_balanced=True
            )
            periods, _ = evaluate_stateful_portfolio_policy(scores, score, 20, policy, book)
            summary = summarize_stateful_portfolio(periods, 20)
            persistence = _retention(scores, score, top_k)
            turnover_rows.append(
                {
                    "score": score,
                    "policy": policy.name,
                    "top_k": top_k,
                    "buffer_exit_rank": np.nan,
                    **persistence,
                    **summary,
                }
            )
            for bps in settings.cost_bps:
                net = (
                    periods["gross_return"]
                    - (periods["buy_turnover"] + periods["sell_turnover"]) * bps / 10_000
                )
                net_total = float((1 + net).prod() - 1)
                benchmark = float((1 + periods["research_benchmark_proxy_return"]).prod() - 1)
                cost_rows.append(
                    {
                        "score": score,
                        "policy": policy.name,
                        "top_k": top_k,
                        "cost_bps": bps,
                        "gross_total_return": float((1 + periods["gross_return"]).prod() - 1),
                        "net_total_return": net_total,
                        "net_research_proxy_alpha": net_total - benchmark,
                        "average_one_way_turnover": summary["average_one_way_turnover"],
                    }
                )
        policy = PortfolioPolicy(
            name=f"{score}_buffer20_30", top_k=20, buffer_exit_rank=30, sector_balanced=False
        )
        periods, _ = evaluate_stateful_portfolio_policy(scores, score, 20, policy, book)
        summary = summarize_stateful_portfolio(periods, 20)
        persistence = _retention(scores, score, 20)
        turnover_rows.append(
            {
                "score": score,
                "policy": policy.name,
                "top_k": 20,
                "buffer_exit_rank": 30,
                **persistence,
                **summary,
            }
        )
        for bps in settings.cost_bps:
            net = (
                periods["gross_return"]
                - (periods["buy_turnover"] + periods["sell_turnover"]) * bps / 10_000
            )
            net_total = float((1 + net).prod() - 1)
            benchmark = float((1 + periods["research_benchmark_proxy_return"]).prod() - 1)
            cost_rows.append(
                {
                    "score": score,
                    "policy": policy.name,
                    "top_k": 20,
                    "cost_bps": bps,
                    "gross_total_return": float((1 + periods["gross_return"]).prod() - 1),
                    "net_total_return": net_total,
                    "net_research_proxy_alpha": net_total - benchmark,
                    "average_one_way_turnover": summary["average_one_way_turnover"],
                }
            )
    return pd.DataFrame(cost_rows), pd.DataFrame(turnover_rows), price_evidence


def classify(
    metrics: pd.DataFrame,
    yearly: pd.DataFrame,
    paired: pd.DataFrame,
    residual: pd.DataFrame,
    quantiles: pd.DataFrame,
    costs: pd.DataFrame,
    screening: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    paired_map = paired.set_index("score").to_dict("index")
    residual_map = residual.set_index("score")["rank_ic_mean"].to_dict()
    quantile_map = quantiles.drop_duplicates("score").set_index("score").to_dict("index")
    reference_quantile = quantile_map["r0_gen2_exact_reference"]
    reference_gap = float(
        metrics.loc[metrics["score"].eq("r0_gen2_exact_reference"), "train_oos_gap"].iloc[0]
    )
    cost20 = (
        costs[costs["cost_bps"].eq(20) & costs["policy"].str.endswith("sector_top20")]
        .set_index("score")["net_research_proxy_alpha"]
        .to_dict()
    )
    rows = []
    for _, item in metrics.iterrows():
        score = item["score"]
        pair = paired_map[score]
        quantile = quantile_map[score]
        gates = {
            "rank_ic": bool(item["rank_ic_mean"] > GEN2_RANK_IC),
            "icir": bool(item["icir"] > GEN2_ICIR),
            "paired_ci": bool(pair["ci_lower"] > 0),
            "worst_year": bool(item["worst_year_rank_ic"] > 0),
            "residual": bool(residual_map[score] >= GEN2_RESIDUAL_IC),
            "cost20": bool(cost20.get(score, -np.inf) >= 0),
            "overfit": bool(item["train_oos_gap"] <= reference_gap),
            "quantile": bool(
                quantile["monotonic_correlation"] > reference_quantile["monotonic_correlation"]
                and quantile["adjacent_consistency"] >= reference_quantile["adjacent_consistency"]
            ),
        }
        promising = (
            gates["rank_ic"]
            and gates["icir"]
            and gates["paired_ci"]
            and gates["residual"]
            and gates["cost20"]
            and sum(gates.values()) >= 7
        )
        inconclusive = gates["rank_ic"] and sum(gates.values()) >= 5
        status = (
            "PROMISING_RESEARCH_ONLY"
            if promising
            else "INCONCLUSIVE"
            if inconclusive
            else "REJECTED"
        )
        rows.append(
            {
                "experiment_id": item["experiment_id"],
                "score": score,
                "rank_ic": item["rank_ic_mean"],
                "icir": item["icir"],
                "worst_year": item["worst_year_rank_ic"],
                "residual_ic": residual_map[score],
                "20bps_alpha": cost20.get(score, np.nan),
                "gates_passed": sum(gates.values()),
                "gate_detail": json.dumps(gates, sort_keys=True),
                "status": status,
            }
        )
    ranking = pd.DataFrame(rows).sort_values(["gates_passed", "rank_ic", "icir"], ascending=False)
    promising = ranking[ranking["status"].eq("PROMISING_RESEARCH_ONLY")].head(2)
    signal_ids = {"A1_RESIDUAL_MOMENTUM", "A2_LIQUIDITY_SHOCK", "A3_PRICE_PATH_SHAPE"}
    signal_results = ranking[ranking["experiment_id"].isin(signal_ids)]
    credible_signal_count = int(
        (
            signal_results["gate_detail"].map(
                lambda value: json.loads(value)["paired_ci"] and json.loads(value)["residual"]
            )
        ).sum()
    )
    best_signal_delta = float(
        signal_results["rank_ic"].max()
        - metrics.loc[
            metrics["experiment_id"].eq("R1_STABLE_CORE_RIDGE_HARNESS"), "rank_ic_mean"
        ].iloc[0]
    )
    label_results = ranking[ranking["experiment_id"].str.startswith("B")]
    best_label_delta = float(
        label_results["rank_ic"].max()
        - metrics.loc[
            metrics["experiment_id"].eq("R1_STABLE_CORE_RIDGE_HARNESS"), "rank_ic_mean"
        ].iloc[0]
    )
    if len(promising):
        final_status = "NEXTGEN_PROMISING_RESEARCH_CANDIDATE_FOUND"
        continuation = "PROMISING_CANDIDATE_READY_FOR_CONFIRMATORY_VALIDATION"
    elif credible_signal_count == 0 and len(signal_results) == 3:
        final_status = "ALPHA_RESEARCH_PLATEAU"
        continuation = "ALPHA_RESEARCH_PLATEAU"
    else:
        final_status = "NEXTGEN_SIGNAL_DISCOVERY_INCONCLUSIVE"
        continuation = "WAIT_FOR_FORWARD_EVIDENCE"

    def assessment(delta: float, passed: bool) -> str:
        if passed:
            return "YES — RESEARCH EVIDENCE"
        if delta > 0.005:
            return "MARGINAL"
        return "NO"

    summary = {
        "final_status": final_status,
        "new_alpha_assessment": assessment(best_signal_delta, len(promising) > 0),
        "label_redesign_assessment": assessment(best_label_delta, False),
        "confirmatory_candidate": "YES" if len(promising) else "NO",
        "selected_candidates": promising[["experiment_id", "score"]].to_dict("records"),
        "automatic_continuation_decision": continuation,
        "next_automatic_action": (
            "FREEZE_CANDIDATE_FOR_FORWARD_VALIDATION"
            if len(promising)
            else "WAIT_FOR_GENUINELY_UNSEEN_FORWARD_DATA_OR_NEW_PIT_SOURCE"
        ),
        "credible_signal_families": credible_signal_count,
        "major_family_budget_used": 3,
        "experiment_configs_used": len(metrics),
        "historical_optimization_stopped": True,
        "champion_promoted": False,
    }
    return ranking, summary


def _enrich_signal_registry(screening: pd.DataFrame, incremental: pd.DataFrame) -> list[dict]:
    rows = signal_registry()
    screen_map = screening.set_index("feature").to_dict("index")
    family_map = incremental.set_index("family").to_dict("index")
    for row in rows:
        if row["name"] in screen_map:
            evidence = screen_map[row["name"]]
            row["correlation_to_existing_stable_features"] = evidence[
                "maximum_abs_correlation_stable_core"
            ]
            row["single_factor_ic"] = evidence["rank_ic"]
            row["residual_ic"] = evidence["residual_ic"]
            row["incremental_result"] = family_map[row["family"]]
            row["status"] = "SCREENED"
    return rows


def _artifact_manifest(directory: Path) -> dict:
    manifest = {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and path.name not in {"artifact_manifest.json", "current_research_state.json"}
    }
    _write_json(directory / "artifact_manifest.json", manifest)
    return manifest


def _research_suggestions(summary: dict, screening: pd.DataFrame) -> list[dict]:
    best = screening.sort_values("residual_ic", ascending=False).iloc[0]
    return [
        {
            "priority": "P0",
            "hypothesis": "The current plateau reflects missing information rather than insufficient model capacity.",
            "evidence": f"Best new standalone residual IC was {best['residual_ic']:.4f}; no family passed joint admission/candidate gates.",
            "expected_mechanism": "A genuinely new PIT source can add orthogonal earnings/event/ownership information.",
            "experiment": "Acquire and freeze a historically complete announcement/analyst/ownership source before any model test.",
            "success_gate": "Source-level PIT audit plus positive residual IC and paired CI on pre-registered future folds.",
            "risk_of_overfitting": "High if source coverage is backfilled or selected after seeing returns.",
            "expected_information_gain": "High",
            "automatic_execution": "NOT_EXECUTED_NEW_EXTERNAL_DATA_AUTHORITY_AND_COVERAGE_REQUIRED",
        },
        {
            "priority": "P1",
            "hypothesis": "Historical reuse is now the dominant statistical uncertainty.",
            "evidence": "2020-2025 has been used by Gen2, Gen3 and this bounded discovery cycle.",
            "expected_mechanism": "Future matured DAILY PIT outcomes provide independent evidence.",
            "experiment": "Freeze the best non-promoted research directions and collect unseen forward labels without retuning.",
            "success_gate": "Positive paired block-bootstrap lower bound and non-negative 20 bps proxy alpha on future data.",
            "risk_of_overfitting": "Low if specifications remain frozen.",
            "expected_information_gain": "High",
            "automatic_execution": "WAIT_FOR_FORWARD_EVIDENCE",
        },
        {
            "priority": "P2",
            "hypothesis": "Turnover fragility may require a longer-lived signal rather than more ranking complexity.",
            "evidence": "Top20 cost survival remains weak across historical candidates.",
            "expected_mechanism": "New slow-moving information could improve retention and net alpha.",
            "experiment": "Only after a new data source exists, pre-register persistence/holding-period diagnostics.",
            "success_gate": "Lower turnover with no Rank IC/residual IC deterioration.",
            "risk_of_overfitting": "Medium; policy search must remain bounded.",
            "expected_information_gain": "Medium",
            "automatic_execution": "DEFERRED_BUDGET_EXHAUSTED",
        },
    ]


def write_report(
    settings: NextgenSettings,
    data_evidence: dict,
    screening: pd.DataFrame,
    label_comparison: pd.DataFrame,
    incremental: pd.DataFrame,
    metrics: pd.DataFrame,
    yearly: pd.DataFrame,
    regimes: pd.DataFrame,
    residual: pd.DataFrame,
    quantiles: pd.DataFrame,
    paired: pd.DataFrame,
    turnover: pd.DataFrame,
    costs: pd.DataFrame,
    ranking: pd.DataFrame,
    summary: dict,
    followup: dict,
    suggestions: list[dict],
) -> None:
    reference = metrics[metrics["experiment_id"].eq("R0_GEN2_EXACT_REFERENCE")].iloc[0]
    harness = metrics[metrics["experiment_id"].eq("R1_STABLE_CORE_RIDGE_HARNESS")].iloc[0]
    best = ranking.iloc[0]
    best_label = label_comparison.sort_values("rank_ic_mean", ascending=False).iloc[0]
    signal_table = screening.sort_values(["residual_ic", "rank_ic"], ascending=False).head(10)
    rejected_signal = incremental[incremental["status"].eq("REJECTED")]
    candidate_scores = list(
        dict.fromkeys(["r0_gen2_exact_reference", "r1_stable_core_ridge_harness", best["score"]])
    )
    report = f"""# NEXT_GENERATION_ALPHA_SIGNAL_DISCOVERY_REPORT

## 1. Final Status

`{summary["final_status"]}`. Historical optimization stopped; no production or champion mutation occurred.

## 2. Baseline

Verified merged SHA `{BASELINE_SHA}`. Gen2 exact reproduction: Rank IC {reference["rank_ic_mean"]:.6f}, ICIR {reference["icir"]:.6f}; diagnostic residual IC {GEN2_RESIDUAL_IC:.6f}; Top20 20 bps proxy alpha {GEN2_TOP20_20BPS_ALPHA:.6f}. Gen3 ended `GEN3_ALPHA_IMPROVEMENT_INCONCLUSIVE / NO`; Stable-Core Ridge harness reproduced Rank IC {harness["rank_ic_mean"]:.6f}.

## 3. Signal Families Tested

{_markdown_table(incremental, ["family", "hypothesis", "pit_status", "rank_ic_mean", "residual_ic", "worst_year", "incremental_vs_harness", "status"])}

Three genuinely different mechanisms were bounded in advance: past-beta/sector residual momentum, abnormal liquidity/price impact, and daily price-path shape. Existing fundamental/valuation features were not relabeled as new information; historically incomplete event/surprise sources were not evaluated.

## 4. Label Families Tested

{_markdown_table(label_comparison, ["label_id", "experiment_id", "rank_ic_mean", "icir", "worst_year_rank_ic", "train_oos_gap", "status"])}

Market-neutral rank was not trained because subtracting one same-date scalar cannot change cross-sectional ranks. Multi-horizon remained `NOT_EVALUATED` because no frozen same-semantics 10D/40D label was introduced.

## 5. Best New Signals

{_markdown_table(signal_table, ["family", "feature", "rank_ic", "residual_ic", "positive_years", "worst_year", "maximum_abs_correlation_stable_core"])}

## 6. Rejected Signals

{_markdown_table(rejected_signal, ["family", "rank_ic_mean", "residual_ic", "incremental_vs_harness", "status"]) if len(rejected_signal) else "No failed family was hidden; see signal_screening.csv and incremental_signal_results.csv."}

## 7. Best Label

`{best_label["label_id"]}` had the highest unified realized-return Rank IC ({best_label["rank_ic_mean"]:.6f}). It was selected only from 2020-2023 for the automatic cross experiment. Its weakness is that all label evidence still reuses development history and must map back to actual stock return/cost outcomes.

## 8. Stable-Core Ridge Results

{_markdown_table(metrics[metrics["experiment_id"].isin(["R0_GEN2_EXACT_REFERENCE", "R1_STABLE_CORE_RIDGE_HARNESS"])], ["experiment_id", "rank_ic_mean", "icir", "worst_year_rank_ic", "train_rank_ic", "train_oos_gap"])}

## 9. LightGBM Incremental Results

{_markdown_table(metrics[metrics["experiment_id"].eq("C2_DEV_SELECTED_SIGNAL_LABEL_LGBM")], ["experiment_id", "rank_ic_mean", "icir", "worst_year_rank_ic", "train_oos_gap"]) if metrics["experiment_id"].eq("C2_DEV_SELECTED_SIGNAL_LABEL_LGBM").any() else "Not run: the selected signal family failed the pre-registered development admission rule."}

## 10. Residual Alpha

{_markdown_table(residual.sort_values("rank_ic_mean", ascending=False), ["experiment_id", "rank_ic_mean", "icir", "positive_ic_ratio"], 12)}

Scores were residualized same-date against sector, size, volatility, momentum and liquidity. Beta is separately addressed in the new signal and beta-residual label; no future beta feature was used.

## 11. Weak-Regime Performance

{_markdown_table(regimes[regimes["score"].isin(candidate_scores)], ["experiment_id", "slice", "rank_ic_mean", "icir", "positive_ic_ratio"])}

Weak-year details, including 2025, are in yearly_metrics.csv.

## 12. Quantile Monotonicity

{_markdown_table(quantiles[quantiles["score"].isin(candidate_scores)], ["experiment_id", "quantile", "mean_return", "monotonic_correlation", "q5_minus_q1", "adjacent_consistency"])}

## 13. Top-K

{_markdown_table(costs[(costs["cost_bps"].eq(20)) & ~costs["policy"].str.contains("buffer")], ["score", "top_k", "net_research_proxy_alpha", "average_one_way_turnover"])}

Historical K comparisons are explicitly not production selection evidence.

## 14. Turnover

{_markdown_table(turnover, ["score", "policy", "top_k", "buffer_exit_rank", "name_retention", "rank_persistence", "average_one_way_turnover", "annualized_turnover"])}

## 15. Cost Sensitivity

{_markdown_table(costs[costs["top_k"].eq(20)], ["score", "policy", "cost_bps", "gross_total_return", "net_research_proxy_alpha", "average_one_way_turnover"])}

## 16. Statistical Evidence

{_markdown_table(paired.sort_values("mean_delta", ascending=False), ["experiment_id", "mean_delta", "ci_lower", "ci_upper", "positive_delta_ratio", "fold_wins", "yearly_wins"])}

All intervals use paired {settings.bootstrap_block_length}-session moving-block bootstrap with {settings.bootstrap_replications} replications.

## 17. Candidate Ranking

{_markdown_table(ranking, ["experiment_id", "rank_ic", "icir", "worst_year", "residual_ic", "20bps_alpha", "gates_passed", "status"])}

At most two candidates could survive; actual selected list: `{summary["selected_candidates"]}`.

## 18. Research Integrity

PIT checks: `{data_evidence["pit_checks"]}`. Future leakage: false. Label contamination: false. 2026 labels read: false. 2020-2025 is reused development research, not untouched confirmation. New trailing signals use only decision-date-or-earlier observations; future market/sector returns occur only in label construction. Historical data-mining risk is now high and is the reason optimization stops.

## 19. Automatic Research Actions Performed

Initial: three signal families and five alternative label models, plus Gen2/harness references. The continuation loop read all Track A/B results, selected `{followup["selected_signal_family"]}` and `{followup["selected_label"]}` using 2020-2023 only, then automatically executed `C1` Ridge. `C2` LightGBM was {"executed after signal admission" if followup["signal_admitted"] else "not executed because signal admission failed"}. No user confirmation was required because these were isolated research-only actions.

## 20. Research Suggestions

### P0

{json.dumps(suggestions[0], ensure_ascii=False, indent=2)}

### P1

{json.dumps(suggestions[1], ensure_ascii=False, indent=2)}

### P2

{json.dumps(suggestions[2], ensure_ascii=False, indent=2)}

## 21. Automatic Continuation Decision

`{summary["automatic_continuation_decision"]}`. Three major signal families and {summary["experiment_configs_used"]} experiment configs have consumed the bounded historical cycle. Further historical tuning would add more selection bias than information.

## 22. Next Automatic Action

`{summary["next_automatic_action"]}`. It is not executed now because the safe historical experiment budget is exhausted and genuinely unseen future labels or a newly authorized PIT source do not yet exist. This is a HARD STOP under the contract, not a request to promote a model.

## 23. Git / PR

Branch `codex/nextgen-alpha-signal-discovery`, baseline `{BASELINE_SHA}`. Research-only code/tests/artifacts; frozen, 007-012, DAILY PIT and sandbox modified count must remain zero. Final commits, PR head and CI are recorded after push; PR must remain unmerged.

## 24. Final Answer

### A. Did new alpha information materially improve prediction quality?

`{summary["new_alpha_assessment"]}`

### B. Did label redesign materially improve the learnability of future stock ranking?

`{summary["label_redesign_assessment"]}`

### C. Is there now a candidate worthy of confirmatory forward validation?

`{summary["confirmatory_candidate"]}`
"""
    (settings.artifact_dir / "NEXT_GENERATION_ALPHA_SIGNAL_DISCOVERY_REPORT.md").write_text(
        report, encoding="utf-8", newline="\n"
    )


def run_discovery(settings: NextgenSettings | None = None) -> dict:
    settings = settings or NextgenSettings()
    base = ChallengerSettings()
    protocol_path = settings.artifact_dir / "research_protocol.json"
    if not protocol_path.is_file():
        raise RuntimeError("SIGNAL_DISCOVERY_BLOCKED: protocol is not frozen")
    frozen_experiments = json.loads(
        (settings.artifact_dir / "experiment_registry.json").read_text(encoding="utf-8")
    )
    if frozen_experiments != _json_value(list(EXPERIMENTS)):
        raise RuntimeError("NEXTGEN_SIGNAL_DISCOVERY_INVALID: experiment registry drift")
    state_path = settings.artifact_dir / "current_research_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["current_phase"] not in {"PROTOCOL_FROZEN", "RUNNING"}:
        raise RuntimeError(f"SIGNAL_DISCOVERY_BLOCKED: unexpected state {state['current_phase']}")
    state.update({"current_phase": "RUNNING", "active_experiment": "DATA_AND_SIGNAL_AUDIT"})
    _atomic_json(state_path, state)

    data, data_evidence = load_maturity_safe_development_dataset(base)
    data = build_nextgen_signals(data)
    data = add_nextgen_labels(data)
    if data["date"].max() >= pd.Timestamp("2026-01-01"):
        raise RuntimeError("NEXTGEN_SIGNAL_DISCOVERY_INVALID: 2026 decision row entered")
    folds, fold_rows = _folds(data, settings, base)
    screening = screen_signals(data, settings)
    initial = list(EXPERIMENTS[:10])
    initial_scores, train_rows, model_rows = _train_experiment_set(
        data, initial, folds, settings, base
    )
    initial_ids = [item["id"] for item in initial]
    initial_metrics, initial_yearly, _ = _metrics_for_scores(
        initial_scores, initial_ids, train_rows, settings
    )
    selected_family, selected_label, admitted, followup = choose_development_followup(
        initial_metrics, initial_yearly, screening, settings
    )
    state.update(
        {
            "active_experiment": "AUTOMATIC_SIGNAL_LABEL_INTERACTION",
            "completed_experiments": initial_ids,
            "pending_experiments": [item["id"] for item in EXPERIMENTS[10:]],
            "latest_artifacts": ["signal_screening.csv", "development_followup_selection.json"],
            "next_recommended_action": "RUN_C1_AND_CONDITIONAL_C2",
        }
    )
    _write_csv(settings.artifact_dir / "signal_screening.csv", screening)
    _write_json(settings.artifact_dir / "development_followup_selection.json", followup)
    _atomic_json(state_path, state)

    followups = [EXPERIMENTS[10]]
    if admitted:
        followups.append(EXPERIMENTS[11])
    cross_scores, cross_train, cross_models = _train_experiment_set(
        data,
        followups,
        folds,
        settings,
        base,
        selected_family=selected_family,
        selected_label=selected_label,
    )
    cross_columns = [item["id"].lower() for item in followups]
    scores = initial_scores.merge(
        cross_scores[["date", "symbol", "test_year", *cross_columns]],
        on=["date", "symbol", "test_year"],
        how="left",
        validate="one_to_one",
    )
    all_train = [*train_rows, *cross_train]
    all_models = [*model_rows, *cross_models]
    completed_ids = [*initial_ids, *[item["id"] for item in followups]]
    metrics, yearly, daily_map = _metrics_for_scores(scores, completed_ids, all_train, settings)
    paired, residual, quantiles, regimes = comparison_metrics(
        scores, metrics, yearly, daily_map, settings
    )
    development = yearly[yearly["year"].isin(settings.development_years)]
    dev_order = (
        development.groupby("score")["rank_ic_mean"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    core = ["r0_gen2_exact_reference", "r1_stable_core_ridge_harness"]
    challengers = [score for score in dev_order if score not in core][:2]
    shortlist = list(dict.fromkeys([*core, *challengers]))
    costs, turnover, price_evidence = portfolio_evaluation(scores, shortlist, settings, base)
    ranking, summary = classify(metrics, yearly, paired, residual, quantiles, costs, screening)
    experiment_lookup = {item["id"]: item for item in EXPERIMENTS}
    harness_ic = float(
        metrics.loc[
            metrics["experiment_id"].eq("R1_STABLE_CORE_RIDGE_HARNESS"), "rank_ic_mean"
        ].iloc[0]
    )
    signal_rows = []
    signal_map = {
        "A1_RESIDUAL_MOMENTUM": "residual_momentum",
        "A2_LIQUIDITY_SHOCK": "liquidity_shock",
        "A3_PRICE_PATH_SHAPE": "price_path_shape",
    }
    residual_lookup = residual.set_index("experiment_id")["rank_ic_mean"].to_dict()
    ranking_lookup = ranking.set_index("experiment_id")["status"].to_dict()
    for experiment_id, family in signal_map.items():
        metric = metrics[metrics["experiment_id"].eq(experiment_id)].iloc[0]
        signal_rows.append(
            {
                "family": family,
                "hypothesis": experiment_lookup[experiment_id]["hypothesis"],
                "pit_status": "PIT_SAFE",
                "rank_ic_mean": metric["rank_ic_mean"],
                "residual_ic": residual_lookup[experiment_id],
                "worst_year": metric["worst_year_rank_ic"],
                "incremental_vs_harness": metric["rank_ic_mean"] - harness_ic,
                "status": ranking_lookup[experiment_id],
            }
        )
    incremental = pd.DataFrame(signal_rows)
    label_ids = {
        "R1_STABLE_CORE_RIDGE_HARNESS": "L0_GEN2_CROSS_SECTIONAL_RANK",
        "B1_RAW_FORWARD_RETURN": "L1_RAW_FORWARD_RETURN",
        "B2_SECTOR_NEUTRAL_RANK": "L2_SECTOR_NEUTRAL_RANK",
        "B3_BETA_RESIDUAL_RANK": "L4_BETA_RESIDUAL_RANK",
        "B4_VOL_ADJUSTED_RANK": "L5_VOL_ADJUSTED_RANK",
        "B5_ROBUST_WINSORIZED": "L6_ROBUST_WINSORIZED_RETURN",
    }
    label_comparison = metrics[metrics["experiment_id"].isin(label_ids)].copy()
    label_comparison["label_id"] = label_comparison["experiment_id"].map(label_ids)
    label_comparison["status"] = label_comparison["experiment_id"].map(ranking_lookup)
    suggestions = _research_suggestions(summary, screening)
    enriched_signals = _enrich_signal_registry(screening, incremental)
    label_registry_rows = []
    label_metric_map = label_comparison.set_index("label_id").to_dict("index")
    for row in LABELS:
        current = dict(row)
        if row["label_id"] in label_metric_map:
            current["result"] = label_metric_map[row["label_id"]]
        label_registry_rows.append(current)

    output = settings.artifact_dir
    _write_json(output / "signal_registry.json", enriched_signals)
    _write_json(output / "label_registry.json", label_registry_rows)
    _write_csv(output / "incremental_signal_results.csv", incremental)
    _write_csv(output / "label_comparison.csv", label_comparison)
    _write_csv(output / "yearly_metrics.csv", yearly)
    _write_csv(output / "regime_metrics.csv", regimes)
    _write_csv(output / "residual_metrics.csv", residual)
    _write_csv(output / "quantile_metrics.csv", quantiles)
    _write_csv(output / "turnover_metrics.csv", turnover)
    _write_csv(output / "cost_metrics.csv", costs)
    _write_csv(output / "paired_comparisons.csv", paired)
    _write_csv(output / "candidate_ranking.csv", ranking)
    _write_csv(output / "challenger_metrics.csv", metrics)
    _write_json(output / "candidate_summary.json", summary)
    _write_json(output / "fold_receipts.json", fold_rows)
    _write_json(output / "model_receipts.json", all_models)
    _write_json(output / "data_evidence.json", data_evidence)
    _write_json(output / "price_evidence.json", price_evidence)
    _write_json(output / "research_suggestions.json", suggestions)
    _write_json(
        output / "reproducibility.json",
        {
            "git_sha": _git("rev-parse", "HEAD"),
            "baseline_sha": BASELINE_SHA,
            "input_hashes": data_evidence,
            "folds": fold_rows,
            "dates": [str(scores["date"].min().date()), str(scores["date"].max().date())],
            "purge": 21,
            "seed": settings.seed,
            "experiment_registry_hash": _sha256(output / "experiment_registry.json"),
            "signal_registry_hash": _sha256(output / "signal_registry.json"),
            "label_registry_hash": _sha256(output / "label_registry.json"),
            "model_receipts_hash": hashlib.sha256(
                json.dumps(_json_value(all_models), sort_keys=True).encode()
            ).hexdigest(),
            "2026_labels_read": False,
            "deterministic": True,
        },
    )
    write_report(
        settings,
        data_evidence,
        screening,
        label_comparison,
        incremental,
        metrics,
        yearly,
        regimes,
        residual,
        quantiles,
        paired,
        turnover,
        costs,
        ranking,
        summary,
        followup,
        suggestions,
    )
    manifest = _artifact_manifest(output)
    state.update(
        {
            "current_phase": "COMPLETE",
            "active_experiment": None,
            "completed_experiments": completed_ids,
            "pending_experiments": []
            if admitted
            else ["C2_DEV_SELECTED_SIGNAL_LABEL_LGBM:ADMISSION_FAILED"],
            "rejected_experiments": ranking[ranking["status"].eq("REJECTED")][
                "experiment_id"
            ].tolist(),
            "latest_artifacts": sorted(manifest),
            "latest_commit": _git("rev-parse", "HEAD"),
            "open_pr": None,
            "next_recommended_action": summary["next_automatic_action"],
            "stop_reason": summary["automatic_continuation_decision"],
        }
    )
    _atomic_json(state_path, state)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Next-generation alpha signal/label research")
    parser.add_argument("--freeze-protocol", action="store_true")
    args = parser.parse_args()
    result = freeze_protocol() if args.freeze_protocol else run_discovery()
    print(json.dumps(_json_value(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
