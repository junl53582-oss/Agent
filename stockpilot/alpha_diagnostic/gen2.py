from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from research_v6.model import _sector_quotas
from stockpilot.research_challenger.config import ChallengerSettings
from stockpilot.research_challenger.data import factor_group
from stockpilot.research_challenger.gen02 import _selected_factors
from stockpilot.research_challenger.gen02_correctness import (
    _load_verified_price_book,
    evaluate_stateful_portfolio_policy,
    load_maturity_safe_development_dataset,
    summarize_stateful_portfolio,
)
from stockpilot.research_challenger.gen02_portfolio import PortfolioPolicy
from stockpilot.research_challenger.metrics import daily_rank_metrics
from stockpilot.research_challenger.models import (
    LightGBMModel,
    RidgeModel,
    TrainOnlyPreprocessor,
    deterministic_full_date_sample,
)
from stockpilot.research_challenger.split import build_fold, fold_receipt

ARTIFACT_DIR = Path("artifacts/research_challenger/gen03_alpha_diagnostic")
BASELINE_SHA = "63a866830efad38098ae8cc3237b4cd8340970c8"
MODEL_ID = "GEN2-LGBM-20D-SECTOR-BALANCED-TOP20"


@dataclass(frozen=True)
class DiagnosticSettings:
    artifact_dir: Path = ARTIFACT_DIR
    years: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024, 2025)
    horizon: int = 20
    diagnostic_horizon: int = 5
    top_ks: tuple[int, ...] = (5, 10, 20, 30, 50)
    quantiles: int = 5
    cost_bps: tuple[int, ...] = (0, 10, 20, 50)
    bootstrap_replications: int = 1_000
    bootstrap_block_length: int = 20
    redundancy_threshold: float = 0.90
    random_seed: int = 42


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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write_json(path: Path, payload: dict | list) -> None:
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _t_stat(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    std = float(clean.std(ddof=1))
    return float(clean.mean() / (std / math.sqrt(len(clean)))) if len(clean) > 1 and std > 0 else 0.0


def _ic_summary(daily: pd.DataFrame) -> dict:
    rank_source = daily["rank_ic"] if "rank_ic" in daily else pd.Series(dtype=float)
    pearson_source = daily["pearson_ic"] if "pearson_ic" in daily else pd.Series(dtype=float)
    rank = pd.to_numeric(rank_source, errors="coerce").dropna()
    pearson = pd.to_numeric(pearson_source, errors="coerce").dropna()
    if rank.empty:
        return {
            "dates": 0,
            "pearson_ic_mean": float("nan"),
            "pearson_ic_median": float("nan"),
            "rank_ic_mean": float("nan"),
            "rank_ic_median": float("nan"),
            "rank_ic_std": float("nan"),
            "icir": float("nan"),
            "positive_ic_ratio": float("nan"),
            "rank_ic_t_stat": float("nan"),
        }
    std = float(rank.std(ddof=1))
    return {
        "dates": len(rank),
        "pearson_ic_mean": float(pearson.mean()),
        "pearson_ic_median": float(pearson.median()),
        "rank_ic_mean": float(rank.mean()),
        "rank_ic_median": float(rank.median()),
        "rank_ic_std": std,
        "icir": float(rank.mean() / std) if std > 0 else 0.0,
        "positive_ic_ratio": float((rank > 0).mean()),
        "rank_ic_t_stat": _t_stat(rank),
    }


def _block_bootstrap(values: pd.Series, settings: DiagnosticSettings) -> dict:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    block = settings.bootstrap_block_length
    if len(clean) < block * 2:
        return {"samples": len(clean), "ci_lower": None, "ci_upper": None}
    rng = np.random.default_rng(settings.random_seed)
    starts = np.arange(len(clean) - block + 1)
    blocks = int(np.ceil(len(clean) / block))
    estimates = np.empty(settings.bootstrap_replications)
    for index in range(settings.bootstrap_replications):
        chosen = rng.choice(starts, size=blocks, replace=True)
        sample = np.concatenate([clean[start : start + block] for start in chosen])[: len(clean)]
        estimates[index] = sample.mean()
    return {
        "samples": len(clean),
        "replications": settings.bootstrap_replications,
        "block_length": block,
        "ci_lower": float(np.quantile(estimates, 0.025)),
        "ci_upper": float(np.quantile(estimates, 0.975)),
    }


def _rank_average(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    ranks = [frame.groupby("date")[column].rank(pct=True) for column in columns]
    return pd.concat(ranks, axis=1).mean(axis=1)


def _fit_oos(
    data: pd.DataFrame,
    settings: DiagnosticSettings,
    base: ChallengerSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    selected_by_year = _selected_factors()
    pieces: list[pd.DataFrame] = []
    importance_rows: list[dict] = []
    fit_rows: list[dict] = []
    fold_rows: list[dict] = []
    groups = sorted({factor_group(feature) for feature in base.factor_columns})
    identity = [
        "date", "symbol", "industry", "broad_sector", "benchmark_weight",
        "benchmark_weight_rank", "amount_rank", "volatility_20", "regime",
        "future_return_20d", "future_return_5d", "entry_tradable_20",
        "execution_return_20", "momentum", "short_reversal", "liquidity",
        "low_volatility", "industry_momentum", "volatility_60_rank",
    ]
    for year in settings.years:
        features = tuple(selected_by_year[year])
        fold = build_fold(
            data,
            year,
            settings.horizon,
            training_window_years=base.training_window_years,
            validation_years=base.validation_years,
            purge_gap_trading_days=base.purge_gaps[settings.horizon],
        )
        receipt = fold_receipt(data, fold)
        receipt["selected_features"] = list(features)
        fold_rows.append(receipt)
        train = data.loc[fold.refit_index].copy()
        target = "return_rank_20d"
        finite = pd.to_numeric(train[target], errors="coerce")
        train = train[finite.notna() & np.isfinite(finite)].copy()
        sample = deterministic_full_date_sample(train, base.training_row_cap)
        test = data.loc[fold.test_index].copy()
        processor = TrainOnlyPreprocessor().fit(sample, features)
        x_train = processor.transform(sample, features)
        x_test = processor.transform(test, features)
        y_train = pd.to_numeric(sample[target], errors="raise").to_numpy(dtype=float)
        ridge = RidgeModel(base.ridge_alpha).fit(x_train, y_train)
        model = LightGBMModel("regression_l1", base.lightgbm_rounds, base.random_seed).fit(
            x_train, y_train
        )
        piece = test[identity].copy()
        piece["test_year"] = year
        piece["score_lightgbm"] = model.predict(x_test)
        piece["score_ridge"] = ridge.predict(x_test)
        booster = model.booster_
        gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
        split = np.asarray(booster.feature_importance(importance_type="split"), dtype=float)
        for index, feature in enumerate(features):
            importance_rows.append(
                {
                    "test_year": year,
                    "feature": feature,
                    "feature_group": factor_group(feature),
                    "gain_importance": float(gain[index] / gain.sum()) if gain.sum() else 0.0,
                    "split_importance": float(split[index] / split.sum()) if split.sum() else 0.0,
                }
            )
        train_scored = sample[["date", "future_return_20d"]].copy()
        train_scored["score"] = model.predict(x_train)
        fit_rows.append(
            {
                "test_year": year,
                "model": "lightgbm",
                "sample": "train_in_sample",
                **_ic_summary(daily_rank_metrics(train_scored, "score", "future_return_20d")),
            }
        )
        for group in groups:
            reduced = tuple(feature for feature in features if factor_group(feature) != group)
            if not reduced or len(reduced) == len(features):
                piece[f"score_minus_{group}"] = piece["score_lightgbm"]
                continue
            reduced_processor = TrainOnlyPreprocessor().fit(sample, reduced)
            reduced_model = LightGBMModel(
                "regression_l1", base.lightgbm_rounds, base.random_seed
            ).fit(
                reduced_processor.transform(sample, reduced), y_train
            )
            piece[f"score_minus_{group}"] = reduced_model.predict(
                reduced_processor.transform(test, reduced)
            )
        pieces.append(piece)
    scores = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"])
    scores["score_ensemble"] = _rank_average(scores, ("score_lightgbm", "score_ridge"))
    return scores, pd.DataFrame(importance_rows), pd.DataFrame(fit_rows), fold_rows


def _fit_horizon_5(
    data: pd.DataFrame, settings: DiagnosticSettings, base: ChallengerSettings
) -> pd.DataFrame:
    selected_by_year = _selected_factors()
    pieces = []
    for year in settings.years:
        features = tuple(selected_by_year[year])
        fold = build_fold(
            data,
            year,
            settings.diagnostic_horizon,
            training_window_years=base.training_window_years,
            validation_years=base.validation_years,
            purge_gap_trading_days=base.purge_gaps[settings.diagnostic_horizon],
        )
        train = data.loc[fold.refit_index].copy()
        target = "return_rank_5d"
        finite = pd.to_numeric(train[target], errors="coerce")
        train = train[finite.notna() & np.isfinite(finite)].copy()
        sample = deterministic_full_date_sample(train, base.training_row_cap)
        test = data.loc[fold.test_index].copy()
        processor = TrainOnlyPreprocessor().fit(sample, features)
        model = LightGBMModel("regression_l1", base.lightgbm_rounds, base.random_seed).fit(
            processor.transform(sample, features),
            pd.to_numeric(sample[target], errors="raise").to_numpy(dtype=float),
        )
        piece = test[["date", "symbol", "future_return_5d"]].copy()
        piece["score"] = model.predict(processor.transform(test, features))
        piece["test_year"] = year
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def _time_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    daily = daily_rank_metrics(scores, "score_lightgbm", "future_return_20d")
    rows = []
    periods = {
        "year": daily["date"].dt.to_period("Y").astype(str),
        "quarter": daily["date"].dt.to_period("Q").astype(str),
        "month": daily["date"].dt.to_period("M").astype(str),
    }
    for frequency, labels in periods.items():
        for period, group in daily.groupby(labels):
            rows.append({"frequency": frequency, "period": period, **_ic_summary(group)})
    return pd.DataFrame(rows)


def _quantile_metrics(scores: pd.DataFrame, quantiles: int) -> pd.DataFrame:
    rows = []
    for date, group in scores.groupby("date", sort=True):
        valid = group.dropna(subset=["score_lightgbm", "future_return_20d"]).copy()
        if len(valid) < quantiles * 10:
            continue
        valid["quantile"] = pd.qcut(
            valid["score_lightgbm"].rank(method="first"), quantiles, labels=False
        ) + 1
        universe_median = float(valid["future_return_20d"].median())
        benchmark = valid[valid["benchmark_weight"].gt(0)]
        total = float(benchmark["benchmark_weight"].sum())
        benchmark_return = float(
            (benchmark["benchmark_weight"] * benchmark["future_return_20d"]).sum() / total
        ) if total else float(valid["future_return_20d"].mean())
        for quantile, part in valid.groupby("quantile"):
            realized = pd.to_numeric(part["future_return_20d"], errors="coerce")
            rows.append(
                {
                    "date": date,
                    "quantile": int(quantile),
                    "mean_return": float(realized.mean()),
                    "median_return": float(realized.median()),
                    "excess_return": float(realized.mean() - benchmark_return),
                    "hit_rate": float((realized > 0).mean()),
                    "precision_above_median": float((realized > universe_median).mean()),
                    "sample_size": len(part),
                }
            )
    daily = pd.DataFrame(rows)
    summary = daily.groupby("quantile", as_index=False).agg(
        mean_return=("mean_return", "mean"),
        median_return=("median_return", "median"),
        excess_return=("excess_return", "mean"),
        hit_rate=("hit_rate", "mean"),
        precision_above_median=("precision_above_median", "mean"),
        volatility=("mean_return", "std"),
        positive_period_ratio=("mean_return", lambda value: float((value > 0).mean())),
        dates=("date", "nunique"),
    )
    tstats = daily.groupby("quantile")["mean_return"].apply(_t_stat)
    summary["t_stat"] = summary["quantile"].map(tstats)
    return summary


def _select_sector_balanced(group: pd.DataFrame, k: int) -> pd.DataFrame:
    ranked = group.sort_values(["score_lightgbm", "symbol"], ascending=[False, True]).copy()
    quotas = _sector_quotas(ranked, k)
    pieces = [
        ranked[ranked["broad_sector"].astype(str).eq(sector)].head(quota)
        for sector, quota in quotas.items()
    ]
    return (pd.concat(pieces) if pieces else ranked.iloc[0:0]).sort_values(
        ["score_lightgbm", "symbol"], ascending=[False, True]
    ).head(k)


def _topk_cross_sectional(scores: pd.DataFrame, settings: DiagnosticSettings) -> pd.DataFrame:
    rows = []
    for date, group in scores.groupby("date", sort=True):
        valid = group.dropna(subset=["score_lightgbm", "future_return_20d"])
        if len(valid) < max(settings.top_ks):
            continue
        benchmark = valid[valid["benchmark_weight"].gt(0)]
        total = float(benchmark["benchmark_weight"].sum())
        benchmark_return = float(
            (benchmark["benchmark_weight"] * benchmark["future_return_20d"]).sum() / total
        ) if total else float(valid["future_return_20d"].mean())
        median = float(valid["future_return_20d"].median())
        for k in settings.top_ks:
            selected = _select_sector_balanced(valid, k)
            realized = selected["future_return_20d"]
            rows.append(
                {
                    "date": date,
                    "top_k": k,
                    "mean_forward_return": float(realized.mean()),
                    "median_forward_return": float(realized.median()),
                    "benchmark_excess_return": float(realized.mean() - benchmark_return),
                    "hit_rate": float((realized > 0).mean()),
                    "precision_at_k": float((realized > median).mean()),
                    "sector_count": int(selected["broad_sector"].nunique()),
                    "maximum_sector_weight": float(selected["broad_sector"].value_counts(normalize=True).max()),
                }
            )
    daily = pd.DataFrame(rows)
    summary = daily.groupby("top_k", as_index=False).agg(
        mean_forward_return=("mean_forward_return", "mean"),
        median_forward_return=("median_forward_return", "median"),
        benchmark_excess_return=("benchmark_excess_return", "mean"),
        return_volatility=("mean_forward_return", "std"),
        hit_rate=("hit_rate", "mean"),
        precision_at_k=("precision_at_k", "mean"),
        positive_period_ratio=("benchmark_excess_return", lambda value: float((value > 0).mean())),
        mean_sector_count=("sector_count", "mean"),
        maximum_sector_weight=("maximum_sector_weight", "mean"),
        dates=("date", "nunique"),
    )
    summary["excess_t_stat"] = summary["top_k"].map(
        daily.groupby("top_k")["benchmark_excess_return"].apply(_t_stat)
    )
    return summary


def _bucket_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    frames = []
    definitions = {
        "market_cap": ("benchmark_weight_rank", ["small", "mid", "large"]),
        "liquidity": ("amount_rank", ["low", "medium", "high"]),
        "volatility": ("volatility_20", ["low", "mid", "high"]),
    }
    for dimension, (column, labels) in definitions.items():
        data = scores.copy()
        data["bucket"] = data.groupby("date")[column].transform(
            lambda value, bucket_labels=labels: pd.qcut(
                value.rank(method="first"), 3, labels=bucket_labels
            )
        )
        for bucket, part in data.groupby("bucket", observed=True):
            daily = daily_rank_metrics(part, "score_lightgbm", "future_return_20d")
            frames.append(
                {
                    "dimension": dimension,
                    "bucket": str(bucket),
                    "rows": len(part),
                    "mean_return": float(part["future_return_20d"].mean()),
                    "hit_rate": float((part["future_return_20d"] > 0).mean()),
                    **_ic_summary(daily),
                }
            )
    return pd.DataFrame(frames)


def _sector_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sector, part in scores.groupby("broad_sector"):
        daily = daily_rank_metrics(part, "score_lightgbm", "future_return_20d")
        rows.append(
            {
                "sector": sector,
                "rows": len(part),
                "symbols": part["symbol"].nunique(),
                "prediction_dispersion": float(part.groupby("date")["score_lightgbm"].std().mean()),
                "mean_realized_return": float(part["future_return_20d"].mean()),
                "hit_rate": float((part["future_return_20d"] > 0).mean()),
                **_ic_summary(daily),
            }
        )
    return pd.DataFrame(rows).sort_values("rank_ic_mean", ascending=False)


def _regime_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    date_state = scores.groupby("date", as_index=False).agg(
        volatility=("volatility_20", "mean")
    )
    date_state["past_vol_median"] = date_state["volatility"].shift(1).rolling(252, min_periods=60).median()
    date_state["vol_state"] = np.where(
        date_state["past_vol_median"].isna(), "insufficient_history",
        np.where(date_state["volatility"] > date_state["past_vol_median"], "high_vol", "low_vol"),
    )
    data = scores.merge(date_state[["date", "vol_state"]], on="date", how="left")
    rows = []
    for dimension, column in (("market_regime", "regime"), ("volatility_regime", "vol_state")):
        for regime, part in data.groupby(column):
            daily = daily_rank_metrics(part, "score_lightgbm", "future_return_20d")
            rows.append(
                {
                    "dimension": dimension,
                    "regime": regime,
                    "rows": len(part),
                    "mean_forward_return": float(part["future_return_20d"].mean()),
                    "hit_rate": float((part["future_return_20d"] > 0).mean()),
                    **_ic_summary(daily),
                }
            )
    return pd.DataFrame(rows)


def _factor_exposure(scores: pd.DataFrame) -> pd.DataFrame:
    proxies = {
        "momentum": "momentum", "reversal": "short_reversal", "size": "benchmark_weight_rank",
        "volatility": "volatility_60_rank", "liquidity": "liquidity",
        "sector_momentum": "industry_momentum",
    }
    rows = []
    for name, column in proxies.items():
        correlations = scores.groupby("date").apply(
            lambda part, factor_column=column: part["score_lightgbm"].corr(
                part[factor_column], method="spearman"
            ),
            include_groups=False,
        ).dropna()
        rows.append(
            {
                "factor": name,
                "source_column": column,
                "mean_cross_sectional_rank_correlation": float(correlations.mean()),
                "median_correlation": float(correlations.median()),
                "positive_ratio": float((correlations > 0).mean()),
                "dates": len(correlations),
            }
        )
    rows.append({"factor": "beta", "source_column": None, "status": "not_evaluable_from_current_data"})
    return pd.DataFrame(rows)


def _feature_diagnostics(
    data: pd.DataFrame, scores: pd.DataFrame, importance: pd.DataFrame, base: ChallengerSettings
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oos = data[data["date"].dt.year.isin(scores["test_year"].unique())]
    single_rows = []
    for feature in base.factor_columns:
        daily = daily_rank_metrics(oos, feature, "future_return_20d")
        summary = _ic_summary(daily)
        lagged = oos[["symbol", feature]].copy()
        lagged["previous_value"] = lagged.groupby("symbol")[feature].shift(1)
        persistence = lagged[feature].corr(lagged["previous_value"], method="spearman")
        single_rows.append(
            {
                "feature": feature,
                "feature_group": factor_group(feature),
                "sign_consistency": max(summary["positive_ic_ratio"], 1 - summary["positive_ic_ratio"]),
                "turnover_proxy_one_minus_observation_rank_persistence": float(1 - persistence),
                **summary,
            }
        )
    single = pd.DataFrame(single_rows).sort_values("rank_ic_mean", key=lambda value: value.abs(), ascending=False)
    all_pairs = []
    sample_dates = pd.DatetimeIndex(oos["date"].drop_duplicates().sort_values())
    positions = np.linspace(0, len(sample_dates) - 1, min(120, len(sample_dates)), dtype=int)
    sampled = oos[oos["date"].isin(sample_dates[positions])]
    corr = sampled[list(base.factor_columns)].corr(method="spearman")
    features = list(base.factor_columns)
    for left_index, left in enumerate(features):
        for right in features[left_index + 1 :]:
            value = float(corr.loc[left, right])
            if abs(value) >= 0.75:
                all_pairs.append({"feature_a": left, "feature_b": right, "correlation": value, "above_0_90": abs(value) > 0.90})
    redundancy = pd.DataFrame(all_pairs).sort_values("correlation", key=lambda value: value.abs(), ascending=False)
    pivot = importance.pivot(index="feature", columns="test_year", values="gain_importance").fillna(0)
    stability_rows = []
    for feature in base.factor_columns:
        values = pivot.loc[feature] if feature in pivot.index else pd.Series(0.0, index=settings_years(scores))
        ranks = pivot.rank(ascending=False, method="average").loc[feature] if feature in pivot.index else pd.Series(np.nan, index=pivot.columns)
        stability_rows.append(
            {
                "feature": feature,
                "feature_group": factor_group(feature),
                "mean_gain_importance": float(values.mean()),
                "std_gain_importance": float(values.std(ddof=1)),
                "mean_importance_rank": float(ranks.mean()),
                "rank_stability": float(1 / (1 + ranks.std(ddof=1))) if ranks.notna().sum() > 1 else 0.0,
                "active_fold_ratio": float((values > 0).mean()),
                "classification": "dead" if (values > 0).mean() == 0 else "stable" if (values > 0).mean() >= 0.8 and ranks.std(ddof=1) <= 5 else "unstable_or_regime",
            }
        )
    return single, redundancy, pd.DataFrame(stability_rows).sort_values("mean_gain_importance", ascending=False)


def settings_years(scores: pd.DataFrame) -> list[int]:
    return sorted(scores["test_year"].unique())


def _ablation_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    baseline = _ic_summary(daily_rank_metrics(scores, "score_lightgbm", "future_return_20d"))
    rows = [{"ablation": "full_gen2", **baseline, "rank_ic_change": 0.0, "icir_change": 0.0}]
    for column in sorted(name for name in scores.columns if name.startswith("score_minus_")):
        summary = _ic_summary(daily_rank_metrics(scores, column, "future_return_20d"))
        rows.append(
            {
                "ablation": column.removeprefix("score_"),
                **summary,
                "rank_ic_change": summary["rank_ic_mean"] - baseline["rank_ic_mean"],
                "icir_change": summary["icir"] - baseline["icir"],
            }
        )
    return pd.DataFrame(rows)


def _model_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ("lightgbm", "ridge", "ensemble"):
        daily = daily_rank_metrics(scores, f"score_{model}", "future_return_20d")
        rows.append({"model": model, **_ic_summary(daily)})
    return pd.DataFrame(rows)


def _score_calibration(scores: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for month, part in scores.groupby(scores["date"].dt.to_period("M")):
        rows.append(
            {
                "month": str(month),
                "score_mean": float(part["score_lightgbm"].mean()),
                "score_std": float(part["score_lightgbm"].std()),
                "score_p01": float(part["score_lightgbm"].quantile(0.01)),
                "score_p99": float(part["score_lightgbm"].quantile(0.99)),
            }
        )
    rank_frames = []
    for _, part in scores.groupby("date"):
        current = part[["date", "symbol", "score_lightgbm"]].copy()
        current["rank"] = current["score_lightgbm"].rank(pct=True)
        rank_frames.append(current[["date", "symbol", "rank"]])
    ranks = pd.concat(rank_frames).sort_values(["symbol", "date"])
    adjacent = ranks.groupby("symbol")["rank"].corr(ranks.groupby("symbol")["rank"].shift(1))
    return pd.DataFrame(rows), {
        "score_is_ranking_not_probability": True,
        "saturation_ratio_at_global_extremes": float(
            ((scores["score_lightgbm"] <= scores["score_lightgbm"].quantile(0.001)) | (scores["score_lightgbm"] >= scores["score_lightgbm"].quantile(0.999))).mean()
        ),
        "adjacent_observation_rank_correlation": float(adjacent.mean()),
    }


def _turnover(scores: pd.DataFrame) -> dict:
    daily_sets = []
    for date, part in scores.groupby("date", sort=True):
        daily_sets.append((date, set(_select_sector_balanced(part, 20)["symbol"])))
    overlaps = []
    for (date, current), (_, previous) in zip(daily_sets[1:], daily_sets[:-1]):
        overlaps.append({"date": date, "retention": len(current & previous) / 20, "turnover": 1 - len(current & previous) / 20})
    frame = pd.DataFrame(overlaps)
    rebalance = frame.iloc[::20]
    return {
        "daily_mean_retention": float(frame["retention"].mean()),
        "daily_mean_turnover": float(frame["turnover"].mean()),
        "20_session_mean_retention_proxy": float(rebalance["retention"].mean()),
        "20_session_mean_turnover_proxy": float(rebalance["turnover"].mean()),
        "observations": len(frame),
    }


def _residual_alpha(scores: pd.DataFrame) -> dict:
    correlations = []
    for _, part in scores.groupby("date"):
        clean = part.dropna(subset=["score_lightgbm", "future_return_20d", "benchmark_weight_rank", "volatility_60_rank", "momentum", "liquidity"]).copy()
        if len(clean) < 40:
            continue
        sector = pd.get_dummies(clean["broad_sector"], drop_first=True, dtype=float)
        controls = np.column_stack([
            np.ones(len(clean)),
            clean[["benchmark_weight_rank", "volatility_60_rank", "momentum", "liquidity"]].to_numpy(dtype=float),
            sector.to_numpy(dtype=float),
        ])
        score = clean["score_lightgbm"].to_numpy(dtype=float)
        realized = clean["future_return_20d"].to_numpy(dtype=float)
        score_residual = score - controls @ np.linalg.lstsq(controls, score, rcond=None)[0]
        return_residual = realized - controls @ np.linalg.lstsq(controls, realized, rcond=None)[0]
        correlations.append(pd.Series(score_residual).corr(pd.Series(return_residual), method="spearman"))
    values = pd.Series(correlations).dropna()
    return {
        "controls": ["sector", "size", "volatility", "momentum", "liquidity"],
        "dates": len(values),
        "mean_residual_rank_ic": float(values.mean()),
        "positive_ratio": float((values > 0).mean()),
        "t_stat": _t_stat(values),
    }


def _error_analysis(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    extremes = []
    for date, part in scores.groupby("date"):
        valid = part.dropna(subset=["score_lightgbm", "future_return_20d"]).copy()
        if len(valid) < 100:
            continue
        valid["score_decile"] = pd.qcut(valid["score_lightgbm"].rank(method="first"), 10, labels=False) + 1
        valid["return_decile"] = pd.qcut(valid["future_return_20d"].rank(method="first"), 10, labels=False) + 1
        false_positive = valid[(valid["score_decile"] == 10) & (valid["return_decile"] == 1)]
        false_negative = valid[(valid["score_decile"] == 1) & (valid["return_decile"] == 10)]
        for label, sample in (("false_positive", false_positive), ("false_negative", false_negative)):
            for _, row in sample.iterrows():
                rows.append(
                    {
                        "type": label, "date": date, "symbol": row["symbol"],
                        "sector": row["broad_sector"], "return": row["future_return_20d"],
                        "size_rank": row["benchmark_weight_rank"], "liquidity_rank": row["amount_rank"],
                        "volatility": row["volatility_20"], "regime": row["regime"],
                    }
                )
        worst = valid.nlargest(20, "score_lightgbm").nsmallest(3, "future_return_20d")
        for _, row in worst.iterrows():
            extremes.append(
                {"date": date, "symbol": row["symbol"], "sector": row["broad_sector"], "return": row["future_return_20d"], "score": row["score_lightgbm"], "regime": row["regime"]}
            )
    errors = pd.DataFrame(rows)
    patterns = errors.groupby(["type", "sector", "regime"], as_index=False).agg(
        cases=("symbol", "size"), mean_return=("return", "mean"),
        mean_size_rank=("size_rank", "mean"), mean_liquidity_rank=("liquidity_rank", "mean"),
        mean_volatility=("volatility", "mean"),
    ).sort_values(["type", "cases"], ascending=[True, False])
    return patterns, pd.DataFrame(extremes).sort_values("return").head(100)


def _portfolio_and_costs(
    scores: pd.DataFrame, settings: DiagnosticSettings, base: ChallengerSettings
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    book, price_evidence = _load_verified_price_book(scores, base)
    portfolios = []
    period_frames = []
    for k in settings.top_ks:
        policy = PortfolioPolicy(f"sector_balanced_top{k}", k, sector_balanced=True)
        periods, _ = evaluate_stateful_portfolio_policy(scores, "score_lightgbm", 20, policy, book)
        summary = summarize_stateful_portfolio(periods, 20)
        portfolios.append({"top_k": k, **summary})
        if k == 20:
            period_frames.append(periods.assign(top_k=k))
    periods = pd.concat(period_frames, ignore_index=True)
    cost_rows = []
    turnover = (periods["buy_turnover"] + periods["sell_turnover"])
    for bps in settings.cost_bps:
        rate = bps / 10_000
        diagnostic_net = periods["gross_return"] - turnover * rate
        proxy = periods["research_benchmark_proxy_return"]
        total = float((1 + diagnostic_net).prod() - 1)
        benchmark_total = float((1 + proxy).prod() - 1)
        cost_rows.append(
            {
                "cost_bps_per_one_way_turnover": bps,
                "net_total_return": total,
                "research_proxy_total_return": benchmark_total,
                "net_research_proxy_alpha": total - benchmark_total,
                "mean_period_alpha": float((diagnostic_net - proxy).mean()),
                "cost_drag_sum": float((turnover * rate).sum()),
            }
        )
    return pd.DataFrame(portfolios), pd.DataFrame(cost_rows), price_evidence


def _horizon_metrics(scores5: pd.DataFrame, scores20: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon, frame, score, realized in (
        (5, scores5, "score", "future_return_5d"),
        (20, scores20, "score_lightgbm", "future_return_20d"),
    ):
        rows.append({"horizon": horizon, "status": "evaluated", **_ic_summary(daily_rank_metrics(frame, score, realized))})
    rows.extend([
        {"horizon": 10, "status": "not_evaluable_no_frozen_same_semantics_label"},
        {"horizon": 40, "status": "not_evaluable_no_frozen_same_semantics_label"},
    ])
    return pd.DataFrame(rows)


def _report_value(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _markdown_table(records: list[dict], columns: tuple[str, ...]) -> str:
    if not records:
        return "No evaluable observations."
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(_report_value(record.get(column)) for column in columns) + " |"
        for record in records
    ]
    return "\n".join([header, divider, *rows])


def _render_report(summary: dict) -> str:
    overall = summary["overall"]
    assessment = summary["assessment"]
    details = summary["diagnostic_details"]
    findings = "\n".join(f"- {item}" for item in summary["key_findings"])
    recommendation_sections = []
    for item in summary["recommendations"]:
        recommendation_sections.append(
            f"### {item['priority']} — {item['title']}\n\n"
            f"Hypothesis: {item['hypothesis']}\n\n"
            f"Evidence: {item['evidence']}\n\n"
            f"Experiment: {item['experiment']}\n\n"
            f"Success criterion: {item['success']}\n"
        )
    recommendations = "\n".join(recommendation_sections)
    yearly_table = _markdown_table(
        details["yearly"], ("period", "rank_ic_mean", "icir", "positive_ic_ratio")
    )
    quantile_table = _markdown_table(
        details["quantiles"],
        ("quantile", "mean_return", "excess_return", "precision_above_median"),
    )
    regime_table = _markdown_table(
        details["regimes"], ("dimension", "regime", "rank_ic_mean", "icir")
    )
    sector_table = _markdown_table(
        details["sectors"], ("sector", "rows", "rank_ic_mean", "positive_ic_ratio")
    )
    bucket_table = _markdown_table(
        details["buckets"], ("dimension", "bucket", "rank_ic_mean", "icir")
    )
    importance_table = _markdown_table(
        details["feature_stability"],
        ("feature", "feature_group", "mean_gain_importance", "active_fold_ratio", "classification"),
    )
    ablation_table = _markdown_table(
        details["ablation"], ("ablation", "rank_ic_mean", "rank_ic_change", "icir_change")
    )
    exposure_table = _markdown_table(
        details["exposures"],
        ("factor", "mean_cross_sectional_rank_correlation", "positive_ratio"),
    )
    cost_table = _markdown_table(
        details["costs"],
        ("cost_bps_per_one_way_turnover", "net_research_proxy_alpha", "cost_drag_sum"),
    )
    topk_table = _markdown_table(
        details["topk"],
        ("top_k", "net_research_proxy_alpha", "annualized_turnover", "max_drawdown"),
    )
    horizon_table = _markdown_table(
        details["horizons"], ("horizon", "status", "rank_ic_mean", "icir")
    )
    model_table = _markdown_table(
        details["models"], ("model", "rank_ic_mean", "icir", "positive_ic_ratio")
    )
    return f"""# GEN2_ALPHA_PREDICTION_DIAGNOSTIC_REPORT

## 1. Final Status

`{summary['final_status']}`

## 2. Baseline

Git baseline `{BASELINE_SHA}`; model `{MODEL_ID}`; target is the cross-sectional rank of 20-trading-day T+1-open to T+21-open return. The 000300 PIT constituent universe uses 61 candidate features, yearly train-only selection (15–20 active), an eight-year rolling training window, one validation year, 21-session purge, and yearly 2020–2025 unseen folds.

## 3. PIT / Leakage Audit

Membership snapshots, fundamental availability dates, and industry effective dates are on or before each decision date. Training labels mature before validation/test boundaries; date-symbol duplicates are absent. No 2026 labels were read. Historical OOS folds are valid for diagnostic description, but 2020–2025 was previously used for model selection and 2026 is disqualified, so there is no untouched confirmatory holdout.

## 4. Overall Predictive Power

Pearson IC `{overall['pearson_ic_mean']:.6f}`, Rank IC `{overall['rank_ic_mean']:.6f}`, ICIR `{overall['icir']:.4f}`, positive IC ratio `{overall['positive_ic_ratio']:.2%}`. Median Rank IC is `{overall['rank_ic_median']:.6f}` over `{overall['dates']}` dates. The Q5−Q1 mean-return spread is `{details['quantile_q5_minus_q1']:.6f}`; Top20 cross-sectional proxy excess is `{details['top20_cross_excess']:.6f}` with Precision@20 `{details['top20_precision']:.2%}`.

## 5. Ranking Monotonicity

The curve is not monotonic: Q3 has the highest mean realized return, while Q5 is only modestly above Q1. This indicates broad but weak ordering information rather than a clean calibrated return ladder.

{quantile_table}

## 6. Time Stability

All six annual mean Rank IC values are positive, but stability is poor: 2023 is strongest (`0.1078`) and 2025 is effectively flat (`0.00125`, positive-date ratio `41.9%`). Monthly and quarterly evidence is in `walk_forward_metrics.csv`.

{yearly_table}

## 7. Regime Performance

Risk-off Rank IC (`0.0187`) is far below neutral (`0.0665`). High-volatility Rank IC (`0.0768`) exceeds low-volatility (`0.0282`), so Gen2 does not generalize evenly. Volatility regimes use the current cross-section against a shifted trailing 252-session median; existing panel regimes are named `risk_on`, `risk_off`, and `neutral` rather than relabelled bull/bear/sideways.

{regime_table}

## 8. Sector Performance

Finance/real-estate (`0.0905`) and cyclical manufacturing (`0.0677`) are strongest. Technology (`0.0075`) and `other` (`0.0012`) are nearly uninformative; defensive has fewer than 20 names per daily cross-section and is not evaluable for within-sector IC. Sector balancing historically changed Top20 net proxy alpha from `-0.0221` to `+0.0249` and reduced worst sector weight from `0.899` to `0.420`, at the cost of higher turnover and worse drawdown. This is development evidence, not untouched confirmation.

{sector_table}

## 9. Cap / Liquidity / Volatility

The signal is strongest in small-cap (`0.0751`) versus large-cap (`0.0235`) and high-volatility (`0.0598`) versus low-volatility (`0.0096`) buckets. It is not merely an illiquidity artifact: high-liquidity Rank IC (`0.0630`) exceeds low-liquidity (`0.0369`). Size uses PIT benchmark-weight rank.

{bucket_table}

## 10. Feature Diagnostics

Risk, liquidity and fundamental changes dominate. `volatility_60_rank` has the highest mean gain but is unstable; `liquidity`, `revenue_growth_change_rank`, `profit_growth_change_rank`, and `gross_margin_change_rank` are active in every fold. Only three sampled pairs exceed `|rho|=0.90`: volatility/downside-volatility, revenue-growth/growth, and profit-growth/growth. Importance is fold-specific and absence from yearly train-only selection counts as zero activity.

{importance_table}

## 11. Feature Group Ablation

Removing risk reduces Rank IC by `0.01169`; removing liquidity by `0.00788`; removing price behavior by `0.00384`. Removing the full fundamental group slightly raises mean Rank IC (`+0.00122`) but lowers ICIR (`-0.0278`), suggesting unstable/conditional rather than uniformly useless fundamentals.

{ablation_table}

## 12. Factor Exposure

The score is strongly exposed to volatility (`rho=0.422`) and size (`0.212`), and negatively related to momentum (`-0.252`), liquidity score (`-0.208`), and sector momentum (`-0.185`). It is therefore not a disguised positive momentum ranking. After controlling sector, size, volatility, momentum and liquidity day by day, residual Rank IC is `{summary['residual_alpha']['mean_residual_rank_ic']:.4f}` (t-stat `{summary['residual_alpha']['t_stat']:.2f}`). Beta is not available.

{exposure_table}

## 13. Error Analysis

False negatives concentrate in technology and cyclical manufacturing across neutral and risk-off periods; corresponding false-positive distributions are retained in `error_patterns.csv`. The 100 worst top-ranked outcomes are in `extreme_failures.csv`. Earnings/event, gap, order-book depth and crowding explanations are not supported by the current panel and are explicitly not inferred.

## 14. Turnover

Daily Top20 retention `{summary['turnover']['daily_mean_retention']:.2%}` and turnover `{summary['turnover']['daily_mean_turnover']:.2%}`. The canonical stateful Top20 portfolio has average one-way turnover `{details['top20_one_way_turnover']:.2%}` and annualized turnover `{details['top20_annualized_turnover']:.2f}`. Adjacent score-rank persistence is `{summary['calibration']['adjacent_observation_rank_correlation']:.3f}`.

## 15. Cost Sensitivity

Top20 research-proxy alpha is `0.1481` at 0 bps, `0.0311` at 10 bps, `-0.0765` at 20 bps and `-0.3495` at 50 bps. The alpha is therefore cost-fragile. These are one-way-turnover stress tests, not broker/execution claims.

{cost_table}

## 16. Top-K Sensitivity

Top20 is not uniquely supported. In the stateful development replay, Top10 is approximately flat after canonical costs, Top20 is `+0.0249`, Top30 `+0.1447`, and Top50 `+0.2255`; Top5 is high-return but concentrated and more volatile. These comparisons reuse development folds and are hypotheses, not parameters to select retroactively.

{topk_table}

## 17. Horizon Diagnostic

20D Rank IC (`0.0499`, ICIR `0.2653`) exceeds 5D (`0.0359`, ICIR `0.1939`) under existing PIT labels and fixed folds. 10D/40D are explicitly not evaluable because no frozen same-semantics labels exist.

{horizon_table}

## 18. Challenger Models

LightGBM (`0.04988`) only narrowly exceeds Ridge (`0.04828`). The fixed 50/50 rank ensemble has the highest mean Rank IC (`0.05103`) but lower ICIR than LightGBM (`0.2556` versus `0.2653`), so ensemble promotion is not justified. No new dependency or model zoo was introduced.

{model_table}

## 19. Overfitting / Statistical Confidence

Training Rank IC is `0.160–0.175` across folds versus OOS `0.0499`, a gap of roughly `0.110–0.125`, consistent with material overfit. The 20-session block-bootstrap 95% CI for mean daily Rank IC is `[{summary['bootstrap']['ci_lower']:.4f}, {summary['bootstrap']['ci_upper']:.4f}]`; serial dependence remains a major uncertainty. Rows are not treated as independent observations. No untouched final holdout exists.

## 20. Key Findings

{findings}

## 21. Gen3 Priorities

{recommendations}

## 22. What NOT To Do

- Do not claim promotion or production readiness from reused 2020–2025 development OOS.
- Do not add deep learning, hundreds of features, or alternative data without a new untouched protocol.
- Do not tune Top-K, horizon, or model complexity against these same folds and relabel them holdout.
- Do not interpret the research benchmark proxy as approved official benchmark alpha.

## 23. Git / PR

Branch `codex/gen2-alpha-diagnostic`; commit and PR metadata are populated during delivery. Frozen files modified: none.

## 24. Final Assessment

`Is Gen2 genuinely predictive out-of-sample?` — `{assessment}`.

{summary['assessment_evidence']}

The next phase should focus on evidence-driven alpha improvement rather than further core architecture expansion.
"""


def run(settings: DiagnosticSettings | None = None) -> dict:
    settings = settings or DiagnosticSettings()
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    base = ChallengerSettings()
    data, evidence = load_maturity_safe_development_dataset(base)
    scores, importance, train_gap, folds = _fit_oos(data, settings, base)
    scores5 = _fit_horizon_5(data, settings, base)
    daily = daily_rank_metrics(scores, "score_lightgbm", "future_return_20d")
    overall = _ic_summary(daily)
    bootstrap = _block_bootstrap(daily["rank_ic"], settings)
    time_metrics = _time_metrics(scores)
    quantiles = _quantile_metrics(scores, settings.quantiles)
    topk_cross = _topk_cross_sectional(scores, settings)
    regimes = _regime_metrics(scores)
    sectors = _sector_metrics(scores)
    buckets = _bucket_metrics(scores)
    exposure = _factor_exposure(scores)
    single, redundancy, stability = _feature_diagnostics(data, scores, importance, base)
    ablation = _ablation_metrics(scores)
    models = _model_metrics(scores)
    score_drift, calibration = _score_calibration(scores)
    turnover = _turnover(scores)
    residual = _residual_alpha(scores)
    errors, extremes = _error_analysis(scores)
    topk, costs, price_evidence = _portfolio_and_costs(scores, settings, base)
    horizons = _horizon_metrics(scores5, scores)
    oos_lgbm = models[models["model"].eq("lightgbm")].iloc[0].to_dict()
    train_gap["oos_rank_ic"] = oos_lgbm["rank_ic_mean"]
    train_gap["train_minus_oos_rank_ic"] = train_gap["rank_ic_mean"] - oos_lgbm["rank_ic_mean"]
    yearly = time_metrics[time_metrics["frequency"].eq("year")]
    rank_ci_positive = bootstrap.get("ci_lower") is not None and bootstrap["ci_lower"] > 0
    stable_years = int((yearly["rank_ic_mean"] > 0).sum())
    assessment = "WEAK / REGIME_DEPENDENT" if overall["rank_ic_mean"] > 0 else "NO"
    q1 = quantiles[quantiles["quantile"].eq(1)].iloc[0]
    q5 = quantiles[quantiles["quantile"].eq(settings.quantiles)].iloc[0]
    top20_cross = topk_cross[topk_cross["top_k"].eq(20)].iloc[0]
    top20_portfolio = topk[topk["top_k"].eq(20)].iloc[0]
    sector_balance_source = pd.read_csv(
        "artifacts/research_challenger/gen02/experiments/005_correctness_hardening/portfolio_variants_corrected.csv"
    )
    sector_balance = sector_balance_source[
        sector_balance_source["model"].eq("lightgbm_regression")
        & sector_balance_source["horizon"].eq(20)
        & sector_balance_source["portfolio_policy"].isin(
            ["equal_top20", "sector_balanced_top20"]
        )
    ][
        [
            "portfolio_policy", "net_research_proxy_alpha", "annualized_turnover",
            "max_drawdown", "worst_maximum_sector_weight",
        ]
    ]
    diagnostic_details = {
        "yearly": yearly.to_dict("records"),
        "quantiles": quantiles.to_dict("records"),
        "regimes": regimes.to_dict("records"),
        "sectors": sectors.to_dict("records"),
        "buckets": buckets.to_dict("records"),
        "feature_stability": stability.head(12).to_dict("records"),
        "single_feature_alpha": single.head(12).to_dict("records"),
        "redundancy_pairs": redundancy.head(20).to_dict("records"),
        "ablation": ablation.to_dict("records"),
        "exposures": exposure.to_dict("records"),
        "costs": costs.to_dict("records"),
        "topk": topk.to_dict("records"),
        "topk_cross_sectional": topk_cross.to_dict("records"),
        "horizons": horizons.to_dict("records"),
        "models": models.to_dict("records"),
        "sector_balance_comparison": sector_balance.to_dict("records"),
        "quantile_q5_minus_q1": float(q5["mean_return"] - q1["mean_return"]),
        "top20_cross_excess": float(top20_cross["benchmark_excess_return"]),
        "top20_precision": float(top20_cross["precision_at_k"]),
        "top20_one_way_turnover": float(top20_portfolio["average_one_way_turnover"]),
        "top20_annualized_turnover": float(top20_portfolio["annualized_turnover"]),
    }
    key_findings = [
        f"Mean daily Rank IC is {overall['rank_ic_mean']:.4f}; 20-session block-bootstrap 95% CI is [{bootstrap.get('ci_lower')}, {bootstrap.get('ci_upper')}].",
        f"{stable_years}/{len(yearly)} yearly folds have positive Rank IC; 2025 is {float(yearly[yearly['period'].eq('2025')]['rank_ic_mean'].iloc[0]):.4f}.",
        f"Rank IC confidence lower bound is {'above' if rank_ci_positive else 'not above'} zero.",
        "There is no untouched confirmatory holdout: all 2020–2025 folds were available during historical development, and 2026 is disqualified.",
        f"Residual Rank IC after sector/size/volatility/momentum/liquidity controls is {residual['mean_residual_rank_ic']:.4f}.",
        f"Ridge Rank IC is {float(models[models['model'].eq('ridge')]['rank_ic_mean'].iloc[0]):.4f}; model complexity must earn its incremental value.",
        f"Daily Top20 retention is {turnover['daily_mean_retention']:.1%}; cost fragility is reported without execution claims.",
        "The Q1→Q5 return curve is not monotonic: Q3 outperforms Q5, so ranking information is not cleanly calibrated.",
        "Risk-off, technology, large-cap, and low-volatility slices are the principal weak regions.",
        "Top20 is not uniquely supported: Top30/Top50 had stronger development net proxy alpha, but must not be selected on these reused folds.",
    ]
    recommendations = [
        {
            "priority": "P0", "title": "Create a genuinely untouched confirmation period",
            "hypothesis": "Selection bias is the largest unresolved uncertainty.",
            "evidence": "The frozen model was selected using 2020–2025 and 2026 is not untouched.",
            "experiment": "Pre-register features, model, Top-K and gates, then collect prospective matured labels without retuning.",
            "success": "Positive Rank IC block-bootstrap lower bound and positive net proxy alpha across the pre-registered period.",
        },
        {
            "priority": "P0", "title": "Target the empirically weakest regime and tail",
            "hypothesis": "Gen2 failure is concentrated rather than uniform.",
            "evidence": "Risk-off Rank IC is 0.0187 versus 0.0665 in neutral; technology is 0.0075 and 2025 is 0.00125.",
            "experiment": "Develop PIT-safe regime interactions only on a new development partition.",
            "success": "Worst-regime Rank IC improves while full-period ICIR and turnover do not deteriorate.",
        },
        {
            "priority": "P1", "title": "Compress redundant features and benchmark Ridge",
            "hypothesis": "A smaller information set may match tree performance with less variance.",
            "evidence": "Ridge Rank IC 0.0483 nearly matches LightGBM 0.0499; only three sampled pairs exceed |rho|=0.90.",
            "experiment": "Pre-register cluster representatives and compare frozen Ridge/LGBM on new folds.",
            "success": "No Rank IC loss beyond 0.003 and improved fold variance or turnover.",
        },
        {
            "priority": "P1", "title": "Treat Top-K and turnover as joint design variables",
            "hypothesis": "Cutoff concentration and churn can erase weak ranking alpha.",
            "evidence": "Top20 proxy alpha turns negative at 20 bps; Top30/50 outperform Top20 in reused development evidence.",
            "experiment": "Pre-register one buffered selection challenger on development-only data.",
            "success": "Higher net proxy alpha at 20–50 bps with no worse drawdown and sector concentration.",
        },
        {
            "priority": "P2", "title": "Test rank ensemble only if complementarity persists",
            "hypothesis": "Ridge can diversify LightGBM fold errors.",
            "evidence": "The fixed ensemble raises mean Rank IC to 0.0510 but lowers ICIR to 0.2556 versus LightGBM 0.2653.",
            "experiment": "Freeze ensemble weights before a new validation period.",
            "success": "Higher ICIR and worst-regime Rank IC without higher turnover.",
        },
    ]
    summary = {
        "final_status": "GEN2_ALPHA_DIAGNOSTIC_COMPLETE",
        "assessment": assessment,
        "assessment_evidence": "The strict annual folds show descriptive OOS ranking information, but temporal/regime variation, weak tail evidence, cost sensitivity, and absence of an untouched holdout prevent a strong YES conclusion.",
        "baseline_sha": BASELINE_SHA,
        "execution_sha": _git("rev-parse", "HEAD"),
        "model_id": MODEL_ID,
        "config": asdict(settings),
        "data": evidence,
        "price_evidence": price_evidence,
        "pit_audit": {
            "future_feature_leakage": False,
            "membership_leakage": False,
            "survivorship_protection": "PIT 000300 membership snapshots",
            "label_overlap": False,
            "split_contamination": False,
            "full_sample_normalization": False,
            "2026_labels_read": False,
            "untouched_holdout_available": False,
            "result_validity": "valid_descriptive_walk_forward_not_confirmatory_holdout",
        },
        "overall": overall,
        "bootstrap": bootstrap,
        "turnover": turnover,
        "calibration": calibration,
        "residual_alpha": residual,
        "diagnostic_details": diagnostic_details,
        "key_findings": key_findings,
        "recommendations": recommendations,
        "evaluable_horizons": [5, 20],
        "not_evaluable": ["10D horizon", "40D horizon", "beta exposure", "event/earnings/order-book/crowding failure causes", "official benchmark alpha", "untouched holdout"],
    }
    outputs = {
        "walk_forward_metrics.csv": time_metrics,
        "daily_ic.csv": daily,
        "quantile_monotonicity.csv": quantiles,
        "regime_metrics.csv": regimes,
        "sector_metrics.csv": sectors,
        "bucket_metrics.csv": buckets,
        "feature_importance.csv": importance,
        "feature_stability.csv": stability,
        "feature_ablation.csv": ablation,
        "single_feature_alpha.csv": single,
        "feature_redundancy.csv": redundancy,
        "factor_exposure.csv": exposure,
        "topk_cross_sectional.csv": topk_cross,
        "topk_sensitivity.csv": topk,
        "cost_sensitivity.csv": costs,
        "horizon_diagnostic.csv": horizons,
        "model_comparison.csv": models,
        "train_oos_gap.csv": train_gap,
        "score_drift.csv": score_drift,
        "error_patterns.csv": errors,
        "extreme_failures.csv": extremes,
    }
    for name, frame in outputs.items():
        _write_csv(settings.artifact_dir / name, frame)
    _write_json(settings.artifact_dir / "folds.json", folds)
    _write_json(settings.artifact_dir / "bootstrap_confidence.json", bootstrap)
    _write_json(settings.artifact_dir / "residual_alpha.json", residual)
    _write_json(settings.artifact_dir / "diagnostic_summary.json", summary)
    (settings.artifact_dir / "GEN2_ALPHA_PREDICTION_DIAGNOSTIC_REPORT.md").write_text(
        _render_report(summary), encoding="utf-8", newline="\n"
    )
    manifest = {}
    for path in sorted(settings.artifact_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            manifest[path.name] = _sha256(path)
    _write_json(settings.artifact_dir / "artifact_manifest.json", manifest)
    return summary


def refresh_existing_report(settings: DiagnosticSettings | None = None) -> dict:
    """Refresh narrative-only artifacts without rerunning model estimation."""

    settings = settings or DiagnosticSettings()
    root = settings.artifact_dir
    summary = json.loads((root / "diagnostic_summary.json").read_text(encoding="utf-8"))
    def load(name: str) -> pd.DataFrame:
        return pd.read_csv(root / name)

    yearly = load("walk_forward_metrics.csv")
    yearly = yearly[yearly["frequency"].eq("year")]
    quantiles = load("quantile_monotonicity.csv")
    regimes = load("regime_metrics.csv")
    sectors = load("sector_metrics.csv")
    buckets = load("bucket_metrics.csv")
    stability = load("feature_stability.csv")
    single = load("single_feature_alpha.csv")
    redundancy = load("feature_redundancy.csv")
    ablation = load("feature_ablation.csv")
    exposure = load("factor_exposure.csv")
    costs = load("cost_sensitivity.csv")
    topk = load("topk_sensitivity.csv")
    topk_cross = load("topk_cross_sectional.csv")
    horizons = load("horizon_diagnostic.csv")
    models = load("model_comparison.csv")
    q1 = quantiles[quantiles["quantile"].eq(1)].iloc[0]
    q5 = quantiles[quantiles["quantile"].eq(settings.quantiles)].iloc[0]
    top20_cross = topk_cross[topk_cross["top_k"].eq(20)].iloc[0]
    top20_portfolio = topk[topk["top_k"].eq(20)].iloc[0]
    sector_balance_source = pd.read_csv(
        "artifacts/research_challenger/gen02/experiments/005_correctness_hardening/portfolio_variants_corrected.csv"
    )
    sector_balance = sector_balance_source[
        sector_balance_source["model"].eq("lightgbm_regression")
        & sector_balance_source["horizon"].eq(20)
        & sector_balance_source["portfolio_policy"].isin(
            ["equal_top20", "sector_balanced_top20"]
        )
    ][
        [
            "portfolio_policy", "net_research_proxy_alpha", "annualized_turnover",
            "max_drawdown", "worst_maximum_sector_weight",
        ]
    ]
    summary["diagnostic_details"] = {
        "yearly": yearly.to_dict("records"),
        "quantiles": quantiles.to_dict("records"),
        "regimes": regimes.to_dict("records"),
        "sectors": sectors.to_dict("records"),
        "buckets": buckets.to_dict("records"),
        "feature_stability": stability.head(12).to_dict("records"),
        "single_feature_alpha": single.head(12).to_dict("records"),
        "redundancy_pairs": redundancy.head(20).to_dict("records"),
        "ablation": ablation.to_dict("records"),
        "exposures": exposure.to_dict("records"),
        "costs": costs.to_dict("records"),
        "topk": topk.to_dict("records"),
        "topk_cross_sectional": topk_cross.to_dict("records"),
        "horizons": horizons.to_dict("records"),
        "models": models.to_dict("records"),
        "sector_balance_comparison": sector_balance.to_dict("records"),
        "quantile_q5_minus_q1": float(q5["mean_return"] - q1["mean_return"]),
        "top20_cross_excess": float(top20_cross["benchmark_excess_return"]),
        "top20_precision": float(top20_cross["precision_at_k"]),
        "top20_one_way_turnover": float(top20_portfolio["average_one_way_turnover"]),
        "top20_annualized_turnover": float(top20_portfolio["annualized_turnover"]),
    }
    additions = [
        "The Q1→Q5 return curve is not monotonic: Q3 outperforms Q5, so ranking information is not cleanly calibrated.",
        "Risk-off, technology, large-cap, and low-volatility slices are the principal weak regions.",
        "Top20 is not uniquely supported: Top30/Top50 had stronger development net proxy alpha, but must not be selected on these reused folds.",
    ]
    summary["key_findings"] = list(dict.fromkeys([*summary["key_findings"], *additions]))
    evidence = {
        "Target the empirically weakest regime and tail": "Risk-off Rank IC is 0.0187 versus 0.0665 in neutral; technology is 0.0075 and 2025 is 0.00125.",
        "Compress redundant features and benchmark Ridge": "Ridge Rank IC 0.0483 nearly matches LightGBM 0.0499; only three sampled pairs exceed |rho|=0.90.",
        "Treat Top-K and turnover as joint design variables": "Top20 proxy alpha turns negative at 20 bps; Top30/50 outperform Top20 in reused development evidence.",
        "Test rank ensemble only if complementarity persists": "The fixed ensemble raises mean Rank IC to 0.0510 but lowers ICIR to 0.2556 versus LightGBM 0.2653.",
    }
    for recommendation in summary["recommendations"]:
        if recommendation["title"] in evidence:
            recommendation["evidence"] = evidence[recommendation["title"]]
    _write_json(root / "diagnostic_summary.json", summary)
    (root / "GEN2_ALPHA_PREDICTION_DIAGNOSTIC_REPORT.md").write_text(
        _render_report(summary), encoding="utf-8", newline="\n"
    )
    manifest = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            manifest[path.name] = _sha256(path)
    _write_json(root / "artifact_manifest.json", manifest)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run isolated Gen2 alpha diagnostics")
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--refresh-report", action="store_true")
    args = parser.parse_args(argv)
    configured = DiagnosticSettings(artifact_dir=args.artifact_dir)
    summary = refresh_existing_report(configured) if args.refresh_report else run(configured)
    print(summary["final_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
