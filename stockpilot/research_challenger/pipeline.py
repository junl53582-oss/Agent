from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ChallengerSettings, MODEL_NAMES
from .data import factor_inventory, load_research_dataset, sha256
from .factors import (
    daily_rank_ic_matrix,
    residualize_cross_section,
    select_factors_train_only,
)
from .metrics import (
    daily_rank_metrics,
    evaluate_topk,
    moving_block_bootstrap_delta,
    quantile_returns,
    summarize_ic,
    summarize_topk,
)
from .models import fit_candidate_models, v6_oos_scores
from .split import build_fold, fold_receipt


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_new(path: Path, payload: dict | list) -> None:
    if path.exists():
        raise RuntimeError(f"V31 immutable artifact already exists: {path}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _write_csv_new(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise RuntimeError(f"V31 immutable artifact already exists: {path}")
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _map_market_regime(values: pd.Series) -> pd.Series:
    return values.map({"risk_on": "bull", "risk_off": "bear", "neutral": "sideways"}).fillna(
        "sideways"
    )


def _permutation_importance(
    model: object,
    processor: object,
    test: pd.DataFrame,
    features: tuple[str, ...],
    target: str,
    seed: int,
) -> dict[str, float]:
    sample = test.sort_values(["date", "symbol"])
    if len(sample) > 20_000:
        positions = np.linspace(0, len(sample) - 1, 20_000, dtype=int)
        sample = sample.iloc[positions]
    x = processor.transform(sample, features)
    y = pd.to_numeric(sample[target], errors="coerce").to_numpy(dtype=float)
    baseline = float(pd.Series(model.predict(x)).corr(pd.Series(y), method="spearman"))
    rng = np.random.default_rng(seed)
    importance = {}
    for index, feature in enumerate(features):
        changed = x.copy()
        changed[:, index] = changed[rng.permutation(len(changed)), index]
        value = float(pd.Series(model.predict(changed)).corr(pd.Series(y), method="spearman"))
        importance[feature] = baseline - value
    return importance


def _factor_oos_stability(
    test: pd.DataFrame,
    features: tuple[str, ...],
    horizon: int,
    year: int,
) -> list[dict]:
    target = f"return_rank_{horizon}d"
    daily = daily_rank_ic_matrix(test, features, target)
    rows = []
    for feature in features:
        values = daily[feature].dropna()
        std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        rows.append(
            {
                "test_year": year,
                "horizon": horizon,
                "factor_name": feature,
                "neutralization": "raw",
                "ic_dates": int(len(values)),
                "mean_rank_ic": float(values.mean()),
                "rank_ic_std": std,
                "rank_ic_ir": float(values.mean() / std) if np.isfinite(std) and std > 0 else 0.0,
                "positive_ratio": float((values > 0).mean()),
            }
        )
    return rows


def _neutralization_diagnostics(
    test: pd.DataFrame,
    selected: tuple[str, ...],
    year: int,
) -> list[dict]:
    rows = []
    target = "return_rank_5d"
    for feature in selected:
        for mode in ("industry", "size", "industry_size"):
            neutral = residualize_cross_section(test, test[feature], mode)
            work = test[["date", target]].copy()
            work["neutral_factor"] = neutral
            daily = daily_rank_metrics(work, "neutral_factor", target)
            summary = summarize_ic(daily)
            rows.append(
                {
                    "test_year": year,
                    "horizon": 5,
                    "factor_name": feature,
                    "neutralization": mode,
                    "ic_dates": summary["dates"],
                    "mean_rank_ic": summary["mean_rank_ic"],
                    "rank_ic_std": summary["rank_ic_std"],
                    "rank_ic_ir": summary["rank_ic_ir"],
                    "positive_ratio": summary["positive_rank_ic_ratio"],
                }
            )
    return rows


def _aggregate_regime(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in daily.groupby(
        ["model", "horizon", "regime_dimension", "regime"], dropna=False
    ):
        model, horizon, dimension, regime = keys
        summary = summarize_ic(group)
        rows.append(
            {
                "model": model,
                "horizon": horizon,
                "regime_dimension": dimension,
                "regime": regime,
                **summary,
            }
        )
    return pd.DataFrame(rows)


def run_v31(settings: ChallengerSettings | None = None) -> dict:
    settings = settings or ChallengerSettings()
    settings.ensure_dirs()
    decision_path = settings.artifact_dir / "decision.json"
    if decision_path.exists():
        raise RuntimeError("V31 final OOS is immutable and may run only once")
    from .freeze import verify_plan_lock

    frozen = verify_plan_lock(settings)
    if not frozen["intact"]:
        raise RuntimeError(f"V31 frozen protocol/code mismatch: {frozen}")
    data, data_evidence = load_research_dataset(settings)
    inventory = factor_inventory(data, settings)
    fold_rows: list[dict] = []
    factor_audits: list[pd.DataFrame] = []
    factor_stability_rows: list[dict] = []
    importance_rows: list[dict] = []
    daily_rows: list[pd.DataFrame] = []
    industry_rows: list[dict] = []
    topk_periods: dict[tuple[str, int, int], list[pd.DataFrame]] = {}
    quantile_rows: list[pd.DataFrame] = []
    selected_by_year: dict[int, tuple[str, ...]] = {}
    correlation_last = pd.DataFrame()

    for year in settings.oos_years:
        selection_fold = build_fold(
            data,
            year,
            settings.selection_horizon,
            training_window_years=settings.training_window_years,
            validation_years=settings.validation_years,
            purge_gap_trading_days=settings.purge_gaps[settings.selection_horizon],
        )
        selection = select_factors_train_only(data.loc[selection_fold.train_index], settings)
        selected_by_year[year] = selection.selected
        audit = selection.audit.copy()
        audit.insert(0, "test_year", year)
        factor_audits.append(audit)
        correlation_last = selection.correlation.copy()

        year_union = data[
            data["date"].dt.year.eq(year)
            & data[[f"future_return_{h}d" for h in settings.horizons]].notna().any(axis=1)
        ].copy()
        v6_union = v6_oos_scores(data, year_union, year)

        for horizon in settings.horizons:
            fold = build_fold(
                data,
                year,
                horizon,
                training_window_years=settings.training_window_years,
                validation_years=settings.validation_years,
                purge_gap_trading_days=settings.purge_gaps[horizon],
            )
            receipt = fold_receipt(data, fold)
            receipt["selected_factors"] = list(selection.selected)
            fold_rows.append(receipt)
            train = data.loc[fold.refit_index].copy()
            test = data.loc[fold.test_index].copy()
            target = f"return_rank_{horizon}d"
            predictions, gains, processor, models = fit_candidate_models(
                train, test, selection.selected, target, settings
            )
            test["score_v6"] = v6_union.reindex(test.index)
            for model, values in predictions.items():
                test[f"score_{model}"] = values
            test_ids = frozenset(
                pd.to_datetime(test["date"]).dt.strftime("%Y-%m-%d") + ":" + test["symbol"]
            )
            if processor.fit_row_ids_.intersection(test_ids):
                raise AssertionError("train-only preprocessor touched OOS rows")
            for row in gains:
                row.update({"test_year": year, "horizon": horizon, "importance_type": "gain"})
            importance_rows.extend(gains)
            if horizon == settings.selection_horizon:
                permutation = _permutation_importance(
                    models[settings.pre_registered_challenger],
                    processor,
                    test,
                    selection.selected,
                    target,
                    settings.random_seed + year,
                )
                for feature, value in permutation.items():
                    importance_rows.append(
                        {
                            "model": settings.pre_registered_challenger,
                            "feature": feature,
                            "gain_importance": np.nan,
                            "model_signature": next(
                                row["model_signature"]
                                for row in gains
                                if row["model"] == settings.pre_registered_challenger
                            ),
                            "training_rows": next(
                                row["training_rows"]
                                for row in gains
                                if row["model"] == settings.pre_registered_challenger
                            ),
                            "test_year": year,
                            "horizon": horizon,
                            "importance_type": "permutation_rank_ic_drop",
                            "permutation_importance": value,
                        }
                    )

            train_daily_vol = train.groupby("date")["volatility_20"].mean()
            volatility_cutoff = float(train_daily_vol.median())
            date_regime = test.groupby("date")["regime"].first().map(
                {"risk_on": "bull", "risk_off": "bear", "neutral": "sideways"}
            )
            date_vol = np.where(
                test.groupby("date")["volatility_20"].mean() >= volatility_cutoff,
                "high_volatility",
                "low_volatility",
            )
            date_volatility = pd.Series(date_vol, index=test["date"].drop_duplicates().sort_values())

            factor_stability_rows.extend(
                _factor_oos_stability(test, settings.factor_columns, horizon, year)
            )
            if horizon == settings.selection_horizon:
                factor_stability_rows.extend(
                    _neutralization_diagnostics(test, selection.selected, year)
                )

            for model in MODEL_NAMES:
                score_column = f"score_{model}"
                daily = daily_rank_metrics(test, score_column, f"future_return_{horizon}d")
                daily["model"] = model
                daily["horizon"] = horizon
                daily["test_year"] = year
                daily["market_regime"] = daily["date"].map(date_regime)
                daily["volatility_regime"] = daily["date"].map(date_volatility)
                market = daily.copy()
                market["regime_dimension"] = "market"
                market["regime"] = market["market_regime"]
                volatility = daily.copy()
                volatility["regime_dimension"] = "volatility"
                volatility["regime"] = volatility["volatility_regime"]
                daily_rows.extend([market, volatility])

                for sector, part in test.groupby("broad_sector"):
                    sector_daily = daily_rank_metrics(
                        part, score_column, f"future_return_{horizon}d"
                    )
                    if sector_daily.empty:
                        continue
                    summary = summarize_ic(sector_daily)
                    industry_rows.append(
                        {
                            "model": model,
                            "horizon": horizon,
                            "test_year": year,
                            "broad_sector": str(sector),
                            **summary,
                        }
                    )
                for k in settings.top_ks:
                    periods = evaluate_topk(
                        test,
                        score_column,
                        horizon,
                        k,
                        rebalance_every=settings.rebalance_every[horizon],
                        buy_rate=settings.buy_rate,
                        sell_rate=settings.sell_rate,
                    )
                    periods["test_year"] = year
                    topk_periods.setdefault((model, horizon, k), []).append(periods)
                quantiles = quantile_returns(
                    test, score_column, f"future_return_{horizon}d", quantiles=5
                )
                quantiles["model"] = model
                quantiles["horizon"] = horizon
                quantiles["test_year"] = year
                quantile_rows.append(quantiles)

    daily_all = pd.concat(daily_rows, ignore_index=True)
    unique_daily = daily_all[daily_all["regime_dimension"].eq("market")].copy()
    model_rows = []
    yearly_rows = []
    for (model, horizon), group in unique_daily.groupby(["model", "horizon"]):
        model_rows.append({"model": model, "horizon": horizon, "run_status": "COMPLETED", **summarize_ic(group)})
        for year, part in group.groupby("test_year"):
            yearly_rows.append({"model": model, "horizon": horizon, "test_year": int(year), **summarize_ic(part)})
    for unavailable in ("xgboost", "catboost"):
        for horizon in settings.horizons:
            model_rows.append(
                {
                    "model": unavailable,
                    "horizon": horizon,
                    "run_status": "DISABLED_DEPENDENCY_NOT_PRESENT",
                }
            )
    model_comparison = pd.DataFrame(model_rows)
    yearly_metrics = pd.DataFrame(yearly_rows)
    regime_metrics = _aggregate_regime(daily_all)
    industry_metrics = pd.DataFrame(industry_rows)

    topk_rows = []
    turnover_rows = []
    topk_all: dict[tuple[str, int, int], pd.DataFrame] = {}
    for key, pieces in topk_periods.items():
        model, horizon, k = key
        periods = pd.concat(pieces, ignore_index=True).sort_values("date")
        topk_all[key] = periods
        summary = summarize_topk(periods, horizon)
        topk_rows.append({"model": model, "horizon": horizon, "top_k": k, **summary})
        turnover_rows.append(
            {
                "model": model,
                "horizon": horizon,
                "top_k": k,
                "average_one_way_turnover": summary.get("average_one_way_turnover"),
                "annualized_turnover": summary.get("annualized_turnover"),
                "average_transaction_cost": summary.get("average_transaction_cost"),
            }
        )
    topk_metrics = pd.DataFrame(topk_rows)
    turnover_metrics = pd.DataFrame(turnover_rows)
    quantile_metrics = (
        pd.concat(quantile_rows, ignore_index=True)
        .groupby(["model", "horizon", "quantile"], as_index=False)
        .agg(actual_return=("actual_return", "mean"), sample_size=("sample_size", "sum"))
    )

    h5_daily = unique_daily[unique_daily["horizon"].eq(5)].pivot(
        index="date", columns="model", values="rank_ic"
    )
    rank_bootstrap = moving_block_bootstrap_delta(
        h5_daily[settings.pre_registered_challenger],
        h5_daily["v6"],
        replications=settings.bootstrap_replications,
        block_length=settings.bootstrap_block_length,
        seed=settings.random_seed,
    )
    challenger_top20 = topk_all[(settings.pre_registered_challenger, 5, 20)].set_index("date")[
        "net_alpha"
    ]
    v6_top20 = topk_all[("v6", 5, 20)].set_index("date")["net_alpha"]
    top20_bootstrap = moving_block_bootstrap_delta(
        challenger_top20,
        v6_top20,
        replications=settings.bootstrap_replications,
        block_length=min(10, settings.bootstrap_block_length),
        seed=settings.random_seed,
    )
    bootstrap = {"rank_ic_5d_delta": rank_bootstrap, "top20_net_alpha_5d_delta": top20_bootstrap}

    comparison = model_comparison.set_index(["model", "horizon"])
    top_comparison = topk_metrics.set_index(["model", "horizon", "top_k"])
    challenger = comparison.loc[(settings.pre_registered_challenger, 5)]
    champion = comparison.loc[("v6", 5)]
    challenger_top = top_comparison.loc[(settings.pre_registered_challenger, 5, 20)]
    champion_top = top_comparison.loc[("v6", 5, 20)]
    yearly_h5 = yearly_metrics[yearly_metrics["horizon"].eq(5)].pivot(
        index="test_year", columns="model", values="mean_rank_ic"
    )
    positive_years = int(
        (yearly_h5[settings.pre_registered_challenger] > yearly_h5["v6"]).sum()
    )
    candidate_regime = regime_metrics[
        regime_metrics["model"].eq(settings.pre_registered_challenger)
        & regime_metrics["horizon"].eq(5)
    ]
    champion_regime = regime_metrics[
        regime_metrics["model"].eq("v6") & regime_metrics["horizon"].eq(5)
    ]
    candidate_sector = industry_metrics[
        industry_metrics["model"].eq(settings.pre_registered_challenger)
        & industry_metrics["horizon"].eq(5)
    ]
    champion_sector = industry_metrics[
        industry_metrics["model"].eq("v6") & industry_metrics["horizon"].eq(5)
    ]
    gates = {
        "mean_rank_ic_meaningfully_better": bool(
            challenger["mean_rank_ic"]
            >= champion["mean_rank_ic"] + settings.minimum_rank_ic_improvement
        ),
        "rank_ic_ir_not_worse": bool(challenger["rank_ic_ir"] >= champion["rank_ic_ir"]),
        "positive_ratio_not_worse": bool(
            challenger["positive_rank_ic_ratio"] >= champion["positive_rank_ic_ratio"]
        ),
        "top20_net_alpha_better": bool(
            challenger_top["net_alpha_total"] > champion_top["net_alpha_total"]
        ),
        "rank_ic_bootstrap_lower_positive": bool(rank_bootstrap["ci_lower"] > 0),
        "top20_bootstrap_lower_positive": bool(top20_bootstrap["ci_lower"] > 0),
        "minimum_positive_years": positive_years >= settings.minimum_positive_years,
        "drawdown_not_materially_worse": bool(
            challenger_top["max_drawdown"]
            >= champion_top["max_drawdown"] - settings.maximum_drawdown_worsening
        ),
        "regime_stability_not_worse": bool(
            (candidate_regime["mean_rank_ic"] > 0).mean()
            >= (champion_regime["mean_rank_ic"] > 0).mean()
        ),
        "industry_stability_not_worse": bool(
            (candidate_sector["mean_rank_ic"] > 0).mean()
            >= (champion_sector["mean_rank_ic"] > 0).mean()
        ),
    }
    promotion = all(gates.values())
    decision = {
        "created_at_utc": _utc(),
        "champion": "V6",
        "challenger": "V31",
        "challenger_model": settings.pre_registered_challenger,
        "v31_status": "RESEARCH_ONLY",
        "historical_oos_passed": promotion,
        "promotion_to_candidate": promotion,
        "decision": "V31_RESEARCH_PASS_CANDIDATE_ELIGIBLE" if promotion else "V31_REJECTED",
        "gates": gates,
        "positive_years_vs_v6": positive_years,
        "benchmark_target": "DISABLED_BENCHMARK_EVIDENCE_UNAPPROVED",
        "production_prediction_ready": False,
        "execution_authorized": False,
        "v6_remains_champion": True,
        "v31_writes_to_prospective": False,
    }

    factor_audit = pd.concat(factor_audits, ignore_index=True)
    factor_stability = pd.DataFrame(factor_stability_rows)
    importance = pd.DataFrame(importance_rows)
    fold_payload = {
        "scheme": "nested yearly purged walk-forward with one validation year",
        "folds": fold_rows,
    }
    report = {
        "created_at_utc": _utc(),
        "model_id": settings.model_id,
        "role": settings.role,
        "status": settings.status,
        "protocol_sha256": sha256(settings.protocol_path),
        "plan_lock_sha256": frozen["lock_sha256"],
        "data": data_evidence,
        "oos_years": list(settings.oos_years),
        "final_oos_years": list(settings.final_oos_years),
        "horizons": list(settings.horizons),
        "factor_count_before_screening": len(settings.factor_columns),
        "selected_factors_by_year": {str(year): list(values) for year, values in selected_by_year.items()},
        "benchmark_target_status": "BENCHMARK_TARGET_DISABLED",
        "models": {
            "v6": "completed",
            "ridge": "completed",
            "lightgbm_regression": "completed",
            "lightgbm_lambdarank": "completed",
            "xgboost": "disabled_dependency_not_present",
            "catboost": "disabled_dependency_not_present",
        },
        "decision": decision,
        "safety": {
            "prospective_v1r4_modified": False,
            "v6_modified": False,
            "v30_logic_modified": False,
            "v30r1_modified": False,
            "v31_trained_for_research_only": True,
            "model_training_runs_are_research_only": True,
            "factor_research_used_prospective_results": False,
            "production_prediction_ready": False,
            "execution_authorized": False,
        },
    }

    outputs = {
        "factor_inventory.csv": inventory,
        "factor_audit.csv": factor_audit,
        "factor_stability.csv": factor_stability,
        "factor_correlation.csv": correlation_last.reset_index(names="factor_name"),
        "model_comparison.csv": model_comparison,
        "yearly_metrics.csv": yearly_metrics,
        "regime_metrics.csv": regime_metrics,
        "industry_metrics.csv": industry_metrics,
        "topk_metrics.csv": topk_metrics,
        "quantile_metrics.csv": quantile_metrics,
        "turnover_metrics.csv": turnover_metrics,
        "feature_importance.csv": importance,
    }
    for name, frame in outputs.items():
        _write_csv_new(settings.artifact_dir / name, frame)
    _write_json_new(settings.artifact_dir / "walk_forward_folds.json", fold_payload)
    _write_json_new(settings.artifact_dir / "bootstrap.json", bootstrap)
    _write_json_new(decision_path, decision)
    _write_json_new(settings.artifact_dir / "report.json", report)
    artifact_hashes = {
        path.name: sha256(path)
        for path in sorted(settings.artifact_dir.iterdir())
        if path.is_file() and path.name not in {"artifact_manifest.json", "artifact_manifest.json.sha256"}
    }
    manifest = {
        "created_at_utc": _utc(),
        "immutable": True,
        "files": artifact_hashes,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    _write_json_new(settings.artifact_dir / "artifact_manifest.json", manifest)
    (settings.artifact_dir / "artifact_manifest.json.sha256").write_text(
        sha256(settings.artifact_dir / "artifact_manifest.json") + "\n", encoding="ascii"
    )
    return {
        "decision": decision["decision"],
        "promotion_to_candidate": promotion,
        "champion": "V6",
        "v31_status": "RESEARCH_ONLY",
        "production_prediction_ready": False,
        "execution_authorized": False,
        "artifact_manifest_sha256": sha256(settings.artifact_dir / "artifact_manifest.json"),
    }
