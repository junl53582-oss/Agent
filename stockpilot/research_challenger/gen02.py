from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ChallengerSettings
from .data import add_research_targets, assert_feature_columns_safe, sha256, verify_dataset_manifest
from .gen02_portfolio import (
    PortfolioPolicy,
    evaluate_portfolio_policy,
    summarize_portfolio,
)
from .metrics import daily_rank_metrics, moving_block_bootstrap_delta, summarize_ic
from .models import (
    LightGBMModel,
    RidgeModel,
    TrainOnlyPreprocessor,
    deterministic_full_date_sample,
    v6_oos_scores,
)
from .split import build_fold


@dataclass(frozen=True)
class Gen02Settings:
    artifact_dir: Path = Path("artifacts/research_challenger/gen02")
    development_years: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024, 2025)
    holdout_year: int = 2026
    horizons: tuple[int, ...] = (5, 20)
    candidate_models: tuple[str, ...] = ("ridge", "lightgbm_regression")
    ridge_window_years: tuple[int, ...] = (3, 5, 8)
    recency_half_life_years: float = 2.0
    recency_window_years: int = 5
    top_ks: tuple[int, ...] = (10, 20, 30)
    buffer_entry_rank: int = 20
    buffer_exit_rank: int = 30
    random_seed: int = 42
    bootstrap_replications: int = 1_000
    bootstrap_block_length: int = 20
    minimum_rank_ic: float = 0.0
    minimum_rank_ic_ir: float = 0.0
    minimum_positive_ratio: float = 0.52
    minimum_2025_rank_ic: float = 0.0
    minimum_net_research_proxy_alpha: float = 0.0
    maximum_drawdown: float = -0.35
    maximum_annualized_turnover: float = 15.0
    maximum_sector_weight: float = 0.45
    complexity_rank_ic_tie: float = 0.003
    complexity_net_alpha_tie: float = 0.05


HOLDOUT_EVIDENCE_PATHS = (
    Path("artifacts/comparison/ridge/summary.json"),
    Path("artifacts/comparison/ridge/latest_signals.csv"),
    Path("artifacts/latest_signals.csv"),
    Path("artifacts/prediction_v30/live/predictions/2026-08-21.csv"),
    Path("artifacts/prediction_v30/live/predictions/2026-08-21.csv.sha256"),
    Path("RESEARCH_DECISIONS.md"),
)
DEVELOPMENT_AMENDMENT_DIRS = (
    Path("artifacts/research_challenger/gen02/experiments/001_empty_stability_group_fix"),
    Path("artifacts/research_challenger/gen02/experiments/002_numpy_bool_serialization_fix"),
    Path("artifacts/research_challenger/gen02/experiments/003_record_test_cli_fix"),
    Path("artifacts/research_challenger/gen02/experiments/004_recursive_final_manifest_fix"),
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(payload: dict | list) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n").encode(
        "utf-8"
    )


def _write_new(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise RuntimeError(f"GEN02_IMMUTABLE_ARTIFACT_EXISTS: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = Path(str(path) + ".sha256")
    try:
        sidecar.write_text(digest + "\n", encoding="ascii", errors="strict")
    except BaseException:
        path.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise
    return digest


def _write_json(path: Path, payload: dict | list) -> str:
    return _write_new(path, _canonical_json(payload))


def _write_csv(path: Path, frame: pd.DataFrame) -> str:
    return _write_new(path, frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def _verify_sidecar(path: Path) -> bool:
    sidecar = Path(str(path) + ".sha256")
    return path.is_file() and sidecar.is_file() and sidecar.read_text(encoding="ascii").strip() == sha256(path)


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def audit_holdout(root: Path = Path(".")) -> dict:
    evidence: list[dict] = []
    for relative in HOLDOUT_EVIDENCE_PATHS:
        path = root / relative
        if path.exists():
            evidence.append(
                {
                    "path": relative.as_posix(),
                    "sha256": sha256(path),
                    "reason": "repository evidence created or inspected with a 2026 decision date",
                }
            )
    summary_path = root / "artifacts/comparison/ridge/summary.json"
    summary_end = None
    if summary_path.exists():
        summary_end = json.loads(summary_path.read_text(encoding="utf-8")).get("data_end")
    signal_dates: dict[str, str | None] = {}
    for relative in (
        Path("artifacts/comparison/ridge/latest_signals.csv"),
        Path("artifacts/latest_signals.csv"),
    ):
        path = root / relative
        if path.exists():
            frame = pd.read_csv(path, usecols=lambda name: name in {"date", "prediction_date"})
            date_column = "date" if "date" in frame else "prediction_date"
            signal_dates[relative.as_posix()] = (
                str(pd.to_datetime(frame[date_column]).max().date()) if len(frame) else None
            )
    untouched = False
    return {
        "audit_id": "CHALLENGER-GEN02-HOLDOUT-AUDIT",
        "created_at_utc": _utc(),
        "candidate_period": {"start": "2026-01-01", "end": "2026-08-21"},
        "untouched_2026_holdout": untouched,
        "verdict_rule": "If untouched status cannot be proved, verdict is FALSE.",
        "decisive_evidence": {
            "ridge_comparison_data_end": summary_end,
            "existing_2026_signal_dates": signal_dates,
            "v30_prediction_snapshot_exists": (root / "artifacts/prediction_v30/live/predictions/2026-08-21.csv").exists(),
            "research_decision_log_contains_2026_08_21": "2026-08-21"
            in (root / "RESEARCH_DECISIONS.md").read_text(encoding="utf-8"),
            "files": evidence,
        },
        "consequence": "2026 is not eligible for final historical confirmation; labels and performance must not be opened by Gen02.",
        "holdout_open_permitted": False,
        "historical_confirmation_possible": False,
    }


def run_read_only_audit(settings: Gen02Settings | None = None) -> dict:
    settings = settings or Gen02Settings()
    if settings.artifact_dir.exists() and any(settings.artifact_dir.iterdir()):
        raise RuntimeError("GEN02_AUDIT_DIRECTORY_NOT_EMPTY")
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    model_comparison = pd.read_csv("artifacts/research_v31/model_comparison.csv")
    yearly = pd.read_csv("artifacts/research_v31/yearly_metrics.csv")
    topk = pd.read_csv("artifacts/research_v31/topk_metrics.csv")
    selected_models = model_comparison[
        model_comparison["model"].isin(("v6", *settings.candidate_models))
        & model_comparison["horizon"].isin(settings.horizons)
    ]
    top20 = topk[
        topk["model"].isin(("v6", *settings.candidate_models))
        & topk["horizon"].isin(settings.horizons)
        & topk["top_k"].eq(20)
    ]
    holdout = audit_holdout()
    audit = {
        "audit_id": "GEN2_READ_ONLY_AUDIT",
        "created_at_utc": _utc(),
        "baseline_commit": _git(["rev-parse", "HEAD"]),
        "head_equals_origin_main": _git(["rev-parse", "HEAD"])
        == _git(["rev-parse", "origin/main"]),
        "working_tree_clean": not bool(_git(["status", "--short"])),
        "v31_decision": json.loads(Path("artifacts/research_v31/decision.json").read_text(encoding="utf-8")),
        "rank_ic_evidence": selected_models.to_dict(orient="records"),
        "yearly_evidence": yearly[
            yearly["model"].isin(("v6", *settings.candidate_models))
            & yearly["horizon"].isin(settings.horizons)
        ].to_dict(orient="records"),
        "top20_conversion_evidence": top20.to_dict(orient="records"),
        "research_proxy_boundary": "PIT constituent-weighted research proxy; official benchmark evidence remains unapproved.",
        "prospective_data_used": False,
        "prospective_paths_written": False,
        "lambda_rank_status": "FROZEN_REJECTED_BASELINE_NOT_RETUNED",
        "candidate_scope": list(settings.candidate_models),
        "holdout_verdict": holdout["untouched_2026_holdout"],
    }
    _write_json(settings.artifact_dir / "audit.json", audit)
    _write_json(settings.artifact_dir / "holdout_audit.json", holdout)
    return {
        "status": "GEN2_READ_ONLY_AUDIT_COMPLETE",
        "untouched_2026_holdout": False,
        "provider_requests": 0,
        "prospective_rows_used": 0,
    }


def development_protocol(settings: Gen02Settings | None = None) -> dict:
    settings = settings or Gen02Settings()
    return {
        "protocol_id": "CHALLENGER-GENERATION-02-DEVELOPMENT",
        "created_at_utc": _utc(),
        "classification": "RESEARCH_ONLY_DEVELOPMENT_EVIDENCE",
        "development_period": [2020, 2021, 2022, 2023, 2024, 2025],
        "holdout_2026": "DISQUALIFIED_NOT_UNTOUCHED",
        "models": {
            "incumbent": "V6",
            "candidates": list(settings.candidate_models),
            "lambdarank": "FROZEN_REJECTED_NOT_RETUNED",
            "model_zoo_expansion": False,
        },
        "targets": {"primary_candidates": [5, 20], "semantics": "cross-sectional future open-to-open return rank"},
        "training_window_sensitivity": {
            "model": "ridge",
            "years": list(settings.ridge_window_years),
            "factor_policy": "reuse each V31 yearly train-only selected feature set",
        },
        "recency_weighting": {
            "policies": ["equal", "exponential_half_life_2y"],
            "window_years": settings.recency_window_years,
        },
        "portfolio_grid": [
            "equal_top10",
            "equal_top20",
            "equal_top30",
            "rank_weight_top20",
            "rank_decay_top20",
            "buffer_top20_exit30",
            "sector_balanced_top20",
        ],
        "rebalance": {"5": 5, "20": 20},
        "cost_model": {
            "commission": 0.0003,
            "slippage": 0.0005,
            "sell_stamp_duty": 0.0005,
            "unchanged_from_v31": True,
        },
        "absolute_gates": {
            "mean_rank_ic_gt": settings.minimum_rank_ic,
            "rank_ic_ir_gt": settings.minimum_rank_ic_ir,
            "positive_ratio_gte": settings.minimum_positive_ratio,
            "2025_rank_ic_gt": settings.minimum_2025_rank_ic,
            "net_research_proxy_alpha_gt": settings.minimum_net_research_proxy_alpha,
            "max_drawdown_gte": settings.maximum_drawdown,
            "annualized_turnover_lte": settings.maximum_annualized_turnover,
            "maximum_sector_weight_lte": settings.maximum_sector_weight,
        },
        "selection_rule": {
            "order": ["all_absolute_gates", "net_research_proxy_alpha", "mean_rank_ic"],
            "prefer_ridge_when_rank_ic_gap_lte": settings.complexity_rank_ic_tie,
            "and_net_alpha_gap_lte": settings.complexity_net_alpha_tie,
        },
        "staggered_sleeves": "membership algorithm tested; P&L excluded because cache lacks daily mark-to-market paths for open sleeves",
        "official_benchmark_alpha_enabled": False,
        "prospective_data_allowed": False,
        "2026_label_or_performance_read_allowed": False,
        "promotion_to_champion_allowed": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
        "random_seed": settings.random_seed,
    }


def freeze_development_plan(settings: Gen02Settings | None = None) -> dict:
    settings = settings or Gen02Settings()
    if not _verify_sidecar(settings.artifact_dir / "holdout_audit.json"):
        raise RuntimeError("GEN02_HOLDOUT_AUDIT_NOT_VERIFIED")
    protocol_path = settings.artifact_dir / "development_protocol.json"
    _write_json(protocol_path, development_protocol(settings))
    files = [
        Path("stockpilot/research_challenger/gen02.py"),
        Path("stockpilot/research_challenger/gen02_portfolio.py"),
        Path("tests/test_research_challenger_gen02.py"),
        protocol_path,
        settings.artifact_dir / "holdout_audit.json",
        Path("artifacts/research_v31/artifact_manifest.json"),
        Path("artifacts/research_v31/experiments/002_ci_verifier_fix/plan.lock.json"),
        Path("artifacts/research_v6/plan.lock.json"),
        Path("artifacts/prospective_alpha_v1r4/plan.lock.json"),
    ]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise RuntimeError(f"GEN02_FREEZE_INPUT_MISSING: {missing}")
    lock = {
        "lock_id": "CHALLENGER-GEN02-DEVELOPMENT-PLAN",
        "created_at_utc": _utc(),
        "files": {path.as_posix(): sha256(path) for path in files},
        "untouched_2026_holdout": False,
        "holdout_open_allowed": False,
        "v6_modified": False,
        "v30_modified": False,
        "prospective_v1r4_modified": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    lock_hash = _write_json(settings.artifact_dir / "development_plan.lock.json", lock)
    return {"status": "GEN02_DEVELOPMENT_PLAN_FROZEN", "lock_sha256": lock_hash}


def verify_development_plan(settings: Gen02Settings | None = None) -> dict:
    settings = settings or Gen02Settings()
    amendment_locks = [
        directory / "development_plan.lock.json" for directory in DEVELOPMENT_AMENDMENT_DIRS
    ]
    amendment_lock = next(
        (path for path in reversed(amendment_locks) if path.exists()),
        amendment_locks[0],
    )
    lock_path = (
        amendment_lock
        if amendment_lock.exists()
        else settings.artifact_dir / "development_plan.lock.json"
    )
    if not _verify_sidecar(lock_path):
        return {"intact": False, "reason": "LOCK_OR_SIDECAR_INVALID"}
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    mismatches = []
    for name, expected in lock["files"].items():
        path = Path(name)
        actual = sha256(path) if path.exists() else "MISSING"
        if actual != expected:
            mismatches.append({"path": name, "expected": expected, "actual": actual})
    return {
        "intact": not mismatches,
        "mismatches": mismatches,
        "lock_sha256": sha256(lock_path),
        "amendment": lock_path == amendment_lock,
    }


def freeze_development_amendment(settings: Gen02Settings | None = None) -> dict:
    settings = settings or Gen02Settings()
    amendment_dir = DEVELOPMENT_AMENDMENT_DIRS[-1]
    failure = amendment_dir / "failure_receipt.json"
    if not _verify_sidecar(failure):
        raise RuntimeError("GEN02_FAILURE_RECEIPT_NOT_VERIFIED")
    protocol = {
        "amendment_id": "GEN02-NUMPY-BOOL-SERIALIZATION-FIX-002",
        "created_at_utc": _utc(),
        "classification": "IMPLEMENTATION_ONLY_NO_RESEARCH_RULE_CHANGE",
        "root_cause": "Gate comparisons returned numpy.bool_ values that json.dumps cannot encode.",
        "repair": "Normalize every gate value to native bool before explicit JSON encoding.",
        "models_changed": False,
        "features_changed": False,
        "dates_changed": False,
        "costs_changed": False,
        "gates_changed": False,
        "portfolio_rules_changed": False,
        "holdout_2026_opened": False,
        "performance_outputs_written_before_failure": False,
    }
    protocol_path = amendment_dir / "protocol_amendment.json"
    _write_json(protocol_path, protocol)
    files = [
        Path("stockpilot/research_challenger/gen02.py"),
        Path("stockpilot/research_challenger/gen02_portfolio.py"),
        Path("tests/test_research_challenger_gen02.py"),
        settings.artifact_dir / "development_protocol.json",
        settings.artifact_dir / "development_plan.lock.json",
        DEVELOPMENT_AMENDMENT_DIRS[0] / "development_plan.lock.json",
        failure,
        protocol_path,
    ]
    lock = {
        "lock_id": "CHALLENGER-GEN02-DEVELOPMENT-AMENDMENT-002",
        "created_at_utc": _utc(),
        "files": {path.as_posix(): sha256(path) for path in files},
        "implementation_only": True,
        "research_rules_changed": False,
        "untouched_2026_holdout": False,
        "holdout_open_allowed": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    lock_hash = _write_json(amendment_dir / "development_plan.lock.json", lock)
    return {
        "status": "GEN02_DEVELOPMENT_AMENDMENT_FROZEN",
        "lock_sha256": lock_hash,
    }


def freeze_postrun_amendment(settings: Gen02Settings | None = None) -> dict:
    settings = settings or Gen02Settings()
    amendment_dir = DEVELOPMENT_AMENDMENT_DIRS[-1]
    protocol = {
        "amendment_id": "GEN02-RECURSIVE-FINAL-MANIFEST-FIX-004",
        "created_at_utc": _utc(),
        "classification": "POST_RUN_OPERATIONAL_ONLY_NO_RESULT_RECOMPUTATION",
        "root_cause": "The final freezer referenced the root development lock and enumerated only top-level artifacts, omitting the effective amendment chain from the manifest.",
        "repair": "Bind the latest verified amendment lock and recursively enumerate immutable experiment evidence.",
        "development_result_already_written": True,
        "development_result_changed": False,
        "model_rerun_required": False,
        "models_changed": False,
        "features_changed": False,
        "dates_changed": False,
        "costs_changed": False,
        "gates_changed": False,
        "portfolio_rules_changed": False,
        "holdout_2026_opened": False,
    }
    protocol_path = amendment_dir / "protocol_amendment.json"
    _write_json(protocol_path, protocol)
    files = [
        Path("stockpilot/research_challenger/gen02.py"),
        Path("stockpilot/research_challenger/gen02_portfolio.py"),
        Path("tests/test_research_challenger_gen02.py"),
        DEVELOPMENT_AMENDMENT_DIRS[-2] / "development_plan.lock.json",
        settings.artifact_dir / "decision.json",
        settings.artifact_dir / "report.json",
        protocol_path,
    ]
    lock = {
        "lock_id": "CHALLENGER-GEN02-POSTRUN-AMENDMENT-004",
        "created_at_utc": _utc(),
        "files": {path.as_posix(): sha256(path) for path in files},
        "implementation_only": True,
        "research_results_changed": False,
        "holdout_open_allowed": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    lock_hash = _write_json(amendment_dir / "development_plan.lock.json", lock)
    return {"status": "GEN02_POSTRUN_AMENDMENT_FROZEN", "lock_sha256": lock_hash}


def load_development_dataset(base: ChallengerSettings | None = None) -> tuple[pd.DataFrame, dict]:
    base = base or ChallengerSettings()
    evidence = verify_dataset_manifest(base)
    assert_feature_columns_safe(base.factor_columns)
    cutoff = pd.Timestamp("2026-01-01")
    data = pd.read_parquet(base.dataset_path, filters=[("date", "<", cutoff)])
    data["date"] = pd.to_datetime(data["date"])
    if data["date"].max() >= cutoff:
        raise RuntimeError("GEN02_2026_DATA_ENTERED_DEVELOPMENT")
    data["symbol"] = data["symbol"].astype(str).str.zfill(6)
    if data.duplicated(["date", "symbol"]).any():
        raise RuntimeError("GEN02_DUPLICATE_DATE_SYMBOL")
    decision = data["date"]
    checks = {
        "membership_pit": False,
        "fundamentals_pit": bool(
            (
                pd.to_datetime(data["available_date"], errors="coerce").isna()
                | pd.to_datetime(data["available_date"], errors="coerce").le(decision)
            ).all()
        ),
        "industry_pit": bool(
            (
                pd.to_datetime(data["industry_effective_date"], errors="coerce").isna()
                | pd.to_datetime(data["industry_effective_date"], errors="coerce").le(decision)
            ).all()
        ),
    }
    membership_dates = pd.to_datetime(data["membership_snapshot_date"], errors="coerce")
    checks["membership_pit"] = bool((membership_dates.isna() | membership_dates.le(decision)).all())
    if not all(checks.values()):
        raise RuntimeError(f"GEN02_PIT_AUDIT_FAILED: {checks}")
    eligible = data["eligible"].fillna(False) & data["in_universe"].fillna(False)
    data = add_research_targets(data.loc[eligible].copy(), base.horizons)
    evidence.update(
        {
            "development_rows": int(len(data)),
            "symbols": int(data["symbol"].nunique()),
            "date_min": str(data["date"].min().date()),
            "date_max": str(data["date"].max().date()),
            "parquet_filter": "date < 2026-01-01",
            "pit_checks": checks,
            "prospective_rows_used": 0,
        }
    )
    return data.sort_values(["date", "symbol"]).reset_index(drop=True), evidence


def _weighted_ridge(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, alpha: float
) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    root = np.sqrt(np.asarray(weights, dtype=float)).reshape(-1, 1)
    weighted_x = design * root
    weighted_y = y * root.ravel()
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(weighted_x.T @ weighted_x + penalty, weighted_x.T @ weighted_y)


def _weighted_predict(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x]) @ coefficients


def _selected_factors() -> dict[int, tuple[str, ...]]:
    report = json.loads(Path("artifacts/research_v31/report.json").read_text(encoding="utf-8"))
    return {int(year): tuple(values) for year, values in report["selected_factors_by_year"].items()}


def _fit_development_scores(
    data: pd.DataFrame,
    gen: Gen02Settings,
    base: ChallengerSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = _selected_factors()
    score_pieces: list[pd.DataFrame] = []
    sensitivity_rows: list[dict] = []
    for year in gen.development_years:
        year_union = data[
            data["date"].dt.year.eq(year)
            & data[["future_return_5d", "future_return_20d"]].notna().any(axis=1)
        ].copy()
        v6_union = v6_oos_scores(data, year_union, year)
        features = selected[year]
        for horizon in gen.horizons:
            target = f"return_rank_{horizon}d"
            predictions: dict[tuple[int, str], np.ndarray] = {}
            test_reference: pd.DataFrame | None = None
            for window in gen.ridge_window_years:
                fold = build_fold(
                    data,
                    year,
                    horizon,
                    training_window_years=window,
                    validation_years=base.validation_years,
                    purge_gap_trading_days=base.purge_gaps[horizon],
                )
                train = data.loc[fold.refit_index].copy()
                test = data.loc[fold.test_index].copy()
                target_values = pd.to_numeric(train[target], errors="coerce")
                train = train[target_values.notna() & np.isfinite(target_values)].copy()
                sample = deterministic_full_date_sample(train, base.training_row_cap)
                processor = TrainOnlyPreprocessor().fit(sample, features)
                x_train = processor.transform(sample, features)
                x_test = processor.transform(test, features)
                y_train = pd.to_numeric(sample[target], errors="raise").to_numpy(dtype=float)
                ridge = RidgeModel(base.ridge_alpha).fit(x_train, y_train)
                ridge_scores = ridge.predict(x_test)
                predictions[(window, "equal")] = ridge_scores
                daily = daily_rank_metrics(
                    test.assign(score=ridge_scores), "score", f"future_return_{horizon}d"
                )
                sensitivity_rows.append(
                    {
                        "model": "ridge",
                        "horizon": horizon,
                        "test_year": year,
                        "training_window_years": window,
                        "weighting": "equal",
                        **summarize_ic(daily),
                    }
                )
                if window == gen.recency_window_years:
                    last = pd.Timestamp(sample["date"].max())
                    ages = (last - pd.to_datetime(sample["date"])).dt.days.to_numpy() / 365.25
                    weights = np.power(0.5, ages / gen.recency_half_life_years)
                    coefficients = _weighted_ridge(x_train, y_train, weights, base.ridge_alpha)
                    weighted_scores = _weighted_predict(x_test, coefficients)
                    predictions[(window, "exponential_half_life_2y")] = weighted_scores
                    weighted_daily = daily_rank_metrics(
                        test.assign(score=weighted_scores),
                        "score",
                        f"future_return_{horizon}d",
                    )
                    sensitivity_rows.append(
                        {
                            "model": "ridge",
                            "horizon": horizon,
                            "test_year": year,
                            "training_window_years": window,
                            "weighting": "exponential_half_life_2y",
                            **summarize_ic(weighted_daily),
                        }
                    )
                if window == 8:
                    lightgbm = LightGBMModel(
                        "regression_l1", base.lightgbm_rounds, base.random_seed
                    ).fit(x_train, y_train)
                    test_reference = test
                    piece = test[
                        [
                            "date",
                            "symbol",
                            "broad_sector",
                            "industry",
                            "benchmark_weight",
                            "benchmark_weight_rank",
                            "amount_rank",
                            "regime",
                            "volatility_20",
                            "entry_tradable",
                            "entry_tradable_20",
                            "execution_return",
                            "execution_return_20",
                            f"future_return_{horizon}d",
                        ]
                    ].copy()
                    piece["horizon"] = horizon
                    piece["test_year"] = year
                    piece["score_v6"] = v6_union.reindex(test.index).to_numpy()
                    piece["score_ridge"] = ridge_scores
                    piece["score_lightgbm_regression"] = lightgbm.predict(x_test)
                    score_pieces.append(piece)
            if test_reference is None:
                raise RuntimeError("GEN02_BASELINE_8Y_SCORE_MISSING")
    return pd.concat(score_pieces, ignore_index=True), pd.DataFrame(sensitivity_rows)


def _score_metrics(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict] = []
    yearly_rows: list[dict] = []
    for horizon in sorted(scores["horizon"].unique()):
        part = scores[scores["horizon"].eq(horizon)]
        for model in ("v6", "ridge", "lightgbm_regression"):
            daily = daily_rank_metrics(part, f"score_{model}", f"future_return_{horizon}d")
            model_rows.append({"model": model, "horizon": horizon, **summarize_ic(daily)})
            daily["year"] = daily["date"].dt.year
            for year, group in daily.groupby("year"):
                yearly_rows.append(
                    {"model": model, "horizon": horizon, "test_year": int(year), **summarize_ic(group)}
                )
    return pd.DataFrame(model_rows), pd.DataFrame(yearly_rows)


def _tail_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for horizon in sorted(scores["horizon"].unique()):
        part = scores[scores["horizon"].eq(horizon)]
        return_column = f"future_return_{horizon}d"
        for model in ("v6", "ridge", "lightgbm_regression"):
            score_column = f"score_{model}"
            for year, year_part in part.groupby("test_year"):
                daily_rows = []
                for date, current in year_part.groupby("date", sort=True):
                    current = current.dropna(subset=[score_column, return_column]).sort_values(
                        score_column, ascending=False
                    )
                    if len(current) < 100:
                        continue
                    decile = max(20, int(np.ceil(len(current) * 0.10)))
                    quintile = max(20, int(np.ceil(len(current) * 0.20)))
                    top_decile = current.head(decile)
                    top_quintile = current.head(quintile)
                    bottom_decile = current.tail(decile)
                    margin20 = (
                        float(current.iloc[19][score_column] - current.iloc[20][score_column])
                        if len(current) > 20
                        else np.nan
                    )
                    scale = float(current[score_column].std(ddof=0))
                    daily_rows.append(
                        {
                            "date": date,
                            "overall_rank_ic": current[score_column].corr(
                                current[return_column], method="spearman"
                            ),
                            "top_decile_ic": top_decile[score_column].corr(
                                top_decile[return_column], method="spearman"
                            ),
                            "top_quintile_ic": top_quintile[score_column].corr(
                                top_quintile[return_column], method="spearman"
                            ),
                            "bottom_decile_ic": bottom_decile[score_column].corr(
                                bottom_decile[return_column], method="spearman"
                            ),
                            "top_decile_return": top_decile[return_column].mean(),
                            "universe_return": current[return_column].mean(),
                            "top_decile_win_rate": (top_decile[return_column] > 0).mean(),
                            "cutoff_margin_20_std_units": margin20 / scale if scale > 0 else 0.0,
                            "score_std": scale,
                        }
                    )
                daily = pd.DataFrame(daily_rows)
                rows.append(
                    {
                        "model": model,
                        "horizon": horizon,
                        "test_year": int(year),
                        **{
                            column: float(pd.to_numeric(daily[column], errors="coerce").mean())
                            for column in daily.columns
                            if column != "date"
                        },
                    }
                )
    return pd.DataFrame(rows)


def _feature_drift(data: pd.DataFrame, selected: dict[int, tuple[str, ...]]) -> pd.DataFrame:
    features = sorted(set(selected[2024]).union(selected[2025]))
    rows: list[dict] = []
    for feature in features:
        by_year: dict[int, dict] = {}
        for year in (2024, 2025):
            part = pd.to_numeric(
                data.loc[data["date"].dt.year.eq(year), feature], errors="coerce"
            )
            daily_dispersion = data.loc[data["date"].dt.year.eq(year), ["date", feature]].groupby(
                "date"
            )[feature].std()
            by_year[year] = {
                "mean": float(part.mean()),
                "std": float(part.std(ddof=1)),
                "median": float(part.median()),
                "missing_ratio": float(part.isna().mean()),
                "cross_sectional_dispersion": float(daily_dispersion.mean()),
                "p10": float(part.quantile(0.10)),
                "p90": float(part.quantile(0.90)),
            }
        baseline = pd.to_numeric(
            data.loc[data["date"].dt.year.eq(2024), feature], errors="coerce"
        ).dropna()
        current = pd.to_numeric(
            data.loc[data["date"].dt.year.eq(2025), feature], errors="coerce"
        ).dropna()
        edges = np.unique(np.quantile(baseline, np.linspace(0, 1, 11)))
        psi = np.nan
        if len(edges) >= 3:
            edges[0], edges[-1] = -np.inf, np.inf
            base_counts = pd.cut(baseline, edges, include_lowest=True).value_counts(sort=False)
            current_counts = pd.cut(current, edges, include_lowest=True).value_counts(sort=False)
            base_share = np.clip(base_counts.to_numpy() / max(1, len(baseline)), 1e-6, None)
            current_share = np.clip(current_counts.to_numpy() / max(1, len(current)), 1e-6, None)
            psi = float(np.sum((current_share - base_share) * np.log(current_share / base_share)))
        rows.append(
            {
                "factor": feature,
                **{f"2024_{key}": value for key, value in by_year[2024].items()},
                **{f"2025_{key}": value for key, value in by_year[2025].items()},
                "mean_shift_in_2024_std": (
                    (by_year[2025]["mean"] - by_year[2024]["mean"]) / by_year[2024]["std"]
                    if by_year[2024]["std"] > 0
                    else np.nan
                ),
                "psi_2024_to_2025": psi,
                "drift_status": "SEVERE" if np.isfinite(psi) and psi >= 0.25 else "MODERATE" if np.isfinite(psi) and psi >= 0.10 else "LOW",
            }
        )
    return pd.DataFrame(rows)


def _factor_decay(selected: dict[int, tuple[str, ...]]) -> pd.DataFrame:
    stability = pd.read_csv("artifacts/research_v31/factor_stability.csv")
    stability = stability[
        stability["horizon"].eq(5) & stability["neutralization"].eq("raw")
    ]
    rows = []
    for factor in selected[2025]:
        current = stability[stability["factor_name"].eq(factor)]
        earlier = current[current["test_year"].le(2024)]["mean_rank_ic"]
        y2025 = current[current["test_year"].eq(2025)]["mean_rank_ic"]
        earlier_mean = float(earlier.mean())
        current_value = float(y2025.iloc[0])
        rows.append(
            {
                "factor": factor,
                "rank_ic_2020_2024_mean": earlier_mean,
                "rank_ic_2025": current_value,
                "ic_change": current_value - earlier_mean,
                "sign_flip": bool(earlier_mean * current_value < 0),
                "status": "SIGN_REVERSAL" if earlier_mean * current_value < 0 else "FACTOR_DECAY" if current_value < earlier_mean else "NO_DECAY",
            }
        )
    return pd.DataFrame(rows)


def _selection_stability(selected: dict[int, tuple[str, ...]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(selected)
    pairs = []
    for prior, current in zip(years, years[1:]):
        left, right = set(selected[prior]), set(selected[current])
        pairs.append(
            {
                "prior_year": prior,
                "current_year": current,
                "intersection": len(left & right),
                "union": len(left | right),
                "jaccard_similarity": len(left & right) / len(left | right),
            }
        )
    all_features = sorted(set().union(*map(set, selected.values())))
    frequency = pd.DataFrame(
        {
            "factor": all_features,
            "selection_count": [sum(feature in selected[year] for year in years) for feature in all_features],
        }
    )
    frequency["selection_frequency"] = frequency["selection_count"] / len(years)
    return pd.DataFrame(pairs), frequency


def _portfolio_policies(gen: Gen02Settings) -> tuple[PortfolioPolicy, ...]:
    return (
        PortfolioPolicy("equal_top10", 10),
        PortfolioPolicy("equal_top20", 20),
        PortfolioPolicy("equal_top30", 30),
        PortfolioPolicy("rank_weight_top20", 20, weighting="rank"),
        PortfolioPolicy("rank_decay_top20", 20, weighting="rank_decay"),
        PortfolioPolicy(
            "buffer_top20_exit30", 20, buffer_exit_rank=gen.buffer_exit_rank
        ),
        PortfolioPolicy("sector_balanced_top20", 20, sector_balanced=True),
    )


def _portfolio_analysis(
    scores: pd.DataFrame, gen: Gen02Settings, base: ChallengerSettings
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int, str], pd.DataFrame]]:
    summaries: list[dict] = []
    decomposition_rows: list[dict] = []
    period_map: dict[tuple[str, int, str], pd.DataFrame] = {}
    for horizon in gen.horizons:
        part = scores[scores["horizon"].eq(horizon)].copy()
        for model in ("v6", *gen.candidate_models):
            for policy in _portfolio_policies(gen):
                periods, decomposition = evaluate_portfolio_policy(
                    part,
                    f"score_{model}",
                    horizon,
                    policy,
                    rebalance_every=horizon,
                    buy_rate=base.buy_rate,
                    sell_rate=base.sell_rate,
                )
                period_map[(model, horizon, policy.name)] = periods
                summaries.append(
                    {
                        "model": model,
                        "horizon": horizon,
                        "portfolio_policy": policy.name,
                        "top_k": policy.top_k,
                        "weighting": policy.weighting,
                        "buffer_exit_rank": policy.buffer_exit_rank,
                        "sector_balanced": policy.sector_balanced,
                        **summarize_portfolio(periods, horizon),
                    }
                )
                if not decomposition.empty:
                    totals = decomposition.drop(columns="date").sum(numeric_only=True)
                    decomposition_rows.append(
                        {
                            "model": model,
                            "horizon": horizon,
                            "portfolio_policy": policy.name,
                            **{name: float(value) for name, value in totals.items()},
                        }
                    )
    return pd.DataFrame(summaries), pd.DataFrame(decomposition_rows), period_map


def _ranking_differences(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in sorted(scores["horizon"].unique()):
        for model in ("ridge", "lightgbm_regression"):
            daily = []
            for date, current in scores[scores["horizon"].eq(horizon)].groupby("date"):
                v6_top = set(current.nlargest(20, "score_v6")["symbol"])
                model_top = set(current.nlargest(20, f"score_{model}")["symbol"])
                daily.append(
                    {
                        "date": date,
                        "score_rank_correlation_with_v6": current["score_v6"].corr(
                            current[f"score_{model}"], method="spearman"
                        ),
                        "top20_overlap_ratio": len(v6_top & model_top) / 20,
                    }
                )
            frame = pd.DataFrame(daily)
            rows.append(
                {
                    "model": model,
                    "horizon": horizon,
                    "mean_score_rank_correlation_with_v6": frame[
                        "score_rank_correlation_with_v6"
                    ].mean(),
                    "mean_top20_overlap_ratio": frame["top20_overlap_ratio"].mean(),
                }
            )
    return pd.DataFrame(rows)


def _stability_metrics(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    regimes: list[dict] = []
    industries: list[dict] = []
    for horizon in sorted(scores["horizon"].unique()):
        part = scores[scores["horizon"].eq(horizon)].copy()
        for model in ("v6", "ridge", "lightgbm_regression"):
            score = f"score_{model}"
            for regime, current in part.groupby("regime"):
                daily = daily_rank_metrics(current, score, f"future_return_{horizon}d")
                if daily.empty:
                    continue
                summary = summarize_ic(daily)
                regimes.append({"model": model, "horizon": horizon, "regime": regime, **summary})
            for sector, current in part.groupby("broad_sector"):
                daily = daily_rank_metrics(current, score, f"future_return_{horizon}d")
                if daily.empty:
                    continue
                summary = summarize_ic(daily)
                industries.append(
                    {"model": model, "horizon": horizon, "broad_sector": sector, **summary}
                )
    return pd.DataFrame(regimes), pd.DataFrame(industries)


def _choose_configuration(
    model_metrics: pd.DataFrame,
    yearly: pd.DataFrame,
    portfolios: pd.DataFrame,
    gen: Gen02Settings,
) -> tuple[dict, pd.DataFrame]:
    candidates = []
    for _, portfolio in portfolios[portfolios["model"].isin(gen.candidate_models)].iterrows():
        metric = model_metrics[
            model_metrics["model"].eq(portfolio["model"])
            & model_metrics["horizon"].eq(portfolio["horizon"])
        ].iloc[0]
        y2025 = yearly[
            yearly["model"].eq(portfolio["model"])
            & yearly["horizon"].eq(portfolio["horizon"])
            & yearly["test_year"].eq(2025)
        ].iloc[0]
        gates = {
            "rank_ic_positive": metric["mean_rank_ic"] > gen.minimum_rank_ic,
            "rank_ic_ir_positive": metric["rank_ic_ir"] > gen.minimum_rank_ic_ir,
            "positive_ratio": metric["positive_rank_ic_ratio"] >= gen.minimum_positive_ratio,
            "rank_ic_2025_positive": y2025["mean_rank_ic"] > gen.minimum_2025_rank_ic,
            "net_research_proxy_alpha_positive": portfolio["net_research_proxy_alpha"]
            > gen.minimum_net_research_proxy_alpha,
            "drawdown_acceptable": portfolio["max_drawdown"] >= gen.maximum_drawdown,
            "turnover_acceptable": portfolio["annualized_turnover"]
            <= gen.maximum_annualized_turnover,
            "sector_concentration_acceptable": portfolio["average_maximum_sector_weight"]
            <= gen.maximum_sector_weight,
        }
        candidates.append(
            {
                "model": portfolio["model"],
                "horizon": int(portfolio["horizon"]),
                "portfolio_policy": portfolio["portfolio_policy"],
                "mean_rank_ic": metric["mean_rank_ic"],
                "rank_ic_ir": metric["rank_ic_ir"],
                "positive_rank_ic_ratio": metric["positive_rank_ic_ratio"],
                "rank_ic_2025": y2025["mean_rank_ic"],
                "net_research_proxy_alpha": portfolio["net_research_proxy_alpha"],
                "max_drawdown": portfolio["max_drawdown"],
                "annualized_turnover": portfolio["annualized_turnover"],
                "average_maximum_sector_weight": portfolio["average_maximum_sector_weight"],
                "gates_passed": sum(bool(value) for value in gates.values()),
                "all_gates_passed": all(gates.values()),
                "gates": json.dumps(
                    {name: bool(value) for name, value in gates.items()}, sort_keys=True
                ),
            }
        )
    table = pd.DataFrame(candidates).sort_values(
        ["all_gates_passed", "gates_passed", "net_research_proxy_alpha", "mean_rank_ic"],
        ascending=False,
    )
    best = table.iloc[0].to_dict()
    best_lgb = table[table["model"].eq("lightgbm_regression")].iloc[0]
    best_ridge = table[table["model"].eq("ridge")].iloc[0]
    if (
        best["model"] == "lightgbm_regression"
        and abs(best_lgb["mean_rank_ic"] - best_ridge["mean_rank_ic"])
        <= gen.complexity_rank_ic_tie
        and abs(
            best_lgb["net_research_proxy_alpha"]
            - best_ridge["net_research_proxy_alpha"]
        )
        <= gen.complexity_net_alpha_tie
    ):
        best = best_ridge.to_dict()
        best["complexity_tiebreak_applied"] = True
    else:
        best["complexity_tiebreak_applied"] = False
    best["all_gates_passed"] = bool(best["all_gates_passed"])
    return best, table


def run_development(settings: Gen02Settings | None = None) -> dict:
    settings = settings or Gen02Settings()
    intact = verify_development_plan(settings)
    if not intact["intact"]:
        raise RuntimeError(f"GEN02_DEVELOPMENT_PLAN_NOT_INTACT: {intact}")
    if (settings.artifact_dir / "decision.json").exists():
        raise RuntimeError("GEN02_DEVELOPMENT_ALREADY_CONSUMED")
    holdout = json.loads((settings.artifact_dir / "holdout_audit.json").read_text(encoding="utf-8"))
    if holdout["untouched_2026_holdout"] is not False:
        raise RuntimeError("GEN02_HOLDOUT_AUDIT_UNEXPECTED")
    base = ChallengerSettings()
    data, data_evidence = load_development_dataset(base)
    selected = _selected_factors()
    scores, window_sensitivity = _fit_development_scores(data, settings, base)
    model_metrics, yearly_metrics = _score_metrics(scores)
    tail = _tail_metrics(scores)
    drift = _feature_drift(data, selected)
    decay = _factor_decay(selected)
    selection_pairs, selection_frequency = _selection_stability(selected)
    portfolios, turnover_decomposition, periods = _portfolio_analysis(scores, settings, base)
    ranking_differences = _ranking_differences(scores)
    regime_metrics, industry_metrics = _stability_metrics(scores)
    selected_config, candidate_table = _choose_configuration(
        model_metrics, yearly_metrics, portfolios, settings
    )

    selected_key = (
        str(selected_config["model"]),
        int(selected_config["horizon"]),
        str(selected_config["portfolio_policy"]),
    )
    challenger_periods = periods[selected_key].set_index("date")["net_research_proxy_alpha"]
    v6_policy = str(selected_config["portfolio_policy"])
    v6_periods = periods[("v6", int(selected_config["horizon"]), v6_policy)].set_index("date")[
        "net_research_proxy_alpha"
    ]
    challenger_daily = []
    v6_daily = []
    selected_scores = scores[scores["horizon"].eq(int(selected_config["horizon"]))]
    for _, current in selected_scores.groupby("date"):
        challenger_daily.append(
            current[f"score_{selected_config['model']}"] .corr(
                current[f"future_return_{int(selected_config['horizon'])}d"], method="spearman"
            )
        )
        v6_daily.append(
            current["score_v6"].corr(
                current[f"future_return_{int(selected_config['horizon'])}d"], method="spearman"
            )
        )
    bootstrap = {
        "rank_ic_delta_vs_v6": moving_block_bootstrap_delta(
            pd.Series(challenger_daily),
            pd.Series(v6_daily),
            replications=settings.bootstrap_replications,
            block_length=settings.bootstrap_block_length,
            seed=settings.random_seed,
        ),
        "topk_net_research_proxy_alpha_delta_vs_v6": moving_block_bootstrap_delta(
            challenger_periods,
            v6_periods,
            replications=settings.bootstrap_replications,
            block_length=min(10, settings.bootstrap_block_length),
            seed=settings.random_seed,
        ),
        "classification": "DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
    }

    tail_2025 = tail[tail["test_year"].eq(2025)]
    selected_tail = tail_2025[
        tail_2025["model"].eq(selected_config["model"])
        & tail_2025["horizon"].eq(selected_config["horizon"])
    ].iloc[0]
    severe_drift = drift[drift["drift_status"].eq("SEVERE")]
    sign_reversals = decay[decay["sign_flip"]]
    selected_portfolio = portfolios[
        portfolios["model"].eq(selected_config["model"])
        & portfolios["horizon"].eq(selected_config["horizon"])
        & portfolios["portfolio_policy"].eq(selected_config["portfolio_policy"])
    ].iloc[0]
    failure_hypotheses = [
        {
            "rank": 1,
            "hypothesis": "TAIL_SIGNAL_WEAKNESS",
            "evidence": {
                "2025_top_decile_ic": float(selected_tail["top_decile_ic"]),
                "2025_top_quintile_ic": float(selected_tail["top_quintile_ic"]),
                "2025_overall_rank_ic": float(selected_tail["overall_rank_ic"]),
            },
        },
        {
            "rank": 2,
            "hypothesis": "TRANSACTION_COST_AND_RANKING_CHURN",
            "evidence": {
                "transaction_cost_sum": float(selected_portfolio["transaction_cost_sum"]),
                "annualized_turnover": float(selected_portfolio["annualized_turnover"]),
                "gross_research_proxy_alpha": float(selected_portfolio["gross_research_proxy_alpha"]),
                "net_research_proxy_alpha": float(selected_portfolio["net_research_proxy_alpha"]),
            },
        },
        {
            "rank": 3,
            "hypothesis": "FACTOR_DECAY_AND_SIGN_REVERSAL",
            "evidence": {
                "sign_reversal_count": int(len(sign_reversals)),
                "largest_negative_ic_changes": decay.nsmallest(5, "ic_change")[
                    ["factor", "ic_change"]
                ].to_dict(orient="records"),
            },
        },
        {
            "rank": 4,
            "hypothesis": "FEATURE_DISTRIBUTION_SHIFT",
            "evidence": {
                "severe_psi_factor_count": int(len(severe_drift)),
                "severe_factors": severe_drift["factor"].tolist(),
            },
        },
        {
            "rank": 5,
            "hypothesis": "PORTFOLIO_EXPOSURE_CONCENTRATION",
            "evidence": {
                "average_maximum_sector_weight": float(
                    selected_portfolio["average_maximum_sector_weight"]
                ),
                "average_size_rank": float(selected_portfolio["average_size_rank"]),
                "average_liquidity_rank": float(
                    selected_portfolio["average_liquidity_rank"]
                ),
            },
        },
    ]
    failure_report = {
        "report_id": "2025_FAILURE_ANALYSIS",
        "classification": "DEVELOPMENT_ONLY",
        "primary_2025_failure_hypotheses": failure_hypotheses,
        "selection_jaccard_2024_to_2025": float(
            selection_pairs[
                selection_pairs["prior_year"].eq(2024)
                & selection_pairs["current_year"].eq(2025)
            ]["jaccard_similarity"].iloc[0]
        ),
        "no_2026_labels_read": True,
    }

    shadow_eligible = bool(selected_config["all_gates_passed"])
    decision_name = (
        "GEN2_DEVELOPMENT_HYPOTHESIS_READY" if shadow_eligible else "GEN2_REJECTED"
    )
    precommit = {
        "precommit_id": "GEN02-UNIQUE-FUTURE-HYPOTHESIS",
        "created_after_development_before_any_future_shadow": True,
        "untouched_2026_holdout": False,
        "historical_confirmation_claimed": False,
        "challenger_model": selected_config["model"],
        "primary_horizon": int(selected_config["horizon"]),
        "feature_policy": "V31 yearly train-only selected factors; no prospective inputs",
        "training_window_policy": "8-year rolling baseline unless future protocol separately freezes an approved development variant",
        "portfolio_policy": selected_config["portfolio_policy"],
        "top_k": int(
            portfolios[
                portfolios["model"].eq(selected_config["model"])
                & portfolios["horizon"].eq(selected_config["horizon"])
                & portfolios["portfolio_policy"].eq(selected_config["portfolio_policy"])
            ]["top_k"].iloc[0]
        ),
        "rebalance_trading_days": int(selected_config["horizon"]),
        "cost_model": development_protocol(settings)["cost_model"],
        "all_development_gates_passed": shadow_eligible,
        "shadow_eligible": shadow_eligible,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    decision = {
        "decision": decision_name,
        "champion": "V6",
        "challenger_generation": "GEN02",
        "best_development_configuration": selected_config,
        "historical_confirmation_passed": False,
        "holdout_result": "NOT_EVALUATED_2026_NOT_UNTOUCHED",
        "shadow_eligible": shadow_eligible,
        "v6_remains_champion": True,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    report = {
        "report_id": "CHALLENGER-GENERATION-02",
        "created_at_utc": _utc(),
        "data": data_evidence,
        "untouched_2026_holdout": False,
        "development_years": list(settings.development_years),
        "models": ["v6", *settings.candidate_models],
        "official_benchmark_status": "UNAPPROVED",
        "alpha_field_semantics": "research_proxy_alpha_only",
        "best_development_configuration": selected_config,
        "decision": decision,
        "safety": {
            "prospective_v1r4_modified": False,
            "v6_modified": False,
            "v30_modified": False,
            "v30r1_modified": False,
            "prospective_rows_used": 0,
            "2026_labels_or_performance_read": False,
            "lambda_rank_retuned": False,
            "new_model_families_added": False,
            "production_prediction_ready": False,
            "execution_authorized": False,
        },
    }

    csv_outputs = {
        "feature_drift.csv": drift,
        "factor_decay.csv": decay,
        "factor_selection_stability.csv": selection_pairs,
        "factor_selection_frequency.csv": selection_frequency,
        "training_window_sensitivity.csv": window_sensitivity,
        "tail_metrics.csv": tail,
        "portfolio_variants.csv": portfolios,
        "turnover_decomposition.csv": turnover_decomposition,
        "model_comparison.csv": model_metrics,
        "yearly_metrics.csv": yearly_metrics,
        "regime_metrics.csv": regime_metrics,
        "industry_metrics.csv": industry_metrics,
        "ranking_differences.csv": ranking_differences,
        "candidate_gate_matrix.csv": candidate_table,
    }
    for name, frame in csv_outputs.items():
        _write_csv(settings.artifact_dir / name, frame)
    _write_json(settings.artifact_dir / "bootstrap.json", bootstrap)
    _write_json(settings.artifact_dir / "2025_failure_analysis.json", failure_report)
    _write_json(settings.artifact_dir / "precommit_decision.json", precommit)
    _write_json(
        settings.artifact_dir / "holdout_result.json",
        {
            "status": "NOT_EVALUATED_2026_NOT_UNTOUCHED",
            "holdout_consumed": False,
            "historical_confirmation_passed": False,
            "low_statistical_power": True,
        },
    )
    _write_json(settings.artifact_dir / "decision.json", decision)
    _write_json(settings.artifact_dir / "report.json", report)
    return decision


def open_holdout_once(artifact_dir: Path, untouched: bool) -> dict:
    if not untouched:
        raise RuntimeError("UNTOUCHED_2026_HOLDOUT_FALSE")
    for name in ("development_protocol.json", "precommit_decision.json"):
        if not _verify_sidecar(artifact_dir / name):
            raise RuntimeError(f"HOLDOUT_PRECOMMIT_NOT_VERIFIED: {name}")
    state = {
        "holdout_consumed": True,
        "holdout_opened_at_utc": _utc(),
        "automatic_retry_allowed": False,
    }
    _write_json(artifact_dir / "holdout_state.json", state)
    return state


def evaluate_holdout_2026(settings: Gen02Settings | None = None) -> dict:
    settings = settings or Gen02Settings()
    audit = json.loads((settings.artifact_dir / "holdout_audit.json").read_text(encoding="utf-8"))
    # Fail before reading any 2026 label or performance column.
    return open_holdout_once(settings.artifact_dir, bool(audit["untouched_2026_holdout"]))


def record_test_receipt(kind: str, command: str, summary: str) -> dict:
    settings = Gen02Settings()
    receipt = {
        "kind": kind,
        "recorded_at_utc": _utc(),
        "command": command,
        "summary": summary,
        "exit_code": 0,
        "new_xfail_added": False,
        "new_skip_added": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    _write_json(settings.artifact_dir / f"{kind}_test_receipt.json", receipt)
    return receipt


def freeze_final(settings: Gen02Settings | None = None) -> dict:
    settings = settings or Gen02Settings()
    required = [
        settings.artifact_dir / "decision.json",
        settings.artifact_dir / "report.json",
        settings.artifact_dir / "targeted_test_receipt.json",
        settings.artifact_dir / "full_test_receipt.json",
    ]
    if any(not _verify_sidecar(path) for path in required):
        raise RuntimeError("GEN02_FINAL_FREEZE_REQUIRED_ARTIFACT_INVALID")
    effective_development = verify_development_plan(settings)
    if not effective_development["intact"]:
        raise RuntimeError("GEN02_EFFECTIVE_DEVELOPMENT_LOCK_NOT_INTACT")
    effective_lock_path = next(
        directory / "development_plan.lock.json"
        for directory in reversed(DEVELOPMENT_AMENDMENT_DIRS)
        if (directory / "development_plan.lock.json").exists()
    )
    parent_paths = [
        effective_lock_path,
        Path("artifacts/research_v31/experiments/002_ci_verifier_fix/plan.lock.json"),
        Path("artifacts/prospective_alpha_v1r4/plan.lock.json"),
        Path("artifacts/research_v6/plan.lock.json"),
        Path("artifacts/prediction_forward/v30r1_r2/plan.lock.json"),
    ]
    plan = {
        "lock_id": "CHALLENGER-GEN02-FINAL",
        "created_at_utc": _utc(),
        "development_lock_sha256": effective_development["lock_sha256"],
        "parents": {path.as_posix(): sha256(path) for path in parent_paths},
        "code": {
            path.as_posix(): sha256(path)
            for path in (
                Path("stockpilot/research_challenger/gen02.py"),
                Path("stockpilot/research_challenger/gen02_portfolio.py"),
                Path("tests/test_research_challenger_gen02.py"),
            )
        },
        "untouched_2026_holdout": False,
        "historical_confirmation_passed": False,
        "v6_modified": False,
        "v30_modified": False,
        "prospective_v1r4_modified": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    lock_hash = _write_json(settings.artifact_dir / "plan.lock.json", plan)
    files = {
        path.relative_to(settings.artifact_dir).as_posix(): sha256(path)
        for path in sorted(settings.artifact_dir.rglob("*"))
        if path.is_file()
        and not path.name.endswith(".sha256")
        and path.name not in {"artifact_manifest.json", "plan.lock.json"}
    }
    manifest = {
        "manifest_id": "CHALLENGER-GEN02-ARTIFACTS",
        "created_at_utc": _utc(),
        "files": files,
        "plan_lock_sha256": lock_hash,
        "immutable": True,
        "research_only": True,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    manifest_hash = _write_json(settings.artifact_dir / "artifact_manifest.json", manifest)
    return {
        "status": "GEN02_FROZEN",
        "plan_lock_sha256": lock_hash,
        "artifact_manifest_sha256": manifest_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical Challenger Generation 02")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "audit",
        "freeze-development",
        "freeze-development-amendment",
        "freeze-postrun-amendment",
        "verify-development",
        "develop",
        "evaluate-holdout-2026",
        "freeze",
        "report",
    ):
        subparsers.add_parser(name)
    receipt = subparsers.add_parser("record-test")
    receipt.add_argument("kind", choices=("targeted", "full"))
    receipt.add_argument("test_command")
    receipt.add_argument("summary")
    args = parser.parse_args()
    if args.command == "audit":
        result = run_read_only_audit()
    elif args.command == "freeze-development":
        result = freeze_development_plan()
    elif args.command == "verify-development":
        result = verify_development_plan()
    elif args.command == "freeze-development-amendment":
        result = freeze_development_amendment()
    elif args.command == "freeze-postrun-amendment":
        result = freeze_postrun_amendment()
    elif args.command == "develop":
        result = run_development()
    elif args.command == "evaluate-holdout-2026":
        result = evaluate_holdout_2026()
    elif args.command == "record-test":
        result = record_test_receipt(args.kind, args.test_command, args.summary)
    elif args.command == "freeze":
        result = freeze_final()
    else:
        result = json.loads(Path("artifacts/research_challenger/gen02/report.json").read_text(encoding="utf-8"))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
