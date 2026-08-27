from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from .backtest import BacktestResult, run_walk_forward
from .config import Settings
from .data import load_panel, make_demo_panel, save_panel
from .exposure import attach_exposures, load_exposures
from .membership import attach_point_in_time_membership, load_membership_history


def run_demo(settings: Settings | None = None) -> BacktestResult:
    settings = settings or Settings.from_env()
    settings.ensure_dirs()
    panel = make_demo_panel()
    save_panel(panel, settings.data_dir / "demo_market.csv")
    result = run_walk_forward(panel, settings)
    result.save(settings.artifact_dir)
    return result


def run_file(
    path: str | Path,
    settings: Settings | None = None,
    membership_path: str | Path | None = None,
    exposure_path: str | Path | None = None,
) -> BacktestResult:
    settings = settings or Settings.from_env()
    settings.ensure_dirs()
    panel = load_panel(path)
    if membership_path is not None:
        panel = attach_point_in_time_membership(panel, load_membership_history(membership_path))
    if exposure_path is not None:
        panel = attach_exposures(panel, load_exposures(exposure_path))
    result = run_walk_forward(panel, settings)
    result.save(settings.artifact_dir)
    return result


def run_comparison(
    path: str | Path,
    model_names: list[str],
    settings: Settings | None = None,
    membership_path: str | Path | None = None,
) -> pd.DataFrame:
    settings = settings or Settings.from_env()
    settings.ensure_dirs()
    panel = load_panel(path)
    if membership_path is not None:
        panel = attach_point_in_time_membership(panel, load_membership_history(membership_path))
    rows: list[dict] = []
    comparison_dir = settings.artifact_dir / "comparison"
    for model_name in model_names:
        model_settings = replace(settings, model_name=model_name)
        result = run_walk_forward(panel, model_settings)
        result.save(comparison_dir / model_name)
        rows.append(
            {
                "model": model_name,
                "total_return": result.metrics["total_return"],
                "annual_return": result.metrics["annual_return"],
                "benchmark_return": result.metrics["benchmark_return"],
                "excess_return": result.metrics["total_return"]
                - result.metrics["benchmark_return"],
                "sharpe": result.metrics["sharpe"],
                "max_drawdown": result.metrics["max_drawdown"],
                "mean_rank_ic": result.metrics["mean_rank_ic"],
                "win_rate": result.metrics["win_rate"],
                "execution_rate": result.metrics["signal_execution_rate"],
            }
        )
    comparison = pd.DataFrame(rows).sort_values("total_return", ascending=False)
    comparison.to_csv(settings.artifact_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    return comparison
