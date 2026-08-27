from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .backtest import BacktestResult, run_walk_forward
from .config import Settings
from .data import load_panel
from .exposure import attach_exposures, exposure_coverage, load_exposures
from .membership import attach_point_in_time_membership, load_membership_history


@dataclass(frozen=True)
class Candidate:
    name: str
    model_name: str
    top_n: int
    weighting: str = "equal"
    hold_buffer: int = 5
    industry_cap: float = 0.3


DEFAULT_CANDIDATES = [
    Candidate("ridge_top20", "ridge", 20),
    Candidate("lgb_top10", "lightgbm", 10),
    Candidate("lgb_top20", "lightgbm", 20),
    Candidate("lgb_top30", "lightgbm", 30),
    Candidate("lgb_top20_invvol", "lightgbm", 20, "inverse_volatility"),
    Candidate("momentum60_top20", "momentum_60", 20),
    Candidate("lowvol_top20", "low_volatility", 20, "inverse_volatility"),
]


def _candidate_settings(
    base: Settings, candidate: Candidate, start: str, end: str | None
) -> Settings:
    return replace(
        base,
        model_name=candidate.model_name,
        top_n=candidate.top_n,
        weighting=candidate.weighting,
        hold_buffer=candidate.hold_buffer,
        industry_cap=candidate.industry_cap,
        evaluation_start=start,
        evaluation_end=end,
    )


def _summary_row(candidate: Candidate, result: BacktestResult) -> dict:
    metrics = result.metrics
    excess = metrics["total_return"] - metrics["benchmark_return"]
    validation_score = excess / max(abs(metrics["max_drawdown"]), 0.1) + metrics["mean_rank_ic"]
    return {
        **asdict(candidate),
        "total_return": metrics["total_return"],
        "benchmark_return": metrics["benchmark_return"],
        "excess_return": excess,
        "sharpe": metrics["sharpe"],
        "max_drawdown": metrics["max_drawdown"],
        "mean_rank_ic": metrics["mean_rank_ic"],
        "average_one_way_turnover": metrics["average_one_way_turnover"],
        "validation_score": validation_score,
        "passes_validation": bool(
            excess > 0 and metrics["mean_rank_ic"] > 0 and metrics["max_drawdown"] > -0.25
        ),
    }


def run_validation_v2(
    market_path: str | Path,
    membership_path: str | Path,
    validation_start: str,
    test_start: str,
    test_end: str | None = None,
    settings: Settings | None = None,
    candidates: list[Candidate] | None = None,
    force: bool = False,
    exposure_path: str | Path | None = None,
) -> dict:
    """Select on validation only, persist the choice, then evaluate the test period once."""
    settings = settings or Settings.from_env()
    candidates = candidates or DEFAULT_CANDIDATES
    target = settings.artifact_dir / "validation_v2"
    report_path = target / "report.json"
    if report_path.exists() and not force:
        raise FileExistsError("最终测试已打开；如确需重跑，请显式使用 --force")
    target.mkdir(parents=True, exist_ok=True)

    validation_end = str((pd.Timestamp(test_start) - pd.Timedelta(days=1)).date())
    plan = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "market_path": str(Path(market_path)),
        "membership_path": str(Path(membership_path)),
        "exposure_path": str(Path(exposure_path)) if exposure_path else None,
        "training_period": f"before {validation_start}",
        "validation_period": f"{validation_start} to {validation_end}",
        "test_period": f"{test_start} to {test_end or 'latest'}",
        "label_mode": settings.label_mode,
        "selection_rule": "excess/max(|drawdown|,10%) + RankIC; no test metrics used",
        "candidates": [asdict(candidate) for candidate in candidates],
        "holdout_status": "retrospective_research",
    }
    (target / "plan.lock.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    panel = load_panel(market_path)
    panel = attach_point_in_time_membership(panel, load_membership_history(membership_path))
    if exposure_path:
        panel = attach_exposures(panel, load_exposures(exposure_path))
    coverage = exposure_coverage(panel)
    rows: list[dict] = []
    validation_dir = target / "validation"
    for candidate in candidates:
        candidate_settings = _candidate_settings(
            settings, candidate, validation_start, validation_end
        )
        result = run_walk_forward(panel, candidate_settings)
        result.save(validation_dir / candidate.name)
        rows.append(_summary_row(candidate, result))
    comparison = pd.DataFrame(rows).sort_values("validation_score", ascending=False)
    comparison.to_csv(target / "validation_candidates.csv", index=False, encoding="utf-8-sig")

    chosen_row = comparison.iloc[0]
    chosen = next(candidate for candidate in candidates if candidate.name == chosen_row["name"])
    selection = {
        **asdict(chosen),
        "validation_score": float(chosen_row["validation_score"]),
        "passes_validation": bool(chosen_row["passes_validation"]),
    }
    (target / "selected_config.lock.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    test_settings = _candidate_settings(settings, chosen, test_start, test_end)
    test_result = run_walk_forward(panel, test_settings)
    test_result.save(target / "final_test")
    test_excess = test_result.metrics["total_return"] - test_result.metrics["benchmark_return"]
    positive_year_ratio = float((test_result.yearly["excess_return"] > 0).mean())
    test_pass = bool(
        test_excess > 0
        and test_result.metrics["mean_rank_ic"] > 0
        and positive_year_ratio >= 0.5
        and test_result.metrics["max_drawdown"] > -0.25
    )
    report = {
        "plan": plan,
        "selected": selection,
        "validation_candidates": len(comparison),
        "test_metrics": test_result.metrics,
        "test_excess_return": test_excess,
        "positive_excess_year_ratio": positive_year_ratio,
        "test_pass": test_pass,
        "exposure_coverage": coverage,
        "decision": "paper_trade" if test_pass else "research_only",
        "note": "本历史测试区间此前已被观察，因此只能视为回顾性留出；真正未触碰测试需依赖未来数据。",
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return report
