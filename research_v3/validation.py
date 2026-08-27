from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stockpilot.data import load_panel
from stockpilot.exposure import attach_exposures, load_exposures
from stockpilot.membership import attach_point_in_time_membership, load_membership_history

from .backtest import run_v3_backtest
from .config import V3Settings
from .features import build_v3_dataset
from .fundamentals import attach_fundamentals_asof, load_fundamentals


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min()) if not equity.empty else 0.0


def _selection_score(group: pd.DataFrame) -> float:
    total = float((1 + group["period_return"]).prod() - 1)
    benchmark = float((1 + group["benchmark_return"]).prod() - 1)
    drawdown = _max_drawdown(group["period_return"])
    turnover = float((group["buy_turnover"] + group["sell_turnover"]).mean() / 2)
    return (
        (total - benchmark) / max(abs(drawdown), 0.10) + group["rank_ic"].mean() - 0.10 * turnover
    )


def nested_year_selection(equity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = equity.copy()
    data["date"] = pd.to_datetime(data["date"])
    years = sorted(data["date"].dt.year.unique())
    selected_rows = []
    folds = []
    for test_year in years[1:]:
        validation = data[(data["date"].dt.year == test_year - 1)]
        test = data[data["date"].dt.year == test_year]
        if validation.empty or test.empty:
            continue
        scores = pd.Series(
            {
                candidate: _selection_score(group)
                for candidate, group in validation.groupby("candidate")
            }
        )
        chosen = str(scores.idxmax())
        chosen_test = test[test["candidate"] == chosen].copy()
        strategy = float((1 + chosen_test["period_return"]).prod() - 1)
        benchmark = float((1 + chosen_test["benchmark_return"]).prod() - 1)
        folds.append(
            {
                "validation_year": test_year - 1,
                "test_year": test_year,
                "selected_candidate": chosen,
                "validation_score": float(scores.loc[chosen]),
                "test_return": strategy,
                "test_benchmark_return": benchmark,
                "test_excess_return": strategy - benchmark,
                "test_periods": len(chosen_test),
            }
        )
        selected_rows.append(chosen_test)
    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    return pd.DataFrame(folds), selected


def run_research_v3(
    market_path: str | Path = "data/market_history.csv",
    membership_path: str | Path = "data/universes/000300/history.csv",
    exposure_path: str | Path = "data/exposures.csv",
    fundamental_path: str | Path = "data/fundamentals_pit.csv",
    settings: V3Settings | None = None,
) -> dict:
    settings = settings or V3Settings(fundamental_path=Path(fundamental_path))
    settings.ensure_dirs()
    panel = load_panel(market_path)
    panel = attach_point_in_time_membership(panel, load_membership_history(membership_path))
    panel = attach_exposures(panel, load_exposures(exposure_path))
    fundamentals = load_fundamentals(fundamental_path)
    panel = attach_fundamentals_asof(panel, fundamentals)
    dataset = build_v3_dataset(panel, settings.horizons)
    equity, candidates, signals = run_v3_backtest(dataset, settings)
    folds, selected = nested_year_selection(equity)
    if selected.empty:
        raise RuntimeError("没有形成可评估的嵌套年度测试折")
    total = float((1 + selected["period_return"]).prod() - 1)
    benchmark = float((1 + selected["benchmark_return"]).prod() - 1)
    fold_positive = float((folds["test_excess_return"] > 0).mean())
    metrics = {
        "periods": len(selected),
        "total_return": total,
        "benchmark_return": benchmark,
        "excess_return": total - benchmark,
        "mean_rank_ic": float(selected["rank_ic"].mean()),
        "max_drawdown": _max_drawdown(selected["period_return"]),
        "positive_fold_ratio": fold_positive,
        "average_cash_weight": float(selected["cash_weight"].mean()),
    }
    gates = {
        "excess_return": metrics["excess_return"] > 0,
        "mean_rank_ic": metrics["mean_rank_ic"] > 0,
        "max_drawdown": metrics["max_drawdown"] > -0.20,
        "positive_fold_ratio": metrics["positive_fold_ratio"] >= 0.50,
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "retrospective_research",
        "settings": {
            **asdict(settings),
            "artifact_dir": str(settings.artifact_dir),
            "fundamental_path": str(settings.fundamental_path),
        },
        "fundamental_rows": len(fundamentals),
        "fundamental_symbols": int(fundamentals["symbol"].nunique()),
        "fundamental_pit_violations": int(
            (fundamentals["available_date"] < fundamentals["report_date"]).sum()
        ),
        "nested_folds": len(folds),
        "metrics": metrics,
        "gates": gates,
        "passed": all(gates.values()),
        "decision": "candidate_for_new_future_protocol"
        if all(gates.values())
        else "continue_research",
        "warning": "所有历史区间均已被观察；结果只能筛选V3研究方向，不能替代新的未来测试。",
    }
    target = settings.artifact_dir
    equity.to_csv(target / "candidate_equity.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(target / "candidate_metrics.csv", index=False, encoding="utf-8-sig")
    signals.to_csv(target / "signals.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(target / "nested_folds.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(target / "nested_selected_equity.csv", index=False, encoding="utf-8-sig")
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return report
