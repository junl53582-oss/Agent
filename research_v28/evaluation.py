from __future__ import annotations

import numpy as np
import pandas as pd

from research_v20.validation import summarize


def _safe_ic(frame, score, target):
    sample = frame[[score, target]].dropna()
    if len(sample) < 10 or sample[score].nunique() < 2:
        return np.nan
    return float(sample[score].corr(sample[target], method="spearman"))


def ranking_gate(scores: pd.DataFrame, score="v28_score") -> dict:
    rows = []
    for date, group in scores[scores["eligible"].fillna(False)].groupby("date"):
        tech = group[group["broad_sector"].eq("technology")]
        rows.append({"date": pd.Timestamp(date), "ic5": _safe_ic(group, score, "label_5"),
                     "ic20": _safe_ic(group, score, "v10_target_20"),
                     "technology_ic5": _safe_ic(tech, score, "label_5")})
    daily = pd.DataFrame(rows)
    daily["year"] = daily.date.dt.year
    annual = daily.groupby("year")[["ic5", "ic20", "technology_ic5"]].mean()
    metrics = {"mean_ic5": float(daily.ic5.mean()), "mean_ic20": float(daily.ic20.mean()),
               "technology_ic5": float(daily.technology_ic5.mean()),
               "positive_ic5_years": int(annual.ic5.gt(0).sum()), "years": len(annual),
               "annual": {str(year): row.to_dict() for year, row in annual.iterrows()}}
    checks = {"ic5_positive": metrics["mean_ic5"] > 0, "ic20_positive": metrics["mean_ic20"] > 0,
              "positive_ic5_years_at_least_4": metrics["positive_ic5_years"] >= 4,
              "technology_ic5_nonnegative": metrics["technology_ic5"] >= 0}
    return {"metrics": metrics, "checks": checks, "passed": all(checks.values())}


def selection_gate(scores: pd.DataFrame, score="v28_score", quantile=0.8) -> dict:
    rows = []
    for date, group in scores[scores["eligible"].fillna(False)].groupby("date"):
        result = {"date": pd.Timestamp(date), "spread5": [], "spread20": [], "precision_lift": []}
        for _, sector in group.groupby("broad_sector"):
            if len(sector) < 10:
                continue
            predicted = sector[sector[score].rank(pct=True, method="first") > quantile]
            for key, target in (("spread5", "label_5"), ("spread20", "v10_target_20")):
                valid = sector.dropna(subset=[target])
                chosen = predicted[predicted[target].notna()]
                rest = valid[~valid.index.isin(chosen.index)]
                if len(chosen) and len(rest):
                    result[key].append(float(chosen[target].mean() - rest[target].mean()))
            valid = sector.dropna(subset=["label_5"])
            chosen = predicted[predicted.label_5.notna()]
            if len(chosen) and len(valid):
                truth = set(valid[valid.label_5.rank(pct=True, method="first") > quantile].index)
                result["precision_lift"].append(len(set(chosen.index) & truth) / len(chosen) - (1 - quantile))
        rows.append({"date": result["date"], **{key: float(np.mean(value)) if value else np.nan for key, value in result.items() if key != "date"}})
    daily = pd.DataFrame(rows)
    daily["year"] = daily.date.dt.year
    annual = daily.groupby("year")[["spread5", "spread20", "precision_lift"]].mean()
    metrics = {"mean_spread5": float(daily.spread5.mean()), "mean_spread20": float(daily.spread20.mean()),
               "precision_lift": float(daily.precision_lift.mean()),
               "positive_spread5_years": int(annual.spread5.gt(0).sum()), "years": len(annual),
               "annual": {str(year): row.to_dict() for year, row in annual.iterrows()}}
    checks = {"spread5_positive": metrics["mean_spread5"] > 0, "spread20_positive": metrics["mean_spread20"] > 0,
              "positive_spread5_years_at_least_4": metrics["positive_spread5_years"] >= 4,
              "precision_lift_positive": metrics["precision_lift"] > 0}
    return {"metrics": metrics, "checks": checks, "passed": all(checks.values())}


def active_drawdown(frame: pd.DataFrame) -> float:
    strategy = (1 + frame["period_return"]).cumprod()
    benchmark = (1 + frame["benchmark_return"]).cumprod()
    active = strategy / benchmark
    return float((active / active.cummax() - 1).min())


def conversion_gate(candidate: pd.DataFrame, control: pd.DataFrame, active_floor=-0.10, tracking_limit=0.06) -> dict:
    candidate_metrics, control_metrics = summarize(candidate), summarize(control)
    excess = candidate["period_return"] - candidate["benchmark_return"]
    tracking_error = float(excess.std(ddof=1) * np.sqrt(252 / 20))
    active_dd = active_drawdown(candidate)
    metrics = {**candidate_metrics, "active_max_drawdown": active_dd, "active_drawdown_sampling": "20-session settlement wealth ratio",
               "tracking_error": tracking_error, "control_turnover": control_metrics["average_one_way_turnover"],
               "control_cost": control_metrics["average_transaction_cost"]}
    checks = {"cumulative_excess_positive": metrics["excess_return"] > 0,
              "positive_excess_years_at_least_4": metrics["positive_excess_years"] >= 4,
              "active_drawdown_not_below_minus_10pct": active_dd >= active_floor,
              "tracking_error_lte_6pct": tracking_error <= tracking_limit,
              "turnover_not_above_control": metrics["average_one_way_turnover"] <= control_metrics["average_one_way_turnover"],
              "cost_not_above_control": metrics["average_transaction_cost"] <= control_metrics["average_transaction_cost"],
              "untouched_126_day_future_test_complete": False}
    return {"metrics": metrics, "checks": checks, "passed": all(checks.values())}


def evaluate_three_gates(scores, equity, settings):
    ranking = ranking_gate(scores)
    selection = selection_gate(scores, quantile=settings.tail_quantile)
    candidate = equity[equity["mode"].eq("v28_confidence_tail")].copy()
    control = equity[equity["mode"].eq("v16_replay")].copy()
    conversion = conversion_gate(candidate, control, settings.active_drawdown_floor, settings.maximum_tracking_error)
    return {"ranking": ranking, "selection": selection, "portfolio_conversion": conversion,
            "all_three_passed": ranking["passed"] and selection["passed"] and conversion["passed"]}

