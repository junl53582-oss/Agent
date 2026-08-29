from __future__ import annotations

import numpy as np
import pandas as pd


TARGETS = ("label_5", "v10_target_20")


def tail_members(frame: pd.DataFrame, score="global_model_score", fraction=0.10):
    valid = frame[frame[score].notna()].sort_values([score, "symbol"], ascending=[False, True]).copy()
    count = max(1, int(np.floor(len(valid) * fraction))) if len(valid) else 0
    valid["tail"] = "middle"
    if count:
        valid.iloc[:count, valid.columns.get_loc("tail")] = "top"
        valid.iloc[-count:, valid.columns.get_loc("tail")] = "bottom"
    return valid


def equal_date_tail_spreads(scores: pd.DataFrame):
    rows = []
    eligible = scores[scores.eligible.eq(True)]
    for date, frame in eligible.groupby("date", sort=True):
        tails = tail_members(frame)
        for target in TARGETS:
            top = tails[tails["tail"].eq("top")][target].dropna()
            bottom = tails[tails["tail"].eq("bottom")][target].dropna()
            rows.append({"date": date, "test_year": pd.Timestamp(date).year, "target": target,
                         "top_mean": float(top.mean()) if len(top) else np.nan,
                         "bottom_mean": float(bottom.mean()) if len(bottom) else np.nan,
                         "top_minus_bottom": float(top.mean() - bottom.mean()) if len(top) and len(bottom) else np.nan,
                         "top_rows": len(top), "bottom_rows": len(bottom)})
    return pd.DataFrame(rows)


def selection_diagnostics(scores: pd.DataFrame, holdings: pd.DataFrame):
    selected = holdings[holdings["mode"].eq("global_only")][["date", "symbol", "target_weight"]].copy()
    selected["date"] = pd.to_datetime(selected.date)
    selected["symbol"] = selected.symbol.astype(str).str.zfill(6)
    current = scores.merge(selected, on=["date", "symbol"], how="left", validate="one_to_one")
    current["target_weight"] = current.target_weight.fillna(0.0)
    current["active_weight"] = current.target_weight - current.benchmark_weight
    current["overweight"] = current.active_weight.gt(1e-10)
    rows = []
    for (date, sector), frame in current[current.eligible.eq(True)].groupby(["date", "broad_sector"], sort=True):
        for target in TARGETS:
            over = frame[frame.overweight][target].dropna()
            other = frame[~frame.overweight][target].dropna()
            contribution = float((frame.active_weight * frame[target].fillna(0.0)).sum())
            rows.append({"date": date, "test_year": pd.Timestamp(date).year, "sector": sector, "target": target,
                         "overweight_mean": float(over.mean()) if len(over) else np.nan,
                         "other_mean": float(other.mean()) if len(other) else np.nan,
                         "overweight_minus_other": float(over.mean() - other.mean()) if len(over) and len(other) else np.nan,
                         "active_weighted_target": contribution, "overweight_rows": len(over), "other_rows": len(other)})
    return pd.DataFrame(rows), current[["date", "symbol", "active_weight", "overweight"]]


def summarize_equal_date(frame: pd.DataFrame, value: str, groups):
    return frame.groupby(list(groups), dropna=False)[value].agg(["mean", "count"]).reset_index().rename(columns={"mean": value, "count": "valid_periods"})


def cost_diagnostic(equity: pd.DataFrame):
    result = {}
    for mode in ("v16_replay", "global_only"):
        frame = equity[equity["mode"].eq(mode)]
        result[mode] = {"average_one_way_turnover": float((frame.buy_turnover + frame.sell_turnover).mean() / 2),
                        "average_transaction_cost": float(frame.transaction_cost.mean()),
                        "total_transaction_cost_fraction": float(frame.transaction_cost.sum())}
    result["global_minus_control"] = {key: result["global_only"][key] - result["v16_replay"][key] for key in result["global_only"]}
    return result
