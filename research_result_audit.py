"""Independent post-run checks; never edit a frozen experiment or approve trading."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


MODES = ("v16_control", "v20_adaptive", "v20_timing")
OUTPUTS = ("equity.csv", "holdings.csv", "daily_nav.csv", "settlements.json")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest().upper()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, message):
    require(np.allclose(actual, expected, rtol=1e-8, atol=1e-10, equal_nan=False), message)


def frame_metrics(frame, daily):
    """Recalculate from saved observations, not the experiment summary function."""
    annual = []
    for year, part in frame.groupby("test_year", sort=True):
        strategy = float(np.prod(1 + part.period_return) - 1)
        benchmark = float(np.prod(1 + part.benchmark_return) - 1)
        annual.append({"year": int(year), "return": strategy, "benchmark_return": benchmark,
                       "excess_return": strategy - benchmark,
                       "rank_ic_5": nullable_mean(part.rank_ic_5),
                       "technology_rank_ic_5": nullable_mean(part.technology_rank_ic_5)})
    nav = np.r_[1.0, daily.nav.to_numpy()]
    peak = np.maximum.accumulate(nav)
    total = float(np.prod(1 + frame.period_return) - 1)
    benchmark = float(np.prod(1 + frame.benchmark_return) - 1)
    held = frame[frame.in_market.eq(True)]
    metrics = {
        "total_return": total, "benchmark_return": benchmark, "excess_return": total - benchmark,
        "max_drawdown": float(np.min(nav / peak - 1)),
        "positive_excess_years": sum(row["excess_return"] > 0 for row in annual),
        "test_years": len(annual), "periods": len(frame), "periods_held": len(held),
        "held_period_win_rate": float(held.period_return.gt(0).mean()) if len(held) else None,
        "rank_ic_5": nullable_mean(frame.rank_ic_5), "rank_ic_20": nullable_mean(frame.rank_ic_20),
        "technology_rank_ic_5": nullable_mean(frame.technology_rank_ic_5),
        "average_one_way_turnover": float((frame.buy_turnover + frame.sell_turnover).mean() / 2),
        "average_transaction_cost": float(frame.transaction_cost.mean()),
    }
    return metrics, annual


def nullable_mean(series):
    value = float(series.mean())
    return value if np.isfinite(value) else None


def gate_screen(metrics):
    """Missing comparable or model-specific evidence must never become a pass."""
    def gate(value, passed, basis):
        return {"status": "unverified" if value is None else ("pass" if passed else "fail"),
                "value": value, "basis": basis}
    tech = metrics["technology_rank_ic_5"]
    return {
        "positive_excess": gate(metrics["excess_return"], metrics["excess_return"] > 0, "PIT proxy only, not official CSI300"),
        "four_of_six_years": gate(metrics["positive_excess_years"], metrics["test_years"] == 6 and metrics["positive_excess_years"] >= 4, "decision-year cohorts; PIT proxy only"),
        "rank_ic_at_least_v6": gate(None, False, "No same-input, same-label V6 comparison"),
        "technology_ic_nonnegative": gate(tech, tech is not None and tech >= 0, "Legacy five-security-bar labels"),
        "max_drawdown_at_least_minus_18pct": gate(metrics["max_drawdown"], metrics["max_drawdown"] >= -0.18, "Opening and post-trade NAV, not intraday worst case"),
        "turnover_and_cost_at_most_v8": gate(None, False, "Archived V8 uses different dates, intervals and accounting"),
        "candidate_future_shadow_126_days": gate(None, False, "Existing lowvol_top20 shadow is not a V20r2 shadow"),
    }


def check_frames(equity, holdings, daily, settings, periods):
    years = settings["test_years"]
    require(set(equity["mode"]) == set(MODES), "missing/unexpected experiment mode")
    require(set(daily["mode"]) == set(MODES), "missing/unexpected daily mode")
    require(set(holdings["mode"]) <= set(MODES), "unexpected holdings mode")
    require(not equity.duplicated(["mode", "date"]).any(), "duplicate equity keys")
    require(not holdings.duplicated(["mode", "date", "symbol"]).any(), "duplicate holdings keys")
    require(not daily.duplicated(["mode", "date", "point"]).any(), "duplicate daily keys")
    require(set(daily.point) <= {"before_rebalance", "after_rebalance"}, "unknown NAV point")
    for frame, columns in ((equity, ["period_return", "benchmark_return", "nav", "buy_turnover", "sell_turnover", "transaction_cost", "cash_weight"]),
                           (holdings, ["units", "weight", "target_weight"]), (daily, ["nav"])):
        require(np.isfinite(frame[columns]).all().all(), "non-finite ledger values")
    require(equity.nav.gt(0).all() and daily.nav.gt(0).all(), "non-positive NAV")
    require((equity[["buy_turnover", "sell_turnover", "transaction_cost", "cash_weight"]] >= -1e-10).all().all(), "negative turnover/cost/cash")
    require(equity.cash_weight.le(1 + 1e-8).all(), "cash exceeds NAV")
    require((holdings[["units", "weight", "target_weight"]] >= -1e-10).all().all(), "negative holding")
    require(holdings.groupby(["mode", "date"]).weight.sum().le(1 + 1e-8).all(), "leveraged holdings")
    rate = settings["fee_rate"] + settings["slippage"]
    same(equity.transaction_cost, equity.buy_turnover * rate + equity.sell_turnover * (rate + settings["stamp_duty"]), "cost/turnover mismatch")
    same(equity.excess_period_return, equity.period_return - equity.benchmark_return, "period excess mismatch")
    reference = equity[equity["mode"].eq(MODES[0])].reset_index(drop=True)
    result = {}
    for mode in MODES:
        frame = equity[equity["mode"].eq(mode)].reset_index(drop=True)
        points = daily[daily["mode"].eq(mode)].reset_index(drop=True)
        require(len(frame) == periods and sorted(frame.test_year.unique()) == years, "incomplete evaluation years/periods")
        require(frame.date.is_monotonic_increasing and points.date.is_monotonic_increasing, "unordered outputs")
        require(frame.date.equals(reference.date), "unmatched evaluation dates")
        same(frame.benchmark_return, reference.benchmark_return, "unmatched benchmark")
        require((frame.date < frame.entry_date).all() and (frame.entry_date < frame.end_date).all(), "invalid execution chronology")
        require(frame.test_year.eq(frame.date.dt.year).all(), "invalid test year")
        require(frame.end_date.iloc[:-1].reset_index(drop=True).equals(frame.entry_date.iloc[1:].reset_index(drop=True)), "non-contiguous periods")
        same((1 + frame.period_return).cumprod(), frame.nav, "NAV compounding mismatch")
        before = np.r_[1.0, frame.nav.to_numpy()[:-1]]
        after_trade = points[points.point.eq("after_rebalance")].set_index("date")
        require(set(after_trade.index) == set(frame.entry_date), "missing/extra rebalance NAV")
        same(after_trade.loc[frame.entry_date, "nav"], before * (1 - frame.transaction_cost), "post-trade NAV/cost mismatch")
        opening = points[points.point.eq("before_rebalance")].set_index("date")
        require(len(opening) == periods * settings["rebalance_every"], "missing daily observations")
        same(opening.loc[frame.end_date, "nav"], frame.nav, "daily/period end NAV mismatch")
        metrics, annual = frame_metrics(frame, points)
        result[mode] = {"metrics": metrics, "annual": annual, "gate_screen": gate_screen(metrics),
                        "blocked_orders": int(frame.blocked_orders.sum()),
                        "stale_position_observations": int(frame.stale_position_observations.sum())}
    return result


def audit_v20r2(root="."):
    # The original verifier also walks the entire parent chain and settings.
    from research_v20r2.freeze import verify
    lock = verify()
    root = Path(root)
    folder = root / "artifacts/research_v20r2"
    read = lambda name: json.loads((folder / name).read_text(encoding="utf-8"))
    report, started, runtime = read("report.json"), read("run.started.json"), read("runtime_status.json")
    require(runtime.get("stage") == "complete", "run not complete")
    require(report["lock_sha256"] == started["lock_sha256"] == lock["lock_sha256"], "run/report lock mismatch")
    require(report.get("execution_authorized") is False and report.get("replacement_approved") is False, "unexpected approval")
    require(set(report["output_sha256"]) == set(OUTPUTS), "output manifest incomplete")
    for name in OUTPUTS:
        require(digest(folder / name) == report["output_sha256"][name], f"output hash mismatch: {name}")
    frames = {}
    for name, dates in (("equity", ["date", "entry_date", "end_date"]), ("holdings", ["date", "execution_date"]), ("daily_nav", ["date"])):
        frames[name] = pd.read_csv(folder / f"{name}.csv", dtype={"symbol": str}, parse_dates=dates)
    result = check_frames(frames["equity"], frames["holdings"], frames["daily_nav"], lock["settings"], read("data_audit.json")["evaluation_dates"])
    for mode, values in result.items():
        for key, value in values["metrics"].items():
            expected = report["metrics"][mode][key]
            if value is None:
                require(expected is None, f"missing metric mismatch: {mode}/{key}")
            else:
                same(value, expected, f"metric mismatch: {mode}/{key}")
    for year in lock["settings"]["test_years"]:
        checkpoint = read(f"checkpoints/through_{year}.json")
        require(checkpoint["completed_year"] == year and checkpoint["partial_result_only"] is True, "invalid checkpoint")
        expected_names = {f"through_{year}_{name}.csv" for name in frames}
        require(set(checkpoint["output_sha256"]) == expected_names, "checkpoint manifest incomplete")
        for name, expected in checkpoint["output_sha256"].items():
            require(digest(folder / "checkpoints" / name) == expected, "checkpoint hash mismatch")
            if year == max(lock["settings"]["test_years"]):
                original = name.removeprefix(f"through_{year}_")
                require(expected == report["output_sha256"][original], "final checkpoint differs from report outputs")
    events = read("settlements.json")["events"]
    actions = json.loads((root / lock["settings"]["action_path"]).read_text(encoding="utf-8"))
    # Event metadata is external to the output; do not infer ratios from PnL.
    registered = {e["old_symbol"]: e for e in actions["events"]}
    keys = set()
    for event in events:
        key = (event["mode"], event["old_symbol"])
        require(key not in keys, "duplicate settlement")
        keys.add(key)
        action = registered[event["old_symbol"]]
        require(event["mode"] in (*MODES, "benchmark") and event["date"] == action["listing_date"] and event["new_symbol"] == action["new_symbol"], "settlement metadata mismatch")
        same(event["raw_share_ratio"], action["ratio"], "swap ratio mismatch")
        require(event["fees"] == 0 and event["old_units"] > 0 and event["new_units"] > 0, "invalid settlement values")
    return {"schema": "stockpilot-post-run-audit-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_report_sha256": digest(folder / "report.json"), "lock_sha256": lock["lock_sha256"],
            "frozen_inputs_intact": True, "output_consistency_passed": True, "settlements_checked": len(events),
            "metrics_and_gates": result, "replacement_approved": False, "execution_authorized": False,
            "decision": "keep_v6_and_diagnose_score_components",
            "limitations": ["Post-hoc output consistency audit, not an independent trade replay or performance preregistration.",
                            "Positive proxy excess is not proof of outperformance against official CSI300.",
                            "All three modes fail technology IC and the -18% drawdown screen.",
                            "V6/V8 comparability and candidate-specific future shadow evidence are absent."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    require(Path.cwd().resolve() == project, "run from project root")
    allowed = (project / "artifacts/autopilot").resolve()
    target = args.output.resolve()
    require(target.is_relative_to(allowed), "audit output must stay in artifacts/autopilot")
    require(not target.exists(), "preserve existing audit")
    result = audit_v20r2()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
