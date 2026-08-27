from __future__ import annotations

from pathlib import Path

import pandas as pd


def audit_inputs(
    membership_path: str | Path,
    market_path: str | Path,
    fundamental_path: str | Path,
    industry_path: str | Path,
) -> dict:
    membership = pd.read_csv(membership_path, dtype={"symbol": str})
    membership["snapshot_date"] = pd.to_datetime(membership["snapshot_date"])
    market = pd.read_csv(market_path, dtype={"symbol": str}, usecols=["date", "symbol"])
    market["date"] = pd.to_datetime(market["date"])
    fundamentals = pd.read_csv(
        fundamental_path,
        dtype={"symbol": str},
        usecols=["symbol", "report_date", "available_date"],
    )
    fundamentals["report_date"] = pd.to_datetime(fundamentals["report_date"])
    fundamentals["available_date"] = pd.to_datetime(fundamentals["available_date"])
    industry = pd.read_csv(
        industry_path, dtype={"symbol": str}, usecols=["symbol", "industry_effective_date"]
    )
    industry["industry_effective_date"] = pd.to_datetime(industry["industry_effective_date"])
    counts = membership.groupby("snapshot_date").size()
    report = {
        "membership_rows": len(membership),
        "membership_snapshots": int(membership["snapshot_date"].nunique()),
        "membership_symbols": int(membership["symbol"].nunique()),
        "membership_min": str(membership["snapshot_date"].min().date()),
        "membership_max": str(membership["snapshot_date"].max().date()),
        "membership_min_size": int(counts.min()),
        "membership_max_size": int(counts.max()),
        "market_rows": len(market),
        "market_symbols": int(market["symbol"].nunique()),
        "market_min": str(market["date"].min().date()),
        "market_max": str(market["date"].max().date()),
        "fundamental_rows": len(fundamentals),
        "fundamental_symbols": int(fundamentals["symbol"].nunique()),
        "fundamental_pit_violations": int(
            (fundamentals["available_date"] < fundamentals["report_date"]).sum()
        ),
        "industry_rows": len(industry),
        "industry_symbols": int(industry["symbol"].nunique()),
        "industry_future_violations": int(
            (industry["industry_effective_date"] > pd.Timestamp("2026-08-21")).sum()
        ),
    }
    gates = {
        "membership_exact_300": report["membership_min_size"]
        == report["membership_max_size"]
        == 300,
        "membership_starts_2015": report["membership_min"] <= "2015-01-31",
        "market_symbols_at_least_650": report["market_symbols"] >= 650,
        "market_starts_2015": report["market_min"] <= "2015-01-31",
        "market_ends_2026_08_21": report["market_max"] >= "2026-08-21",
        "fundamental_symbols_at_least_650": report["fundamental_symbols"] >= 650,
        "fundamental_pit_clean": report["fundamental_pit_violations"] == 0,
        "industry_symbols_at_least_630": report["industry_symbols"] >= 630,
        "industry_pit_clean": report["industry_future_violations"] == 0,
    }
    report["gates"] = gates
    report["passed"] = all(gates.values())
    if not report["passed"]:
        failed = ", ".join(name for name, passed in gates.items() if not passed)
        raise RuntimeError(f"V9输入数据质量门未通过: {failed}")
    return report

