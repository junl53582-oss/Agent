from __future__ import annotations

from pathlib import Path

import pandas as pd


def audit_v10_inputs(
    membership_path: str | Path = "data/universes/000300/history_v10.csv",
    market_path: str | Path = "data/market_history_v10_hfq.csv",
    fundamental_path: str | Path = "data/fundamentals_pit_v10_extended.csv",
    industry_path: str | Path = "data/industry_history_v10.csv",
) -> dict:
    membership = pd.read_csv(membership_path, dtype={"symbol": str})
    membership["snapshot_date"] = pd.to_datetime(membership["snapshot_date"])
    market = pd.read_csv(
        market_path,
        dtype={"symbol": str},
        usecols=["date", "symbol", "open", "high", "low", "close", "volume", "amount"],
    )
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
    spans = market.groupby("symbol")["date"].agg(market_min="min", market_max="max")
    coverage = membership.join(spans, on="symbol")
    coverage["covered"] = (
        coverage["market_min"].notna()
        # A constituent can be suspended on a snapshot date.  A first trade
        # within 60 calendar days is coverage, not a reason to fabricate bars.
        & (coverage["market_min"] <= coverage["snapshot_date"] + pd.Timedelta(days=60))
        & (coverage["market_max"] >= coverage["snapshot_date"])
    )
    snapshot_coverage = coverage.groupby("snapshot_date")["covered"].sum()
    price_positive = bool((market[["open", "high", "low", "close"]] > 0).all().all())
    report = {
        "membership_rows": len(membership),
        "membership_snapshots": int(membership["snapshot_date"].nunique()),
        "membership_symbols": int(membership["symbol"].nunique()),
        "membership_min": str(membership["snapshot_date"].min().date()),
        "membership_max": str(membership["snapshot_date"].max().date()),
        "membership_min_size": int(counts.min()),
        "membership_max_size": int(counts.max()),
        "minimum_snapshot_market_coverage": int(snapshot_coverage.min()),
        "market_rows": len(market),
        "market_symbols": int(market["symbol"].nunique()),
        "market_min": str(market["date"].min().date()),
        "market_max": str(market["date"].max().date()),
        "price_positive": price_positive,
        "volume_coverage": float(market["volume"].notna().mean()),
        "amount_coverage": float(market["amount"].notna().mean()),
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
        "membership_starts_2010": report["membership_min"] <= "2010-01-04",
        "market_symbols_at_least_790": report["market_symbols"] >= 790,
        "snapshot_market_coverage_at_least_295": report["minimum_snapshot_market_coverage"] >= 295,
        "market_starts_2010": report["market_min"] <= "2010-01-04",
        "market_ends_2026_08_21": report["market_max"] >= "2026-08-21",
        "positive_prices": report["price_positive"],
        "volume_complete": report["volume_coverage"] == 1.0,
        "amount_complete": report["amount_coverage"] == 1.0,
        "fundamental_symbols_at_least_790": report["fundamental_symbols"] >= 790,
        "fundamental_pit_clean": report["fundamental_pit_violations"] == 0,
        "industry_symbols_at_least_750": report["industry_symbols"] >= 750,
        "industry_pit_clean": report["industry_future_violations"] == 0,
    }
    report["gates"] = gates
    report["passed"] = all(gates.values())
    if not report["passed"]:
        failed = ", ".join(name for name, passed in gates.items() if not passed)
        raise RuntimeError(f"V10输入质量门未通过: {failed}")
    return report
