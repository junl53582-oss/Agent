from __future__ import annotations

import argparse

import pandas as pd

from .data import fetch_industry_history


def main() -> None:
    parser = argparse.ArgumentParser(description="V9独立PIT数据工具")
    sub = parser.add_subparsers(dest="command", required=True)
    industry = sub.add_parser("industry")
    industry.add_argument("--membership", required=True)
    industry.add_argument("--end", default="2026-08-21")
    industry.add_argument("--output", default="data/industry_history_v9.csv")
    args = parser.parse_args()
    membership = pd.read_csv(args.membership, dtype={"symbol": str})
    result, failures = fetch_industry_history(
        membership["symbol"], args.end, args.output
    )
    print(
        {
            "rows": len(result),
            "symbols": int(result["symbol"].nunique()),
            "failures": len(failures),
        }
    )


if __name__ == "__main__":
    main()

