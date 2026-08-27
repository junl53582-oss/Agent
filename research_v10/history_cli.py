from __future__ import annotations

import argparse
import json

import pandas as pd

from .history_data import fetch_hfq_history


def main() -> None:
    parser = argparse.ArgumentParser(description="V10后复权历史行情抓取")
    parser.add_argument("--membership", default="data/universes/000300/history_v10.csv")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-08-21")
    parser.add_argument("--output", default="data/market_history_v10_hfq.csv")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    membership = pd.read_csv(args.membership, dtype={"symbol": str})
    panel, failures = fetch_hfq_history(
        membership["symbol"], args.start, args.end, args.output, workers=args.workers
    )
    print(
        json.dumps(
            {
                "rows": len(panel),
                "symbols": int(panel["symbol"].nunique()),
                "failures": len(failures),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

