from __future__ import annotations

import argparse
import json

from stockpilot.membership import load_membership_history

from .fundamentals import fetch_fundamentals
from .validation import run_research_v3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StockPilot独立Research V3")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fundamentals-fetch", help="下载带公告日的PIT基本面")
    fetch.add_argument("--membership", default="data/universes/000300/history.csv")
    fetch.add_argument("--output", default="data/fundamentals_pit.csv")
    fetch.add_argument("--workers", type=int, default=4, choices=range(1, 7))
    run = sub.add_parser("run", help="运行V3多周期集成和嵌套走步验证")
    run.add_argument("--input", default="data/market_history.csv")
    run.add_argument("--membership", default="data/universes/000300/history.csv")
    run.add_argument("--exposures", default="data/exposures.csv")
    run.add_argument("--fundamentals", default="data/fundamentals_pit.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "fundamentals-fetch":
        history = load_membership_history(args.membership)
        data, failures = fetch_fundamentals(
            history["symbol"].drop_duplicates(), args.output, workers=args.workers
        )
        print(
            json.dumps(
                {
                    "rows": len(data),
                    "symbols": int(data["symbol"].nunique()),
                    "failures": failures,
                    "pit_violations": int((data["available_date"] < data["report_date"]).sum()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(
        json.dumps(
            run_research_v3(args.input, args.membership, args.exposures, args.fundamentals),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
