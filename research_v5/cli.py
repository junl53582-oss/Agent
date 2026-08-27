from __future__ import annotations

import argparse
import json

from .validation import run_research_v5


def main() -> None:
    parser = argparse.ArgumentParser(description="StockPilot锁定Research V5多维行业专家模型")
    parser.add_argument("--input", default="data/market_history.csv")
    parser.add_argument("--membership", default="data/universes/000300/history.csv")
    parser.add_argument("--exposures", default="data/exposures.csv")
    parser.add_argument("--fundamentals", default="data/fundamentals_pit.csv")
    args = parser.parse_args()
    report = run_research_v5(args.input, args.membership, args.exposures, args.fundamentals)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
