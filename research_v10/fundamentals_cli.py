from __future__ import annotations

import argparse
import json

import pandas as pd

from .fundamentals import build_extended_fundamentals


def main() -> None:
    parser = argparse.ArgumentParser(description="构建V10扩展PIT财务数据")
    parser.add_argument("--membership", default="data/universes/000300/history_v10.csv")
    parser.add_argument("--output", default="data/fundamentals_pit_v10_extended.csv")
    args = parser.parse_args()
    membership = pd.read_csv(args.membership, dtype={"symbol": str})
    result, failures = build_extended_fundamentals(
        membership["symbol"], output_path=args.output
    )
    print(
        json.dumps(
            {"rows": len(result), "symbols": int(result.symbol.nunique()), "failures": len(failures)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

