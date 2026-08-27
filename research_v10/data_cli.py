from __future__ import annotations

import argparse
import json

from .data import normalize_cached_market


def main() -> None:
    parser = argparse.ArgumentParser(description="V10独立数据规范化")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output", default="data/market_history_v10.csv")
    args = parser.parse_args()
    _, report = normalize_cached_market(args.raw_dir, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

