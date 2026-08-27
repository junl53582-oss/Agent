from __future__ import annotations

import argparse
import json

from .freeze import freeze, verify
from .validation import run_research_v9


def main() -> None:
    parser = argparse.ArgumentParser(description="V9冻结研究")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    sub.add_parser("verify")
    sub.add_parser("run")
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze()
    elif args.command == "verify":
        result = verify()
    else:
        result = run_research_v9()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

