from __future__ import annotations

import argparse
import json

from .freeze import freeze_research, verify_research
from .validation import run_research_v19


def main() -> None:
    parser = argparse.ArgumentParser(description="V19 市场状态自适应权重研究")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    sub.add_parser("verify")
    sub.add_parser("run")
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_research()
    elif args.command == "verify":
        result = verify_research()
    else:
        result = run_research_v19()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
