from __future__ import annotations

import argparse
import json

from .freeze import freeze_research, verify_research
from .validation import run_research_v12


def main() -> None:
    parser = argparse.ArgumentParser(description="V12冻结模型研究")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    sub.add_parser("verify")
    sub.add_parser("run")
    args = parser.parse_args()
    result = freeze_research() if args.command == "freeze" else verify_research() if args.command == "verify" else run_research_v12()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

