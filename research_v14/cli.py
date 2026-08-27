from __future__ import annotations

import argparse
import json

from .freeze import freeze_research, verify_research
from .quality import audit_v14_external
from .validation import run_research_v14


def main() -> None:
    parser = argparse.ArgumentParser(description="V14 PIT公告增强冻结研究")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit-data")
    sub.add_parser("freeze")
    sub.add_parser("verify")
    sub.add_parser("run")
    args = parser.parse_args()
    if args.command == "audit-data":
        result = audit_v14_external()
    elif args.command == "freeze":
        result = freeze_research()
    elif args.command == "verify":
        result = verify_research()
    else:
        result = run_research_v14()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
