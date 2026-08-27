from __future__ import annotations

import argparse
import json

from .data import build_v15_event_data
from .freeze import freeze_research, verify_research
from .validation import run_research_v15
from .quality import audit_event_data
from .preflight import run_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="V15 PIT公告标题文本研究")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-data")
    sub.add_parser("audit-data")
    sub.add_parser("preflight")
    sub.add_parser("freeze")
    sub.add_parser("verify")
    sub.add_parser("run")
    args = parser.parse_args()
    if args.command == "build-data":
        result = build_v15_event_data()
    elif args.command == "audit-data":
        result = audit_event_data()
    elif args.command == "preflight":
        result = run_preflight()
    elif args.command == "freeze":
        result = freeze_research()
    elif args.command == "verify":
        result = verify_research()
    else:
        result = run_research_v15()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
