from __future__ import annotations

import argparse
import json

from .freeze import freeze_research, verify_research
from .preflight import run_preflight
from .validation import run_research_v16


def main() -> None:
    parser = argparse.ArgumentParser(description="V16 PIT公告标题char+word集成文本研究")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("freeze")
    sub.add_parser("verify")
    sub.add_parser("run")
    args = parser.parse_args()
    if args.command == "preflight":
        result = run_preflight()
    elif args.command == "freeze":
        result = freeze_research()
    elif args.command == "verify":
        result = verify_research()
    else:
        result = run_research_v16()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
