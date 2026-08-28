from __future__ import annotations

import argparse
import json

from .embed import build_embeddings
from .freeze import freeze_research, verify_research
from .validation import run_research_v18


def main() -> None:
    parser = argparse.ArgumentParser(description="V18 预训练语义嵌入文本研究")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("embed")
    sub.add_parser("freeze")
    sub.add_parser("verify")
    sub.add_parser("run")
    args = parser.parse_args()
    if args.command == "embed":
        result = {"shape": build_embeddings().shape}
    elif args.command == "freeze":
        result = freeze_research()
    elif args.command == "verify":
        result = verify_research()
    else:
        result = run_research_v18()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
