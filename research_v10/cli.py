from __future__ import annotations

import argparse
import json

from .audit import run_core_audit
from .freeze import freeze_audit_lock, verify_audit_lock


def main() -> None:
    parser = argparse.ArgumentParser(description="V10分阶段冻结研究")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit-freeze")
    sub.add_parser("audit-verify")
    sub.add_parser("audit-run")
    args = parser.parse_args()
    if args.command == "audit-freeze":
        result = freeze_audit_lock()
    elif args.command == "audit-verify":
        result = verify_audit_lock()
    else:
        result = run_core_audit()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

