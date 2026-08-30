from __future__ import annotations

import argparse
import json

from stockpilot.prospective_r2.observation import load_verified_observations

from .config import OperationalSettings
from .freeze import create_lock, prepare_freeze_artifacts, verify_lock
from .orchestrator import run_daily
from .settlement import load_verified_label_records
from .status import build_runtime_status


def status(settings: OperationalSettings) -> dict:
    return build_runtime_status(
        settings,
        load_verified_observations(settings),
        load_verified_label_records(settings.labels_root),
    ).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Official Prospective Alpha V1r3 evidence-derived operations"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    daily = commands.add_parser("daily", help="run the only authorized new daily chain")
    daily.add_argument("--date", dest="target_date")
    commands.add_parser("status")
    commands.add_parser("verify")
    commands.add_parser("prepare-freeze-artifacts")
    commands.add_parser("freeze")
    args = parser.parse_args()
    settings = OperationalSettings()
    if args.command == "daily":
        result = run_daily(target_date=args.target_date, settings=settings)
    elif args.command == "status":
        result = status(settings)
    elif args.command == "verify":
        result = verify_lock(settings)
    elif args.command == "prepare-freeze-artifacts":
        result = prepare_freeze_artifacts(settings)
    elif args.command == "freeze":
        result = create_lock(settings)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
