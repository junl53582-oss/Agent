from __future__ import annotations

import argparse
import json

from stockpilot.prospective_r2.observation import load_verified_observations
from stockpilot.prospective_r3.settlement import load_verified_label_records
from stockpilot.prospective_r3.status import build_runtime_status

from .config import OperationalSettings
from .freeze import create_lock, prepare_freeze_artifacts, verify_lock
from .orchestrator import run_daily
from .preflight import run_preflight, seal_prediction_inputs
from .settlement import certify_operational_label


def status(settings: OperationalSettings) -> dict:
    return build_runtime_status(
        settings,
        load_verified_observations(settings),
        load_verified_label_records(settings.labels_root),
        label_certifier=certify_operational_label,
    ).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prospective Alpha V1r4 operational closure")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "seal-inputs", "daily"):
        command = commands.add_parser(name)
        command.add_argument("--date", dest="target_date")
    commands.add_parser("status")
    commands.add_parser("verify")
    commands.add_parser("prepare-freeze-artifacts")
    commands.add_parser("freeze")
    args = parser.parse_args()
    settings = OperationalSettings()
    if args.command == "preflight":
        result = run_preflight(target_date=args.target_date, settings=settings)
    elif args.command == "seal-inputs":
        if not args.target_date:
            parser.error("seal-inputs requires --date")
        result = seal_prediction_inputs(args.target_date, settings=settings)
    elif args.command == "daily":
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
