from __future__ import annotations

import argparse
import json

from .config import OperationalSettings
from .freeze import create_lock, prepare_freeze_artifacts, verify_lock, verify_parent_locks
from .labels import load_verified_label_records
from .observation import load_verified_observations
from .orchestrator import run_daily
from .readiness import derive_readiness


def status(settings: OperationalSettings) -> dict:
    observations = load_verified_observations(settings)
    labels = load_verified_label_records(settings.labels_root)
    return {
        "active_version": settings.version,
        **derive_readiness(
            observations, labels, thresholds=settings.thresholds
        ).to_dict(),
        "v31_trained": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Official Prospective Alpha V1r2 operations"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    daily = subparsers.add_parser("daily", help="run the single official daily chain")
    daily.add_argument("--date", dest="target_date")
    subparsers.add_parser("status")
    subparsers.add_parser("verify")
    subparsers.add_parser("freeze")
    subparsers.add_parser("prepare-freeze-artifacts")
    args = parser.parse_args()
    settings = OperationalSettings()
    if args.command == "daily":
        result = run_daily(target_date=args.target_date, settings=settings)
    elif args.command == "status":
        result = status(settings)
    elif args.command == "verify":
        result = verify_lock(settings)
    elif args.command == "freeze":
        result = create_lock(settings)
    elif args.command == "prepare-freeze-artifacts":
        result = prepare_freeze_artifacts(settings)
    else:  # pragma: no cover
        result = verify_parent_locks()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
