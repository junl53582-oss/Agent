from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from .pipeline import acquire_market, materialize_features, verify_daily_feature_partition
from .runtime import preflight, seal_inputs, verify_effective_daily_runtime_freeze


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="StockPilot append-only daily PIT input pipeline")
    commands = value.add_subparsers(dest="command", required=True)
    acquire = commands.add_parser("acquire-market")
    acquire.add_argument("target_date")
    acquire.add_argument(
        "--confirm-real-provider-acquisition",
        action="store_true",
        help="required acknowledgement before AkShare provider calls",
    )
    materialize = commands.add_parser("materialize-features")
    materialize.add_argument("target_date")
    verify = commands.add_parser("verify")
    verify.add_argument("target_date")
    seal = commands.add_parser("seal-inputs")
    seal.add_argument("target_date")
    gate = commands.add_parser("preflight")
    gate.add_argument("target_date")
    commands.add_parser("verify-activation")
    return value


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    now = datetime.now(timezone.utc)
    if args.command == "acquire-market":
        if not args.confirm_real_provider_acquisition:
            raise SystemExit(
                "REAL_PROVIDER_ACQUISITION_REQUIRED: rerun with --confirm-real-provider-acquisition"
            )
        print("REAL_PROVIDER_ACQUISITION_REQUIRED", flush=True)
        result = acquire_market(args.target_date, [], now=now)
    elif args.command == "materialize-features":
        result = materialize_features(args.target_date)
    elif args.command == "verify":
        result = verify_daily_feature_partition(args.target_date)
    elif args.command == "seal-inputs":
        result = seal_inputs(args.target_date, now=now)
    elif args.command == "preflight":
        result = preflight(args.target_date, now=now)
    else:
        result = verify_effective_daily_runtime_freeze()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
