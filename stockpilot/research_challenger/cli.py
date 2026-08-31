from __future__ import annotations

import argparse
import json

from .config import ChallengerSettings


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical research-only ranking challenger")
    parser.add_argument(
        "command",
        choices=(
            "audit",
            "freeze",
            "freeze-amendment",
            "freeze-ci-amendment",
            "verify",
            "run",
            "report",
        ),
    )
    args = parser.parse_args()
    settings = ChallengerSettings()
    if args.command == "audit":
        from .data import factor_inventory, load_research_dataset

        data, evidence = load_research_dataset(settings)
        result = {
            "status": "READ_ONLY_AUDIT",
            "data": evidence,
            "factor_count": len(factor_inventory(data, settings)),
            "provider_requests": 0,
            "prospective_rows_used": 0,
        }
    elif args.command == "freeze":
        from .freeze import freeze_plan

        result = freeze_plan(settings)
    elif args.command == "verify":
        from .freeze import verify_plan_lock

        result = verify_plan_lock(settings)
    elif args.command == "freeze-amendment":
        from .freeze import freeze_amendment

        result = freeze_amendment(settings)
    elif args.command == "freeze-ci-amendment":
        from .freeze import freeze_ci_amendment

        result = freeze_ci_amendment(settings)
    elif args.command == "run":
        from .pipeline import run_v31

        result = run_v31(settings)
    else:
        path = settings.artifact_dir / "report.json"
        if not path.exists():
            raise RuntimeError("V31 report does not exist")
        result = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
