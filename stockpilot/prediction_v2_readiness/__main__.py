from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import AuditSettings, run_and_write


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Prediction V2 new-information readiness")
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/prediction_v2/new_information_readiness"),
    )
    args = parser.parse_args()
    result = run_and_write(
        AuditSettings(
            source_root=args.source_root.resolve(),
            repo_root=args.repo_root.resolve(),
            artifact_dir=args.artifact_dir.resolve(),
        )
    )
    print(json.dumps(result["final_decision"], ensure_ascii=False, indent=2))
    return 0 if result["final_decision"]["audit_completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
