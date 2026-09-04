from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import AcquisitionSettings, audit_acquisition, write_outputs
from .probe import probe_eastmoney_report_schema


def main() -> int:
    parser = argparse.ArgumentParser(description="Prediction V2 data-acquisition foundation")
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/prediction_v2/data_acquisition"),
    )
    parser.add_argument(
        "--import-root", type=Path, default=Path("data/prediction_v2_acquisition/import")
    )
    parser.add_argument(
        "--probe-path",
        type=Path,
        default=Path("data/prediction_v2_acquisition/probes/eastmoney_report_schema.json"),
    )
    args = parser.parse_args()
    settings = AcquisitionSettings(
        source_root=args.source_root.resolve(),
        repo_root=args.repo_root.resolve(),
        artifact_dir=args.artifact_dir.resolve(),
        import_root=args.import_root.resolve(),
    )
    probe = probe_eastmoney_report_schema(args.probe_path.resolve())
    result = audit_acquisition(settings, probe)
    write_outputs(settings, result)
    print(json.dumps(result["acquisition_decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
