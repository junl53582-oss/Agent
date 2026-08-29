from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def _result_files(root: Path) -> list[Path]:
    candidates = [
        root / "protocol.json",
        root / "plan.lock.json",
        root / "runtime_status.json",
        root / "certification" / "status.json",
        root / "models" / "manifest.json",
        root / "validation" / "report.json",
        root / "live" / "latest.json",
    ]
    for directory in (root / "models", root / "validation", root / "live" / "predictions"):
        if directory.exists():
            candidates.extend(path for path in directory.iterdir() if path.is_file())
    return sorted(set(candidates), key=lambda path: path.as_posix())


def create_result_lock(root: Path, parent_lock: Path) -> dict:
    target = root / "validation.lock.json"
    if target.exists():
        raise RuntimeError(f"result lock already exists: {target}")
    files = [parent_lock, *_result_files(root)]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise RuntimeError("cannot lock missing result files: " + ", ".join(missing))
    certification = json.loads((root / "certification" / "status.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    payload = {
        "version": protocol.get("version", root.name.replace("prediction_", "").upper()),
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "immutable_result": True,
        "production_prediction_ready": bool(certification["production_prediction_ready"]),
        "execution_authorized": False,
        "files": {path.as_posix(): sha256(path) for path in files},
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload | {"lock_sha256": sha256(target)}


def verify_result_lock(root: Path) -> dict:
    target = root / "validation.lock.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    mismatches = [
        name
        for name, expected in payload["files"].items()
        if not Path(name).exists() or sha256(Path(name)) != expected
    ]
    return {"intact": not mismatches, "mismatches": mismatches, "lock_sha256": sha256(target)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or verify an immutable prediction result lock")
    parser.add_argument("mode", choices=("create", "verify"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--parent-lock", type=Path)
    args = parser.parse_args()
    if args.mode == "create":
        if args.parent_lock is None:
            parser.error("--parent-lock is required for create")
        result = create_result_lock(args.root, args.parent_lock)
    else:
        result = verify_result_lock(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
