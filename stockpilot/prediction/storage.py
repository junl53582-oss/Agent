from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def canonical_prediction_csv(frame: pd.DataFrame) -> bytes:
    ordered = frame.sort_values([column for column in ("date", "rank_5d", "symbol") if column in frame.columns])
    return ordered.to_csv(index=False, lineterminator="\n", float_format="%.10g").encode("utf-8-sig")


def write_immutable_prediction_snapshot(frame: pd.DataFrame, path: Path) -> tuple[bool, str]:
    payload = canonical_prediction_csv(frame)
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != payload:
            raise RuntimeError(f"immutable prediction snapshot hash mismatch: {path}")
        return False, digest
    path.write_bytes(payload)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    return True, digest


def write_latest_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

