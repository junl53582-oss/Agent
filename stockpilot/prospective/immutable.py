from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def canonical_frame_bytes(frame: pd.DataFrame, keys: list[str]) -> bytes:
    missing = set(keys) - set(frame.columns)
    if missing:
        raise ValueError(f"canonical frame keys missing: {sorted(missing)}")
    ordered = frame.sort_values(keys, kind="mergesort").reset_index(drop=True)
    return ordered.to_csv(
        index=False, lineterminator="\n", float_format="%.12g", na_rep=""
    ).encode("utf-8-sig")


def write_new_bytes(path: str | Path, payload: bytes) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(payload)
    digest = sha256_bytes(payload)
    with target.with_suffix(target.suffix + ".sha256").open("x", encoding="ascii") as stream:
        stream.write(digest + "\n")
    return digest


def write_new_json(path: str | Path, value: Any) -> str:
    return write_new_bytes(path, canonical_json_bytes(value))


def write_new_frame(path: str | Path, frame: pd.DataFrame, keys: list[str]) -> str:
    return write_new_bytes(path, canonical_frame_bytes(frame, keys))


def verify_sidecar(path: str | Path) -> str:
    target = Path(path)
    expected = target.with_suffix(target.suffix + ".sha256").read_text(encoding="ascii").strip()
    actual = sha256_file(target)
    if actual != expected:
        raise RuntimeError(f"immutable artifact hash mismatch: {target}")
    return actual
