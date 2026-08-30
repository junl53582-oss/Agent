from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


class IncompleteArtifactError(RuntimeError):
    pass


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


def _fsync_write(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def write_atomic_reservation(path: str | Path, value: Any) -> str:
    """Create the global date reservation using one O_EXCL filesystem operation."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    # The reservation deliberately remains if the process is interrupted after
    # O_EXCL succeeds.  That fail-closed behaviour prevents a same-day retry.
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_bytes(payload)


def write_immutable_bytes(path: str | Path, payload: bytes) -> str:
    """Write payload+sidecar with an explicit incomplete marker.

    Two path renames cannot be one filesystem transaction.  The marker makes a
    crash between them fail closed rather than appear intact.
    """
    target = Path(path)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    marker = target.with_suffix(target.suffix + ".incomplete")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or sidecar.exists() or marker.exists():
        raise FileExistsError(target)
    marker_digest = write_atomic_reservation(
        marker,
        {"target": target.name, "state": "WRITE_IN_PROGRESS", "retry_allowed": False},
    )
    del marker_digest
    digest = sha256_bytes(payload)
    payload_tmp: Path | None = None
    sidecar_tmp: Path | None = None
    try:
        fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        os.close(fd)
        payload_tmp = Path(name)
        payload_tmp.unlink()
        _fsync_write(payload_tmp, payload)
        fd, name = tempfile.mkstemp(prefix=f".{sidecar.name}.", suffix=".tmp", dir=target.parent)
        os.close(fd)
        sidecar_tmp = Path(name)
        sidecar_tmp.unlink()
        _fsync_write(sidecar_tmp, (digest + "\n").encode("ascii"))
        os.replace(payload_tmp, target)
        payload_tmp = None
        os.replace(sidecar_tmp, sidecar)
        sidecar_tmp = None
        # O_EXCL reservation files are intentionally read-only.  The transient
        # marker is the one exception: after both durable renames it must be
        # made removable on Windows before completing the transaction.
        marker.chmod(0o666)
        marker.unlink()
        return digest
    except Exception:
        for temporary in (payload_tmp, sidecar_tmp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        raise


def write_immutable_json(path: str | Path, value: Any) -> str:
    return write_immutable_bytes(path, canonical_json_bytes(value))


def write_immutable_frame(path: str | Path, frame: pd.DataFrame, keys: list[str]) -> str:
    return write_immutable_bytes(path, canonical_frame_bytes(frame, keys))


def verify_immutable(path: str | Path) -> str:
    target = Path(path)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    marker = target.with_suffix(target.suffix + ".incomplete")
    if marker.exists() or target.exists() != sidecar.exists():
        raise IncompleteArtifactError(f"immutable artifact is incomplete: {target}")
    if not target.exists():
        raise FileNotFoundError(target)
    expected = sidecar.read_text(encoding="ascii").strip()
    actual = sha256_file(target)
    if actual != expected:
        raise RuntimeError(f"immutable artifact hash mismatch: {target}")
    return actual


def read_verified_json(path: str | Path) -> dict:
    verify_immutable(path)
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"verified JSON manifest must be an object: {path}")
    return value
