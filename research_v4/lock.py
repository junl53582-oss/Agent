from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import PLAN_LOCK_SHA256


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_plan_lock(
    path: str | Path, expected_sha256: str = PLAN_LOCK_SHA256
) -> dict:
    lock_path = Path(path)
    if not lock_path.exists():
        raise RuntimeError(f"V4预注册锁不存在: {lock_path}")
    actual = sha256_file(lock_path)
    if actual != expected_sha256.upper():
        raise RuntimeError(f"V4预注册锁哈希不匹配: expected={expected_sha256}, actual={actual}")
    return json.loads(lock_path.read_text(encoding="utf-8"))


def verify_locked_inputs(plan: dict, workspace: str | Path = ".") -> None:
    root = Path(workspace)
    mismatches = []
    for relative, expected in plan["data_sha256"].items():
        path = root / relative
        actual = sha256_file(path) if path.exists() else "MISSING"
        if actual != expected.upper():
            mismatches.append(f"{relative}: expected={expected}, actual={actual}")
    if mismatches:
        raise RuntimeError("V4输入数据与预注册版本不一致:\n" + "\n".join(mismatches))
