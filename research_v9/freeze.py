from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_v4.lock import sha256_file

from .quality import audit_inputs


DATA_FILES = [
    "data/universes/000300/history_v9.csv",
    "data/market_history_v9.csv",
    "data/fundamentals_pit_v9.csv",
    "data/industry_history_v9.csv",
]
CODE_FILES = [
    "research_v9/config.py",
    "research_v9/data.py",
    "research_v9/data_cli.py",
    "research_v9/features.py",
    "research_v9/model.py",
    "research_v9/backtest.py",
    "research_v9/quality.py",
    "research_v9/freeze.py",
    "research_v9/validation.py",
    "research_v9/cli.py",
    "tests/test_research_v9.py",
]
PROTOCOL_PATH = "artifacts/research_v9/protocol.json"
LOCK_PATH = "artifacts/research_v9/plan.lock.json"
LOCK_HASH_PATH = "artifacts/research_v9/plan.lock.sha256"


def freeze(workspace: str | Path = ".") -> dict:
    root = Path(workspace)
    quality = audit_inputs(*(root / name for name in DATA_FILES))
    try:
        import lightgbm

        lightgbm_version = lightgbm.__version__
    except ImportError:  # pragma: no cover
        lightgbm_version = "missing"
    lock = {
        "protocol": "stockpilot-v9-three-line-frozen-study",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(root / PROTOCOL_PATH),
        "data_sha256": {name: sha256_file(root / name) for name in DATA_FILES},
        "code_sha256": {name: sha256_file(root / name) for name in CODE_FILES},
        "quality": quality,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lightgbm_version,
        },
        "valid_performance_seen_before_lock": False,
        "invalid_implementation_run_seen": True,
    }
    target = root / LOCK_PATH
    target.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = sha256_file(target)
    (root / LOCK_HASH_PATH).write_text(digest + "\n", encoding="utf-8")
    return {"lock_sha256": digest, **lock}


def verify(workspace: str | Path = ".") -> dict:
    root = Path(workspace)
    lock_path = root / LOCK_PATH
    hash_path = root / LOCK_HASH_PATH
    if not lock_path.exists() or not hash_path.exists():
        raise RuntimeError("V9冻结锁不存在，禁止运行绩效验证")
    expected = hash_path.read_text(encoding="utf-8").strip().upper()
    actual = sha256_file(lock_path)
    if actual != expected:
        raise RuntimeError(f"V9冻结锁哈希不匹配: expected={expected}, actual={actual}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks = {PROTOCOL_PATH: lock["protocol_sha256"], **lock["data_sha256"], **lock["code_sha256"]}
    mismatches = []
    for name, expected_hash in checks.items():
        path = root / name
        actual_hash = sha256_file(path) if path.exists() else "MISSING"
        if actual_hash != expected_hash:
            mismatches.append(f"{name}: expected={expected_hash}, actual={actual_hash}")
    if mismatches:
        raise RuntimeError("V9冻结输入或代码已变化:\n" + "\n".join(mismatches))
    return lock


if __name__ == "__main__":
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))
