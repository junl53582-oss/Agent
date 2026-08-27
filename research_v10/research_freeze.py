from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_v4.lock import sha256_file

from .quality import audit_v10_inputs


PROTOCOL = "artifacts/research_v10/protocol.json"
LOCK = "artifacts/research_v10/plan.lock.json"
LOCK_HASH = "artifacts/research_v10/plan.lock.sha256"
DATA_FILES = [
    "data/universes/000300/history_v10.csv",
    "data/market_history_v10_hfq.csv",
    "data/fundamentals_pit_v10_extended.csv",
    "data/industry_history_v10.csv",
    "artifacts/research_v10/core_audit_report.json",
]
CODE_FILES = [
    "research_v10/research_config.py",
    "research_v10/fundamentals.py",
    "research_v10/features.py",
    "research_v10/model.py",
    "research_v10/portfolio.py",
    "research_v10/backtest.py",
    "research_v10/quality.py",
    "research_v10/research_freeze.py",
    "research_v10/validation.py",
    "research_v10/research_cli.py",
    "tests/test_research_v10_history.py",
    "tests/test_research_v10_model.py",
]
DEPENDENCY_FILES = [
    "stockpilot/data.py",
    "stockpilot/features.py",
    "stockpilot/membership.py",
    "stockpilot/model.py",
    "stockpilot/portfolio.py",
    "stockpilot/trading.py",
    "research_v3/features.py",
    "research_v3/fundamentals.py",
    "research_v4/config.py",
    "research_v4/stability.py",
    "research_v5/features.py",
    "research_v5/models.py",
    "research_v6/config.py",
    "research_v6/model.py",
    "research_v9/data.py",
    "research_v9/features.py",
]


def freeze_research(root: str | Path = ".") -> dict:
    workspace = Path(root)
    quality = audit_v10_inputs(*(workspace / name for name in DATA_FILES[:4]))
    core_audit = json.loads(
        (workspace / "artifacts/research_v10/core_audit_report.json").read_text(encoding="utf-8")
    )
    if not core_audit.get("passed"):
        raise RuntimeError("V10核心复制审计未通过，禁止冻结模型研究")
    try:
        import lightgbm

        lightgbm_version = lightgbm.__version__
    except ImportError:  # pragma: no cover
        lightgbm_version = "missing"
    lock = {
        "protocol": "stockpilot-v10-multihorizon-benchmark-relative",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(workspace / PROTOCOL),
        "data_sha256": {name: sha256_file(workspace / name) for name in DATA_FILES},
        "code_sha256": {name: sha256_file(workspace / name) for name in CODE_FILES},
        "dependency_sha256": {
            name: sha256_file(workspace / name) for name in DEPENDENCY_FILES
        },
        "quality": quality,
        "core_audit": core_audit["metrics"],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lightgbm_version,
        },
        "v10_model_performance_seen_before_lock": False,
    }
    target = workspace / LOCK
    target.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = sha256_file(target)
    (workspace / LOCK_HASH).write_text(digest + "\n", encoding="utf-8")
    return {"lock_sha256": digest, **lock}


def verify_research(root: str | Path = ".") -> dict:
    workspace = Path(root)
    expected = (workspace / LOCK_HASH).read_text(encoding="utf-8").strip().upper()
    actual = sha256_file(workspace / LOCK)
    if actual != expected:
        raise RuntimeError(f"V10研究冻结锁不匹配: expected={expected}, actual={actual}")
    lock = json.loads((workspace / LOCK).read_text(encoding="utf-8"))
    checks = {
        PROTOCOL: lock["protocol_sha256"],
        **lock["data_sha256"],
        **lock["code_sha256"],
        **lock["dependency_sha256"],
    }
    mismatches = []
    for name, expected_hash in checks.items():
        path = workspace / name
        actual_hash = sha256_file(path) if path.exists() else "MISSING"
        if actual_hash != expected_hash:
            mismatches.append(f"{name}: expected={expected_hash}, actual={actual_hash}")
    if mismatches:
        raise RuntimeError("V10冻结数据或代码已变化:\n" + "\n".join(mismatches))
    return {"lock_sha256": actual, **lock}

