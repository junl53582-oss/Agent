from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import jieba
import lightgbm
import scipy
import sklearn

from research_v10.quality import audit_v10_inputs
from research_v15.freeze import verify_research as verify_v15
from research_v15.quality import AMENDMENT_PATH, ORIGINAL_QUALITY_PATH, QUALITY_PATH
from research_v4.lock import sha256_file


PROTOCOL = "artifacts/research_v16/protocol.json"
LOCK = "artifacts/research_v16/plan.lock.json"
LOCK_HASH = "artifacts/research_v16/plan.lock.sha256"
DATA_FILES = [
    "data/universes/000300/history_v10.csv",
    "data/market_history_v10_hfq.csv",
    "data/fundamentals_pit_v10_extended.csv",
    "data/industry_history_v10.csv",
    "artifacts/research_v10/core_audit_report.json",
    "data/announcements_pit_v14.csv",
    "artifacts/research_v14/data_quality.json",
    "data/event_documents_pit_v15.csv",
    ORIGINAL_QUALITY_PATH,
    QUALITY_PATH,
]
CODE_FILES = [
    "research_v16/config.py", "research_v16/data.py", "research_v16/text_model.py",
    "research_v16/model.py", "research_v16/portfolio.py", "research_v16/backtest.py",
    "research_v16/validation.py", "research_v16/freeze.py", "research_v16/preflight.py",
    "research_v16/cli.py", "tests/test_research_v16.py",
]
DEPENDENCY_FILES = [
    "research_v15/config.py", "research_v15/features.py", "research_v15/data.py",
    "research_v15/text_model.py", "research_v15/model.py", "research_v15/portfolio.py",
    "research_v15/backtest.py", "research_v15/validation.py", "research_v15/freeze.py",
    "research_v15/cli.py", "research_v15/quality.py", "research_v15/preflight.py",
    "research_v14/features.py", "research_v14/model.py", "research_v14/freeze.py",
    "research_v13/config.py", "research_v13/model.py", "research_v12/features.py",
    "research_v12/model.py", "research_v10/features.py", "research_v10/model.py",
    "research_v10/portfolio.py", "research_v10/backtest.py", "research_v10/research_config.py",
    "research_v10/fundamentals.py", "research_v10/quality.py", "research_v9/data.py",
    "research_v9/features.py", "research_v6/model.py", "research_v5/models.py",
    "research_v4/stability.py", "research_v4/lock.py", "stockpilot/data.py",
    "stockpilot/membership.py", "stockpilot/portfolio.py", "stockpilot/trading.py",
]


def _environment() -> dict:
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "lightgbm": lightgbm.__version__,
        "scikit_learn": sklearn.__version__, "scipy": scipy.__version__,
        "jieba": jieba.__version__,
    }


def freeze_research(root: str | Path = ".") -> dict:
    workspace = Path(root)
    if (workspace / LOCK).exists() or (workspace / LOCK_HASH).exists():
        raise RuntimeError("V16冻结锁已存在，禁止覆盖")
    if (workspace / "artifacts/research_v16/run.started.json").exists() or (workspace / "artifacts/research_v16/report.json").exists():
        raise RuntimeError("V16已运行，不得重新冻结")
    v15_lock = verify_v15(workspace)
    quality = audit_v10_inputs(*(workspace / name for name in DATA_FILES[:4]))
    core = json.loads((workspace / DATA_FILES[4]).read_text(encoding="utf-8"))
    event_quality = json.loads((workspace / QUALITY_PATH).read_text(encoding="utf-8"))
    preflight = json.loads((workspace / "artifacts/research_v16/preflight.json").read_text(encoding="utf-8"))
    if not quality.get("passed") or not preflight.get("passed"):
        raise RuntimeError("V16基础数据或预检未通过")
    if not core.get("passed"):
        raise RuntimeError("核心复制审计未通过，禁止冻结V16")
    if not event_quality.get("passed") or not all(event_quality.get("gates", {}).values()):
        raise RuntimeError("V16复用的V15文本事件数据审计未通过，禁止冻结")
    if event_quality["event_document_sha256"] != sha256_file(workspace / "data/event_documents_pit_v15.csv"):
        raise RuntimeError("V16事件数据与验收报告不一致")
    if event_quality["amendment_sha256"] != sha256_file(workspace / AMENDMENT_PATH):
        raise RuntimeError("V15修订与验收报告不一致")
    dependencies = set(DEPENDENCY_FILES)
    for package in ["stockpilot", *[f"research_v{version}" for version in range(3, 16)]]:
        dependencies.update(path.relative_to(workspace).as_posix() for path in (workspace / package).rglob("*.py"))
    lock = {
        "protocol": "stockpilot-v16-pit-title-wordchar-ensemble-text-alpha",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(workspace / PROTOCOL),
        "v15_parent_lock_sha256": v15_lock["lock_sha256"],
        "data_sha256": {name: sha256_file(workspace / name) for name in DATA_FILES},
        "code_sha256": {name: sha256_file(workspace / name) for name in CODE_FILES},
        "dependency_sha256": {name: sha256_file(workspace / name) for name in sorted(dependencies)},
        "quality": quality,
        "core_audit": core["metrics"],
        "event_data_quality": event_quality,
        "environment": _environment(),
        "v16_model_performance_seen_before_lock": False,
    }
    target = workspace / LOCK
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = sha256_file(target)
    (workspace / LOCK_HASH).write_text(digest + "\n", encoding="utf-8")
    return {"lock_sha256": digest, **lock}


def verify_research(root: str | Path = ".") -> dict:
    workspace = Path(root)
    expected = (workspace / LOCK_HASH).read_text(encoding="utf-8").strip().upper()
    actual = sha256_file(workspace / LOCK)
    if actual != expected:
        raise RuntimeError(f"V16冻结锁不匹配: expected={expected}, actual={actual}")
    lock = json.loads((workspace / LOCK).read_text(encoding="utf-8"))
    checks = {
        PROTOCOL: lock["protocol_sha256"],
        **lock["data_sha256"], **lock["code_sha256"], **lock["dependency_sha256"],
    }
    mismatches = []
    for name, expected_hash in checks.items():
        path = workspace / name
        actual_hash = sha256_file(path) if path.exists() else "MISSING"
        if actual_hash != expected_hash:
            mismatches.append(f"{name}: expected={expected_hash}, actual={actual_hash}")
    if mismatches:
        raise RuntimeError("V16冻结数据或代码已变化:\n" + "\n".join(mismatches))
    v15_lock = verify_v15(workspace)
    if v15_lock["lock_sha256"] != lock["v15_parent_lock_sha256"]:
        raise RuntimeError("V15父锁已变化，V16拒绝运行")
    if _environment() != lock["environment"]:
        raise RuntimeError("V16运行环境与冻结时不一致")
    return {"lock_sha256": actual, **lock}
