from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_v10.quality import audit_v10_inputs
from research_v4.lock import sha256_file


PROTOCOL = "artifacts/research_v14/protocol.json"
DATA_PROTOCOL = "artifacts/research_v14/data_protocol.json"
DATA_AMENDMENT = "artifacts/research_v14/data_amendment_001.json"
MODEL_AMENDMENT = "artifacts/research_v14/model_amendment_002.json"
DATA_AMENDMENT_003 = "artifacts/research_v14/data_amendment_003.json"
DATA_AMENDMENT_004 = "artifacts/research_v14/data_amendment_004.json"
DATA_AMENDMENT_005 = "artifacts/research_v14/data_amendment_005.json"
LOCK = "artifacts/research_v14/plan.lock.json"
LOCK_HASH = "artifacts/research_v14/plan.lock.sha256"
BASE_DATA_FILES = [
    "data/universes/000300/history_v10.csv",
    "data/market_history_v10_hfq.csv",
    "data/fundamentals_pit_v10_extended.csv",
    "data/industry_history_v10.csv",
    "artifacts/research_v10/core_audit_report.json",
]
EXTERNAL_DATA_FILES = [
    "data/analyst_reports_pit_v14.csv",
    "data/northbound_holdings_pit_v14.csv",
    "data/announcements_pit_v14.csv",
    "artifacts/research_v14/preflight.json",
    "artifacts/research_v14/data_quality.json",
]
CODE_FILES = [
    "research_v14/data_config.py",
    "research_v14/fetch_external.py",
    "research_v14/quality.py",
    "research_v14/config.py",
    "research_v14/features.py",
    "research_v14/model.py",
    "research_v14/portfolio.py",
    "research_v14/backtest.py",
    "research_v14/validation.py",
    "research_v14/freeze.py",
    "research_v14/cli.py",
    "tests/test_research_v14.py",
]
DEPENDENCY_FILES = [
    "research_v13/config.py",
    "research_v13/model.py",
    "research_v12/config.py",
    "research_v12/features.py",
    "research_v12/model.py",
    "research_v10/features.py",
    "research_v10/model.py",
    "research_v10/portfolio.py",
    "research_v10/backtest.py",
    "research_v10/research_config.py",
    "research_v10/fundamentals.py",
    "research_v10/quality.py",
    "research_v9/data.py",
    "research_v9/features.py",
    "research_v6/model.py",
    "research_v5/models.py",
    "research_v4/stability.py",
    "research_v4/lock.py",
    "stockpilot/data.py",
    "stockpilot/membership.py",
    "stockpilot/portfolio.py",
    "stockpilot/trading.py",
]


def freeze_research(root: str | Path = ".") -> dict:
    workspace = Path(root)
    quality = audit_v10_inputs(*(workspace / name for name in BASE_DATA_FILES[:4]))
    core = json.loads((workspace / BASE_DATA_FILES[4]).read_text(encoding="utf-8"))
    external = json.loads((workspace / EXTERNAL_DATA_FILES[-1]).read_text(encoding="utf-8"))
    if not core.get("passed"):
        raise RuntimeError("核心复制审计未通过，禁止冻结V14")
    if external.get("accepted_sources") != ["announcements"]:
        raise RuntimeError(
            "V14外部数据门槛不满足：必须仅公告源通过，且分析师/北向源保持排除；"
            f"actual={external.get('accepted_sources')}"
        )
    if not all(external.get("gates", {}).get("announcements", {}).values()):
        raise RuntimeError("公告PIT数据审计未全部通过，禁止冻结V14")
    import lightgbm
    data_files = [*BASE_DATA_FILES, *EXTERNAL_DATA_FILES]
    lock = {
        "protocol": "stockpilot-v14-pit-announcement-event-gated-two-stage",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(workspace / PROTOCOL),
        "data_protocol_sha256": sha256_file(workspace / DATA_PROTOCOL),
        "data_amendment_sha256": sha256_file(workspace / DATA_AMENDMENT),
        "model_amendment_sha256": sha256_file(workspace / MODEL_AMENDMENT),
        "data_amendment_003_sha256": sha256_file(workspace / DATA_AMENDMENT_003),
        "data_amendment_004_sha256": sha256_file(workspace / DATA_AMENDMENT_004),
        "data_amendment_005_sha256": sha256_file(workspace / DATA_AMENDMENT_005),
        "data_sha256": {name: sha256_file(workspace / name) for name in data_files},
        "code_sha256": {name: sha256_file(workspace / name) for name in CODE_FILES},
        "dependency_sha256": {name: sha256_file(workspace / name) for name in DEPENDENCY_FILES},
        "quality": quality,
        "core_audit": core["metrics"],
        "external_data_quality": external,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "v14_model_performance_seen_before_lock": False,
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
        raise RuntimeError(f"V14冻结锁不匹配: expected={expected}, actual={actual}")
    lock = json.loads((workspace / LOCK).read_text(encoding="utf-8"))
    checks = {
        PROTOCOL: lock["protocol_sha256"],
        DATA_PROTOCOL: lock["data_protocol_sha256"],
        DATA_AMENDMENT: lock["data_amendment_sha256"],
        MODEL_AMENDMENT: lock["model_amendment_sha256"],
        DATA_AMENDMENT_003: lock["data_amendment_003_sha256"],
        DATA_AMENDMENT_004: lock["data_amendment_004_sha256"],
        DATA_AMENDMENT_005: lock["data_amendment_005_sha256"],
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
        raise RuntimeError("V14冻结数据或代码已变化:\n" + "\n".join(mismatches))
    return {"lock_sha256": actual, **lock}
