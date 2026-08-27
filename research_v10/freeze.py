from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from research_v4.lock import sha256_file
AUDIT_PROTOCOL = "artifacts/research_v10/audit_protocol.json"
AUDIT_LOCK = "artifacts/research_v10/audit.lock.json"
AUDIT_HASH = "artifacts/research_v10/audit.lock.sha256"
AUDIT_DATA = [
    "data/universes/000300/history_v9.csv",
    "data/market_history_v10.csv",
]
AUDIT_CODE = [
    "research_v10/config.py",
    "research_v10/data.py",
    "research_v10/data_cli.py",
    "research_v10/audit.py",
    "research_v10/freeze.py",
    "research_v10/cli.py",
    "tests/test_research_v10.py",
]


def freeze_audit_lock(root: str | Path = ".") -> dict:
    workspace = Path(root)
    lock = {
        "protocol": "stockpilot-v10-core-replication-audit",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(workspace / AUDIT_PROTOCOL),
        "data_sha256": {name: sha256_file(workspace / name) for name in AUDIT_DATA},
        "code_sha256": {name: sha256_file(workspace / name) for name in AUDIT_CODE},
        "performance_seen_before_lock": False,
    }
    target = workspace / AUDIT_LOCK
    target.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = sha256_file(target)
    (workspace / AUDIT_HASH).write_text(digest + "\n", encoding="utf-8")
    return {"lock_sha256": digest, **lock}


def verify_audit_lock(root: str | Path = ".") -> dict:
    workspace = Path(root)
    expected = (workspace / AUDIT_HASH).read_text(encoding="utf-8").strip().upper()
    actual = sha256_file(workspace / AUDIT_LOCK)
    if actual != expected:
        raise RuntimeError(f"V10核心复制冻结锁不匹配: expected={expected}, actual={actual}")
    lock = json.loads((workspace / AUDIT_LOCK).read_text(encoding="utf-8"))
    checks = {
        AUDIT_PROTOCOL: lock["protocol_sha256"],
        **lock["data_sha256"],
        **lock["code_sha256"],
    }
    mismatches = []
    for name, expected_hash in checks.items():
        actual_hash = sha256_file(workspace / name) if (workspace / name).exists() else "MISSING"
        if actual_hash != expected_hash:
            mismatches.append(f"{name}: expected={expected_hash}, actual={actual_hash}")
    if mismatches:
        raise RuntimeError("V10核心复制输入或代码已变化:\n" + "\n".join(mismatches))
    return {"lock_sha256": actual, **lock}
