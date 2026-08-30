from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from stockpilot.prospective_r2.integrity import read_verified_json, sha256_file, write_immutable_json
from stockpilot.prospective_r3.freeze import verify_lock as verify_v1r3
from stockpilot.prospective_r3.status import build_runtime_status

from .config import OperationalSettings


ROOT = Path(__file__).resolve().parents[2]


def verify_parent_locks() -> dict:
    parent = verify_v1r3()
    if parent.get("frozen_inputs_intact") is not True:
        raise RuntimeError("V1r3 parent is not intact")
    return {
        "parent_v1r3_lock_sha256": parent["v1r3_lock_sha256"],
        "v6_lock_sha256": parent["v6_lock_sha256"],
        "v18_lock_sha256": parent["v18_lock_sha256"],
        "v20r2_lock_sha256": parent["v20r2_corporate_action_lock_sha256"],
        "v30r1_forward_r2_lock_sha256": parent["v30r1_forward_r2_lock_sha256"],
        "all_parent_locks_intact": True,
    }


def _junit(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        "tests": sum(int(item.get("tests", 0)) for item in suites),
        "failures": sum(int(item.get("failures", 0)) for item in suites),
        "errors": sum(int(item.get("errors", 0)) for item in suites),
        "skipped": sum(int(item.get("skipped", 0)) for item in suites),
        "sha256": sha256_file(path),
    }


def _status(settings: OperationalSettings) -> dict:
    return build_runtime_status(settings, [], []).to_dict()


def prepare_freeze_artifacts(settings: OperationalSettings | None = None) -> dict:
    settings = settings or OperationalSettings()
    root = ROOT / settings.artifact_root
    targeted = _junit(root / "targeted_pytest.xml")
    full = _junit(root / "full_pytest.xml")
    status = _status(settings)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    audit = {
        "version": settings.version,
        "baseline_commit": "134c5c56686185ada7a9754cfefb32b8964e4c9a",
        "build_commit": commit,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "findings": [
            {
                "severity": "CRITICAL",
                "finding": "V1r3 reserved the only daily attempt before prediction inputs were checked",
                "resolution": "V1r4 hard preflight validates time and sealed inputs before reservation",
            },
            {
                "severity": "HIGH",
                "finding": "V1r3 had no post-close/upstream data-window gate",
                "resolution": "18:30 Asia/Shanghai conservative pre-reservation gate",
            },
            {
                "severity": "HIGH",
                "finding": "No approved official benchmark open-price evidence exists",
                "resolution": "freeze acquisition protocol and remain fail-closed UNAPPROVED",
            },
        ],
        "benchmark_status": "UNAPPROVED",
        "default_settlement_status": "SETTLEMENT_BLOCKED_BENCHMARK_UNAPPROVED",
        "real_provider_requests": 0,
        "financial_provider_requests": 0,
        "benchmark_acquisition_requests": 0,
        "v6_modified": False,
        "v30_logic_modified": False,
        "v30r1_logic_modified": False,
        "v31_trained": False,
        "model_retrain_runs": 0,
        "factor_research_runs": 0,
    }
    for name, value in {
        "audit.json": audit,
        "initial_status.json": status,
    }.items():
        write_immutable_json(root / name, value)
    paths = [
        *sorted((ROOT / "stockpilot/prospective_r4").glob("*.py")),
        ROOT / "tests/test_prospective_alpha_v1r4.py",
        *sorted(root.glob("*protocol.json")),
        *sorted(root.glob("*protocol.json.sha256")),
        root / "settlement_source_manifest.json",
        root / "settlement_source_manifest.json.sha256",
        root / "audit.json",
        root / "audit.json.sha256",
        root / "initial_status.json",
        root / "initial_status.json.sha256",
        root / "targeted_pytest.xml",
        root / "full_pytest.xml",
        ROOT / settings.v1r3_barrier_path,
    ]
    manifest = {
        "version": settings.version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths},
        "runtime_evidence_root": settings.data_root.as_posix(),
        "runtime_evidence_append_only": True,
        "provider_requests_made": 0,
        "model_retrain_runs": 0,
        "factor_research_runs": 0,
        "v31_trained": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    manifest_hash = write_immutable_json(root / "artifact_manifest.json", manifest)
    return {
        **status,
        "artifact_manifest_sha256": manifest_hash,
        "targeted": targeted,
        "full": full,
    }


def frozen_paths(settings: OperationalSettings | None = None) -> list[Path]:
    settings = settings or OperationalSettings()
    root = ROOT / settings.artifact_root
    paths = [
        *sorted((ROOT / "stockpilot/prospective_r4").glob("*.py")),
        ROOT / "tests/test_prospective_alpha_v1r4.py",
        root / "protocol.json",
        root / "protocol.json.sha256",
        root / "operational_preflight_protocol.json",
        root / "operational_preflight_protocol.json.sha256",
        root / "benchmark_protocol.json",
        root / "benchmark_protocol.json.sha256",
        root / "settlement_source_manifest.json",
        root / "settlement_source_manifest.json.sha256",
        root / "audit.json",
        root / "audit.json.sha256",
        root / "initial_status.json",
        root / "initial_status.json.sha256",
        root / "artifact_manifest.json",
        root / "artifact_manifest.json.sha256",
        root / "targeted_pytest.xml",
        root / "full_pytest.xml",
        ROOT / "artifacts/prospective_alpha_v1r3/plan.lock.json",
        ROOT / "artifacts/prospective_alpha_v1r3/plan.lock.json.sha256",
        ROOT / "artifacts/prediction_forward/v30r1_r2/plan.lock.json",
        ROOT / "artifacts/research_v6/plan.lock.json",
        ROOT / "artifacts/research_v18/plan.lock.json",
        ROOT / "artifacts/research_v20r2/plan.lock.json",
        ROOT / "artifacts/research_v20r2/plan.lock.sha256",
        ROOT / settings.calendar_path,
        ROOT / settings.v1r3_barrier_path,
    ]
    return list(dict.fromkeys(paths))


def create_lock(settings: OperationalSettings | None = None) -> dict:
    settings = settings or OperationalSettings()
    target = ROOT / settings.plan_lock_path
    if target.exists():
        raise RuntimeError("prospective alpha V1r4 is already frozen")
    parents = verify_parent_locks()
    paths = frozen_paths(settings)
    missing = [path.as_posix() for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"cannot freeze missing V1r4 paths: {missing}")
    payload = {
        "version": settings.version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "minimal-operational-closure",
        "parent_locks": parents,
        "calendar_sha256": sha256_file(ROOT / settings.calendar_path),
        "benchmark_status": "UNAPPROVED",
        "files": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths},
        "runtime_evidence_root": settings.data_root.as_posix(),
        "runtime_evidence_append_only": True,
        "provider_requests_made": 0,
        "model_retrain_runs": 0,
        "factor_research_runs": 0,
        "v6_modified": False,
        "v30_logic_modified": False,
        "v30r1_logic_modified": False,
        "v31_trained": False,
        "model_training_ready": False,
        "replacement_evaluation_ready": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    digest = write_immutable_json(target, payload)
    return payload | {"v1r4_lock_sha256": digest}


def verify_lock(settings: OperationalSettings | None = None) -> dict:
    settings = settings or OperationalSettings()
    target = ROOT / settings.plan_lock_path
    payload = read_verified_json(target)
    parents = verify_parent_locks()
    if payload.get("parent_locks") != parents:
        raise RuntimeError("V1r4 parent lock evidence changed")
    mismatches = [
        name for name, expected in payload["files"].items()
        if not (ROOT / name).exists() or sha256_file(ROOT / name) != expected
    ]
    if mismatches:
        raise RuntimeError(f"V1r4 frozen input changed: {mismatches}")
    return {
        "v1r4_lock_sha256": sha256_file(target),
        **parents,
        "frozen_inputs_intact": True,
        "model_training_ready": False,
        "replacement_evaluation_ready": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
        "v31_trained": False,
    }
