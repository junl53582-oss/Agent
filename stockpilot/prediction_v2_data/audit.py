from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import (
    validate_actual_versions,
    validate_analyst_estimates,
    validate_announcement_documents,
)


@dataclass(frozen=True)
class AcquisitionSettings:
    source_root: Path
    repo_root: Path
    artifact_dir: Path
    import_root: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> str:
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)
    digest = hashlib.sha256(raw).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(f"{digest}\n", encoding="ascii")
    return digest


def _candidate(path: Path, validator, protocol: dict) -> dict:
    if not path.exists():
        return {
            "path": path.as_posix(),
            "exists": False,
            "passed": False,
            "reason": "NO_APPROVED_IMPORT_DELIVERED",
        }
    frame = pd.read_csv(path, dtype={"symbol": "string"})
    return {
        "path": path.as_posix(),
        "exists": True,
        "sha256": sha256_file(path),
        **validator(frame, protocol),
    }


def audit_acquisition(settings: AcquisitionSettings, probe: dict[str, Any]) -> dict[str, Any]:
    protocol_path = settings.artifact_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    candidates = {
        "announcement_documents": _candidate(
            settings.import_root / "announcement_documents.csv",
            validate_announcement_documents,
            protocol,
        ),
        "analyst_estimates": _candidate(
            settings.import_root / "analyst_estimates.csv", validate_analyst_estimates, protocol
        ),
        "actual_versions": _candidate(
            settings.import_root / "actual_versions.csv", validate_actual_versions, protocol
        ),
    }
    all_ready = all(item["passed"] for item in candidates.values())
    parent = json.loads(
        (
            settings.repo_root
            / "artifacts/prediction_v2/new_information_readiness/readiness_audit.json"
        ).read_text(encoding="utf-8")
    )
    existing = {
        "announcement_titles": parent["sources"]["announcement_titles"],
        "announcement_bodies": parent["sources"]["announcement_bodies"],
        "analyst_report_metadata": parent["sources"]["analyst_report_metadata"],
        "analyst_consensus_vintages": parent["sources"]["analyst_consensus_vintages"],
        "fundamental_actuals": parent["sources"]["fundamental_actuals"],
    }
    return {
        "protocol": protocol["protocol"],
        "audit_date": "2026-09-05",
        "protocol_sha256": sha256_file(protocol_path),
        "parent_readiness_sha256": sha256_file(
            settings.repo_root
            / "artifacts/prediction_v2/new_information_readiness/readiness_audit.json"
        ),
        "scope": {
            "model_training": False,
            "return_labels_read": False,
            "alpha_selection": False,
            "paid_source_activated": False,
            "production_modified": False,
        },
        "existing_repository_evidence": existing,
        "public_schema_probe": probe,
        "import_candidates": candidates,
        "supplier_shortlist": [
            {
                "provider": "Wind Client API",
                "official_evidence": "official page advertises earnings forecasts and historical backtracking",
                "official_url": "https://www.wind.com.cn/portal/zh/ClientApi/index.html",
                "status": "SAMPLE_AND_LICENSE_REVIEW_REQUIRED",
            },
            {
                "provider": "RESSET",
                "official_evidence": "official manual describes analyst forecast error and divergence data",
                "official_url": "https://manual.resset.com/RESSETRTAS.pdf",
                "status": "SAMPLE_AND_LICENSE_REVIEW_REQUIRED",
            },
            {
                "provider": "CSMAR",
                "official_evidence": "official database offers historical and custom research data",
                "official_url": (
                    "https://www.csmar.com/channels/"
                    "%E4%B8%AD%E5%9B%BD%E7%BB%8F%E6%B5%8E%E9%87%91%E8%9E%8D%E7%A0%94%E7%A9%B6"
                    "%E6%95%B0%E6%8D%AE%E5%BA%93%EF%BC%88CSMAR%EF%BC%89.html"
                ),
                "status": "EXACT_ANALYST_VINTAGE_FIELDS_NOT_PUBLICLY_CONFIRMED",
            },
        ],
        "acquisition_decision": {
            "all_imports_ready": all_ready,
            "status": (
                "PREDICTION_V2_DATA_FOUNDATION_READY"
                if all_ready
                else "PREDICTION_V2_DATA_ACQUISITION_BLOCKED"
            ),
            "blocking_items": [
                name for name, result in candidates.items() if not result["passed"]
            ],
            "historical_analyst_expectations": "PROCUREMENT_OR_APPROVED_DELIVERY_REQUIRED",
            "announcement_archive": "RECONSTRUCTION_ALLOWED_BUT_NOT_YET_HISTORICAL_PIT_APPROVED",
            "level_2": "DEFERRED_NOT_REQUIRED",
            "challenger_experiment": "NOT_STARTED",
        },
    }


def render_report(result: dict[str, Any]) -> str:
    decision = result["acquisition_decision"]
    probe = result["public_schema_probe"]
    candidates = result["import_candidates"]
    lines = [
        "# PREDICTION_V2_EVENT_AND_EXPECTATION_DATA_ACQUISITION_REPORT",
        "",
        f"Status: `{decision['status']}`",
        "",
        "## Work completed",
        "",
        "- Frozen supplier-neutral schemas and PIT/revision rules for announcement documents, analyst estimates, and actual versions.",
        "- Added deterministic validators and a strict pre-release earnings-surprise constructor.",
        "- Performed one bounded Eastmoney schema probe; no response rows were committed or admitted for training.",
        "- Audited candidate import locations without training a model or reading return labels.",
        "",
        "## Public analyst-source probe",
        "",
        f"- Requests this run: {probe.get('network_requests')}",
        f"- Requests across the immutable probe lineage: {probe.get('network_requests_total')}",
        f"- Raw response SHA256: `{probe.get('raw_sha256')}`",
        f"- Response currentYear: `{probe.get('current_year')}`",
        f"- Forecast fields: `{', '.join(probe.get('forecast_fields', []))}`",
        f"- Explicit forecast period per value: `{probe.get('has_explicit_forecast_period_per_value')}`",
        f"- Revision/supersession link: `{probe.get('has_revision_or_supersession_link')}`",
        "- Training admission: `REJECTED`. Current/next-year dynamic fields cannot be safely rebound to old report dates.",
        "",
        "## Import gates",
        "",
    ]
    for name, candidate in candidates.items():
        lines.append(f"- {name}: `{'PASS' if candidate['passed'] else 'FAIL'}` — {candidate.get('reason', 'validation failed')}")
    lines.extend(
        [
            "",
            "## Required external action",
            "",
            "Request sample extracts from Wind, RESSET, and optionally CSMAR using the committed schema and supplier questionnaire. No purchase should be approved until a sample demonstrates original publication timestamps, stable record IDs, historical revisions, delisted-stock coverage, and permitted local model use.",
            "",
            "CNInfo body reconstruction may proceed as a separate archive job, but reconstructed documents remain non-training evidence until announcement-version lineage is proved. Level-2 remains deferred.",
            "",
            "## Model boundary",
            "",
            "Gen2 and production DAILY prediction remain unchanged. `PREDICTION_V2_BOUNDED_CHALLENGER_EXPERIMENT` was not started.",
            "",
            "## Final status",
            "",
            f"`{decision['status']}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(settings: AcquisitionSettings, result: dict[str, Any]) -> None:
    _write_json(settings.artifact_dir / "acquisition_audit.json", result)
    report = settings.artifact_dir / "PREDICTION_V2_EVENT_AND_EXPECTATION_DATA_ACQUISITION_REPORT.md"
    report.write_text(render_report(result), encoding="utf-8", newline="\n")
    report.with_name(f"{report.name}.sha256").write_text(
        f"{sha256_file(report)}\n", encoding="ascii"
    )
    files = sorted(
        path
        for path in settings.artifact_dir.iterdir()
        if path.is_file() and not path.name.startswith("artifact_manifest")
    )
    _write_json(
        settings.artifact_dir / "artifact_manifest.json",
        {"files": {path.name: sha256_file(path) for path in files}},
    )
