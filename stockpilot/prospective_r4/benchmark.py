from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stockpilot.prospective_r2.integrity import read_verified_json, sha256_file, verify_immutable


APPROVED_IDENTITY = {
    "name": "CSI 300 Price Index",
    "code": "000300",
    "provider": "China Securities Index Co., Ltd.",
    "instrument_type": "PRICE_INDEX",
    "price_field": "official_open",
}


def verify_benchmark_evidence(
    benchmark: dict,
    *,
    as_of: str | pd.Timestamp,
) -> dict:
    """Verify an approved official index-open source without accepting proxies."""
    if benchmark.get("status") != "APPROVED":
        raise RuntimeError("BENCHMARK_UNAPPROVED")
    identity = benchmark.get("identity")
    if identity != APPROVED_IDENTITY:
        raise RuntimeError("BENCHMARK_IDENTITY_UNAPPROVED")
    path = Path(benchmark["path"])
    source_sidecar = path.with_suffix(path.suffix + ".sha256")
    evidence_path = Path(benchmark["evidence_manifest_path"])
    source_hash = sha256_file(path)
    if not source_sidecar.exists() or source_sidecar.read_text(encoding="ascii").strip() != source_hash:
        raise RuntimeError("BENCHMARK_SOURCE_SIDECAR_INVALID")
    evidence = read_verified_json(evidence_path)
    evidence_hash = verify_immutable(evidence_path)
    if evidence.get("identity") != APPROVED_IDENTITY:
        raise RuntimeError("BENCHMARK_EVIDENCE_IDENTITY_INVALID")
    if evidence.get("source_path") != path.as_posix() or evidence.get("source_sha256") != source_hash:
        raise RuntimeError("BENCHMARK_EVIDENCE_BINDING_INVALID")
    if evidence.get("source_kind") != "OFFICIAL_INDEX_OPEN_SERIES" or evidence.get("fallback_allowed") is not False:
        raise RuntimeError("BENCHMARK_PROXY_OR_FALLBACK_FORBIDDEN")
    frame = pd.read_csv(path)
    if set(frame.columns) != {"date", "open"}:
        raise ValueError("BENCHMARK_SCHEMA_INVALID")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
    if frame.empty or frame["date"].duplicated().any():
        raise ValueError("BENCHMARK_DUPLICATE_OR_EMPTY")
    if not frame["date"].is_monotonic_increasing:
        raise ValueError("BENCHMARK_DATES_NOT_ORDERED")
    if not np.isfinite(frame["open"]).all() or (frame["open"] <= 0).any():
        raise ValueError("BENCHMARK_OPEN_INVALID")
    cutoff = pd.Timestamp(as_of).normalize()
    if (frame["date"] > cutoff).any():
        raise ValueError("BENCHMARK_CONTAINS_DATA_AFTER_AS_OF")
    if evidence.get("date_min") != str(frame["date"].min().date()) or evidence.get("date_max") != str(frame["date"].max().date()):
        raise RuntimeError("BENCHMARK_EVIDENCE_COVERAGE_MISMATCH")
    if int(evidence.get("rows", -1)) != len(frame):
        raise RuntimeError("BENCHMARK_EVIDENCE_ROW_COUNT_MISMATCH")
    return {
        "status": "APPROVED",
        "identity": identity,
        "path": path.as_posix(),
        "sha256": source_hash,
        "evidence_manifest_path": evidence_path.as_posix(),
        "evidence_manifest_sha256": evidence_hash,
        "date_min": str(frame["date"].min().date()),
        "date_max": str(frame["date"].max().date()),
        "rows": len(frame),
    }
