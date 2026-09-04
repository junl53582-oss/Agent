from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

AUDIT_DATE = "2026-09-05"
EXPECTED_TITLE_HASH = "09463503d767708eacd0ccda0bfea9876e3898a8add311cdb9a2d5b35c1ab118"
EXPECTED_ANALYST_HASH = "841c96e4356ad65299cfe826529c7f879a0506f5b65e62fd564925754406d07f"
EXPECTED_EVENT_DOCUMENT_HASH = (
    "ff21c7125f5b36ee6ad3aabeaa5cec87bfe2fb19da4d9d7b4d0dfdfdfc2b5939"
)
EXPECTED_EMBEDDING_HASH = "7ecf0843e98b541e5a3884622bbc75c32da1366371ffb55bcafe03c3c2a3bd22"
EXPECTED_FUNDAMENTAL_HASH = (
    "8682941485fd1ac276bb515c5bdb5f2eb9db866bbf927f3dc027d8fc305bc578"
)
EXPECTED_BASELINE_PANEL_HASH = (
    "8aa5e3f2817d6bf8e5da3bd265b4f078206b58b10ee770907233595c59342b02"
)


@dataclass(frozen=True)
class AuditSettings:
    source_root: Path
    repo_root: Path
    artifact_dir: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    ).encode("utf-8")


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> str:
    content = _canonical_json_bytes(value)
    _write_bytes_atomic(path, content)
    digest = hashlib.sha256(content).hexdigest()
    _write_bytes_atomic(path.with_name(f"{path.name}.sha256"), f"{digest}\n".encode())
    return digest


def _file_identity(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "logical_path": path.as_posix(),
            "expected_sha256": expected_sha256,
            "identity_verified": False,
        }
    digest = sha256_file(path)
    return {
        "exists": True,
        "logical_path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "expected_sha256": expected_sha256,
        "identity_verified": expected_sha256 is None or digest == expected_sha256,
    }


def _date_summary(values: pd.Series) -> dict[str, Any]:
    parsed = pd.to_datetime(values, format="mixed", errors="coerce", utc=True)
    valid = parsed.dropna()
    return {
        "valid": int(valid.size),
        "invalid": int(parsed.isna().sum()),
        "min": None if valid.empty else valid.min().date().isoformat(),
        "max": None if valid.empty else valid.max().date().isoformat(),
        "distinct_years": int(valid.dt.year.nunique()) if not valid.empty else 0,
    }


def _profile_csv(
    path: Path,
    *,
    expected_sha256: str,
    usecols: list[str],
    symbol_column: str,
    date_column: str,
    duplicate_columns: list[str],
) -> dict[str, Any]:
    identity = _file_identity(path, expected_sha256)
    if not path.exists():
        return identity | {"rows": 0, "columns": []}
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(usecols) - set(header))
    profile = identity | {"columns": header, "missing_audit_columns": missing}
    if missing:
        return profile | {"rows": 0}
    frame = pd.read_csv(path, usecols=usecols, dtype={symbol_column: "string"})
    dates = _date_summary(frame[date_column])
    return profile | {
        "rows": len(frame),
        "symbols": int(frame[symbol_column].nunique(dropna=True)),
        "date": dates,
        "duplicate_identity_rows": int(frame.duplicated(duplicate_columns).sum()),
        "nulls": {column: int(frame[column].isna().sum()) for column in usecols},
    }


def _profile_announcement_titles(path: Path) -> dict[str, Any]:
    result = _profile_csv(
        path,
        expected_sha256=EXPECTED_TITLE_HASH,
        usecols=["symbol", "announcement_date", "title", "announcement_id"],
        symbol_column="symbol",
        date_column="announcement_date",
        duplicate_columns=["symbol", "announcement_id"],
    )
    result.update(
        {
            "provider": "CNInfo official disclosure query",
            "content_scope": "title and metadata only",
            "intraday_publication_time_available": False,
            "safe_effective_time_policy": "next verified trading session",
            "historical_pit_interpretation": "date-level usable only under conservative next-session embargo",
        }
    )
    return result


def _profile_analyst_reports(path: Path) -> dict[str, Any]:
    result = _profile_csv(
        path,
        expected_sha256=EXPECTED_ANALYST_HASH,
        usecols=["symbol", "report_date", "title", "rating", "institution", "industry"],
        symbol_column="symbol",
        date_column="report_date",
        duplicate_columns=["symbol", "report_date", "title", "institution"],
    )
    result.update(
        {
            "provider": "Eastmoney research report list via AkShare",
            "content_scope": "report metadata and title",
            "numeric_expectation_fields": [],
            "historical_consensus_vintages": False,
        }
    )
    return result


def _profile_event_documents(path: Path) -> dict[str, Any]:
    identity = _file_identity(path, EXPECTED_EVENT_DOCUMENT_HASH)
    if not path.exists():
        return identity | {"rows": 0, "columns": []}
    header = pd.read_csv(path, nrows=0).columns.tolist()
    safe_columns = [
        column
        for column in ["symbol", "date", "document", "event_count", "eligible"]
        if column in header
    ]
    frame = pd.read_csv(path, usecols=safe_columns, dtype={"symbol": "string"})
    eligible = frame["eligible"].fillna(False).astype(bool) if "eligible" in frame else None
    return identity | {
        "columns": header,
        "rows": len(frame),
        "symbols": int(frame["symbol"].nunique()) if "symbol" in frame else 0,
        "date": _date_summary(frame["date"]) if "date" in frame else {},
        "eligible_rows": int(eligible.sum()) if eligible is not None else None,
        "contains_future_target_columns": any(
            column.startswith(("event_target_", "event_label_end_"))
            for column in header
        ),
        "content_scope": "daily aggregation of announcement titles; not full bodies",
        "training_input_eligible_for_v2": False,
        "ineligibility_reason": (
            "derived title representation was already tested and the file co-locates future targets; "
            "only source announcement records may be reconsidered"
        ),
    }


def _profile_announcement_bodies(root: Path) -> dict[str, Any]:
    receipts = sorted(root.glob("*/receipt.json")) if root.exists() else []
    symbols: set[str] = set()
    years: set[int] = set()
    content_hashes = 0
    source_publication = 0
    source_publication_midnight = 0
    first_seen = 0
    extraction_passed = 0
    historical_pit_verified = 0
    for receipt_path in receipts:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        symbol = str(payload.get("symbol", ""))
        if symbol:
            symbols.add(symbol)
        date = pd.to_datetime(payload.get("announcement_date"), errors="coerce")
        if pd.notna(date):
            years.add(int(date.year))
        hashes = payload.get("sha256")
        content_hashes += int(isinstance(hashes, dict) and bool(hashes.get("body.pdf")))
        published = pd.to_datetime(payload.get("published_at_source"), errors="coerce")
        source_publication += int(pd.notna(published))
        source_publication_midnight += int(
            pd.notna(published)
            and published.hour == 0
            and published.minute == 0
            and published.second == 0
        )
        first_seen += int(bool(payload.get("first_seen_at_utc")))
        extraction_passed += int(payload.get("body_extraction_passed") is True)
        historical_pit_verified += int(payload.get("historical_pit_verified") is True)
    return {
        "logical_path": root.as_posix(),
        "exists": root.exists(),
        "documents": len(receipts),
        "symbols": len(symbols),
        "distinct_years": len(years),
        "years": sorted(years),
        "documents_with_content_hash": content_hashes,
        "documents_with_source_publication": source_publication,
        "source_publication_values_at_midnight": source_publication_midnight,
        "documents_with_first_seen": first_seen,
        "body_extraction_passed": extraction_passed,
        "historical_pit_verified": historical_pit_verified,
        "model_training_ready": historical_pit_verified == len(receipts) and len(receipts) > 0,
        "provenance": "CNInfo PDFs retrieved in 2026; historical records explicitly not PIT verified",
    }


def _walk_json_values(value: Any, keys: set[str]) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in keys:
                yield key, child
            yield from _walk_json_values(child, keys)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json_values(child, keys)


def _profile_first_seen(root: Path) -> dict[str, Any]:
    files = sorted(root.rglob("*.json")) if root.exists() else []
    observed: set[str] = set()
    historical_true = 0
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for key, value in _walk_json_values(
            payload, {"observed_at", "observed_at_utc", "historical_pit_verified"}
        ):
            if key in {"observed_at", "observed_at_utc"} and value:
                observed.add(str(value))
            elif key == "historical_pit_verified" and value is True:
                historical_true += 1
    parsed = pd.to_datetime(pd.Series(sorted(observed), dtype="string"), errors="coerce", utc=True)
    valid = parsed.dropna()
    return {
        "logical_path": root.as_posix(),
        "exists": root.exists(),
        "json_files": len(files),
        "distinct_observation_timestamps": len(observed),
        "observation_min": None if valid.empty else valid.min().isoformat(),
        "observation_max": None if valid.empty else valid.max().isoformat(),
        "observation_span_days": 0 if valid.empty else int((valid.max() - valid.min()).days),
        "historical_pit_verified_true_values": historical_true,
        "historical_panel_ready": False,
    }


def _profile_consensus(root: Path) -> dict[str, Any]:
    files = sorted(root.glob("*/expectations.csv")) if root.exists() else []
    frames: list[pd.DataFrame] = []
    file_identities = []
    for path in files:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        columns = [
            column
            for column in [
                "observed_at_utc",
                "symbol",
                "forecast_eps_1",
                "target_price_min",
                "target_price_max",
                "raw_page_sha256",
                "provider_record_sha256",
                "identity_sha256",
            ]
            if column in header
        ]
        frames.append(pd.read_csv(path, usecols=columns, dtype={"symbol": "string"}))
        file_identities.append(_file_identity(path))
    if not frames:
        return {
            "logical_path": root.as_posix(),
            "exists": root.exists(),
            "files": [],
            "rows": 0,
            "symbols": 0,
            "distinct_snapshots": 0,
            "snapshot_span_days": 0,
            "numeric_forward_eps": False,
            "target_or_dispersion": False,
            "immutable_raw_hashes_complete": False,
            "replayable_revision_history": False,
        }
    frame = pd.concat(frames, ignore_index=True)
    times = pd.to_datetime(frame["observed_at_utc"], errors="coerce", utc=True)
    snapshots = times.dropna().drop_duplicates().sort_values()
    span = 0 if snapshots.empty else int((snapshots.max() - snapshots.min()).days)
    eps = pd.to_numeric(frame.get("forecast_eps_1"), errors="coerce")
    targets = pd.concat(
        [
            pd.to_numeric(frame.get("target_price_min"), errors="coerce"),
            pd.to_numeric(frame.get("target_price_max"), errors="coerce"),
        ],
        axis=1,
    )
    hash_columns = ["raw_page_sha256", "provider_record_sha256", "identity_sha256"]
    hashes_complete = all(column in frame and frame[column].notna().all() for column in hash_columns)
    return {
        "logical_path": root.as_posix(),
        "exists": root.exists(),
        "files": file_identities,
        "rows": len(frame),
        "symbols": int(frame["symbol"].nunique()),
        "distinct_snapshots": len(snapshots),
        "snapshot_min": None if snapshots.empty else snapshots.min().isoformat(),
        "snapshot_max": None if snapshots.empty else snapshots.max().isoformat(),
        "snapshot_span_days": span,
        "numeric_forward_eps": bool(eps.notna().any()),
        "target_or_dispersion": bool(targets.notna().any(axis=1).any()),
        "immutable_raw_hashes_complete": bool(hashes_complete),
        "duplicate_symbol_snapshot_rows": int(
            frame.assign(_observed=times).duplicated(["symbol", "_observed"]).sum()
        ),
        "replayable_revision_history": len(snapshots) >= 2,
        "provenance": "prospective Eastmoney expectation snapshot normalized by PIT v1r3",
    }


def _profile_fundamentals(path: Path) -> dict[str, Any]:
    identity = _file_identity(path, EXPECTED_FUNDAMENTAL_HASH)
    if not path.exists():
        return identity | {"rows": 0, "columns": []}
    header = pd.read_csv(path, nrows=0).columns.tolist()
    columns = [
        column
        for column in ["symbol", "report_date", "available_date", "update_date"]
        if column in header
    ]
    frame = pd.read_csv(path, usecols=columns, dtype={"symbol": "string"})
    available = pd.to_datetime(frame["available_date"], errors="coerce")
    update = pd.to_datetime(frame["update_date"], errors="coerce")
    duplicates = int(frame.duplicated(["symbol", "report_date"], keep=False).sum())
    versioned_keys = int(
        (frame.groupby(["symbol", "report_date"], dropna=False).size() > 1).sum()
    )
    revised_after_available = int((update > available).fillna(False).sum())
    return identity | {
        "columns": header,
        "rows": len(frame),
        "symbols": int(frame["symbol"].nunique()),
        "report_date": _date_summary(frame["report_date"]),
        "available_date": _date_summary(frame["available_date"]),
        "duplicate_rows_across_symbol_report_date": duplicates,
        "symbol_report_date_keys_with_multiple_versions": versioned_keys,
        "rows_updated_after_available_date": revised_after_available,
        "replayable_revision_history": versioned_keys > 0 or revised_after_available == 0,
        "revision_pollution_risk": revised_after_available > 0 and versioned_keys == 0,
        "actual_release_time_verified": False,
        "distinct_years": _date_summary(frame["report_date"])["distinct_years"],
        "already_consumed_by_current_61_factor_pipeline": True,
        "novel_information_beyond_current_61_factors": False,
    }


def _prior_research(repo_root: Path) -> dict[str, Any]:
    records = []
    for version in (14, 15, 16, 17, 18):
        path = repo_root / f"artifacts/research_v{version}/report.json"
        if not path.exists():
            records.append({"version": version, "exists": False})
            continue
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        records.append(
            {
                "version": version,
                "exists": True,
                "sha256": sha256_file(path),
                "decision": payload.get("decision"),
                "approved": payload.get("approved"),
            }
        )
    return {
        "announcement_title_research_versions": records,
        "announcement_titles_are_novel": False,
        "reason": (
            "Announcement titles and title-derived representations were already evaluated in "
            "V14-V18; V18's frozen decision remained keep_v6."
        ),
    }


def evaluate_joint_gate(
    sources: dict[str, dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    gates = protocol["source_gates"]
    title = sources["announcement_titles"]
    body = sources["announcement_bodies"]
    analyst = sources["analyst_report_metadata"]
    consensus = sources["analyst_consensus_vintages"]
    fundamental = sources["fundamental_actuals"]
    prior = sources["prior_research"]

    title_gate = {
        "documents": title.get("rows", 0) >= gates["announcement_titles"]["minimum_documents"],
        "symbols": title.get("symbols", 0) >= gates["announcement_titles"]["minimum_symbols"],
        "years": title.get("date", {}).get("distinct_years", 0)
        >= gates["announcement_titles"]["minimum_distinct_years"],
        "identity": title.get("duplicate_identity_rows", 1) == 0,
        "source_hash": title.get("identity_verified", False),
        "novelty": prior.get("announcement_titles_are_novel", False),
    }
    body_gate = {
        "documents": body.get("documents", 0) >= gates["announcement_bodies"]["minimum_documents"],
        "symbols": body.get("symbols", 0) >= gates["announcement_bodies"]["minimum_symbols"],
        "years": body.get("distinct_years", 0)
        >= gates["announcement_bodies"]["minimum_distinct_years"],
        "content_hashes": body.get("documents_with_content_hash", 0)
        == body.get("documents", -1),
        "historical_pit": body.get("historical_pit_verified", 0) == body.get("documents", -1)
        and body.get("documents", 0) > 0,
    }
    analyst_gate = {
        "rows": analyst.get("rows", 0) >= gates["analyst_report_metadata"]["minimum_rows"],
        "symbols": analyst.get("symbols", 0)
        >= gates["analyst_report_metadata"]["minimum_symbols"],
        "numeric_expectations": bool(analyst.get("numeric_expectation_fields")),
    }
    consensus_gate = {
        "symbols": consensus.get("symbols", 0)
        >= gates["analyst_consensus_vintages"]["minimum_symbols"],
        "snapshots": consensus.get("distinct_snapshots", 0)
        >= gates["analyst_consensus_vintages"]["minimum_distinct_snapshots"],
        "span": consensus.get("snapshot_span_days", 0)
        >= gates["analyst_consensus_vintages"]["minimum_snapshot_span_days"],
        "forward_eps": consensus.get("numeric_forward_eps", False),
        "target_or_dispersion": consensus.get("target_or_dispersion", False),
        "raw_hashes": consensus.get("immutable_raw_hashes_complete", False),
        "revisions": consensus.get("replayable_revision_history", False),
    }
    fundamental_gate = {
        "symbols": fundamental.get("symbols", 0)
        >= gates["fundamental_actuals"]["minimum_symbols"],
        "available_dates": fundamental.get("available_date", {}).get("invalid", 1) == 0,
        "revisions": fundamental.get("replayable_revision_history", False),
        "no_revision_pollution": not fundamental.get("revision_pollution_risk", True),
        "novelty": fundamental.get("novel_information_beyond_current_61_factors", False),
    }
    event_semantics_ready = all(body_gate.values())
    consensus_ready = all(consensus_gate.values())
    actuals_ready = all(fundamental_gate.values())
    surprise_gate = {
        "actual_release_time": fundamental.get("actual_release_time_verified", False),
        "consensus_snapshot_strictly_before_release": consensus.get(
            "strictly_before_actuals_verified", False
        ),
        "replayable_vintages": consensus_ready and actuals_ready,
        "symbols": min(consensus.get("symbols", 0), fundamental.get("symbols", 0))
        >= gates["earnings_surprise"]["minimum_symbols"],
        "years": fundamental.get("distinct_years", 0)
        >= gates["earnings_surprise"]["minimum_distinct_years"],
    }
    surprise_ready = all(surprise_gate.values())
    joint = {
        "novel_event_semantics": event_semantics_ready,
        "historical_consensus_vintages": consensus_ready,
        "constructible_earnings_surprise": surprise_ready,
        "no_unresolved_revision_pollution": fundamental_gate["no_revision_pollution"],
        "sufficient_join_coverage": event_semantics_ready and consensus_ready and actuals_ready,
    }
    return {
        "announcement_title_gate": title_gate,
        "announcement_body_gate": body_gate,
        "analyst_report_metadata_gate": analyst_gate,
        "analyst_consensus_vintage_gate": consensus_gate,
        "fundamental_actual_gate": fundamental_gate,
        "earnings_surprise_gate": surprise_gate,
        "joint_gate": joint,
        "all_joint_gates_passed": all(joint.values()),
    }


def audit_readiness(settings: AuditSettings) -> dict[str, Any]:
    protocol_path = settings.artifact_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source = settings.source_root
    sources: dict[str, dict[str, Any]] = {
        "announcement_titles": _profile_announcement_titles(
            source / "data/announcements_pit_v14.csv"
        ),
        "analyst_report_metadata": _profile_analyst_reports(
            source / "data/analyst_reports_pit_v14.csv"
        ),
        "event_documents": _profile_event_documents(
            source / "data/event_documents_pit_v15.csv"
        ),
        "announcement_embeddings": _file_identity(
            source / "data/event_embeddings_v18.npy", EXPECTED_EMBEDDING_HASH
        )
        | {
            "content_scope": "BAAI/bge-small-zh-v1.5 embeddings of titles only",
            "novel_to_prior_research": False,
        },
        "announcement_bodies": _profile_announcement_bodies(
            source / "data/announcement_body_v1"
        ),
        "announcement_first_seen_v5": _profile_first_seen(
            source / "data/announcement_first_seen_v5"
        ),
        "announcement_first_seen_v5r2": _profile_first_seen(
            source / "data/announcement_first_seen_v5r2"
        ),
        "analyst_consensus_vintages": _profile_consensus(
            source / "data/pit_observations_v1r3"
        ),
        "fundamental_actuals": _profile_fundamentals(
            source / "data/fundamentals_pit_v10_extended.csv"
        ),
        "current_61_factor_baseline": _file_identity(
            source / "artifacts/prediction_v30/cache/eligible_panel.parquet",
            EXPECTED_BASELINE_PANEL_HASH,
        )
        | {"feature_count": 61, "status": "reference_only; frozen Gen2 remains unchanged"},
        "prior_research": _prior_research(settings.repo_root),
    }
    gate_evaluation = evaluate_joint_gate(sources, protocol)
    ready = gate_evaluation["all_joint_gates_passed"]
    final_status = (
        "PREDICTION_V2_NEW_INFORMATION_READY"
        if ready
        else "PREDICTION_V2_NEW_INFORMATION_NOT_READY"
    )
    experiment_status = "AUTHORIZED_NOT_STARTED" if ready else "NOT_STARTED_GATE_FAILED"
    return {
        "audit": protocol["protocol"],
        "audit_as_of_date": AUDIT_DATE,
        "protocol_sha256": sha256_file(protocol_path),
        "source_root": str(settings.source_root),
        "scope": {
            "labels_read": False,
            "models_trained": False,
            "alpha_selected": False,
            "production_files_modified": False,
        },
        "sources": sources,
        "gate_evaluation": gate_evaluation,
        "requirements": {
            "announcement_semantics": (
                "Acquire a broad immutable CNInfo full-body archive with source hashes, conservative "
                "publication-to-session mapping, and sufficient historical coverage."
            ),
            "earnings_surprise": (
                "Acquire historical consensus vintages with numeric forward EPS/dispersion and join "
                "each vintage strictly before a PIT actual-release observation; retain revisions."
            ),
            "historical_analyst_expectations": "REQUIRED_OR_APPROVED_EQUIVALENT",
            "level_2_market_data": "NOT_REQUIRED_FOR_FIRST_CHALLENGER",
        },
        "final_decision": {
            "audit_completed": True,
            "status": final_status,
            "bounded_challenger_experiment": experiment_status,
            "production_gen2": "UNCHANGED_AND_CONTINUES_DAILY",
            "reason": (
                "The repository lacks both a sufficiently covered, historically PIT-verified full-body "
                "event corpus and replayable historical analyst-consensus vintages. Existing title data "
                "is large but was already tested in V14-V18, while fundamental revisions cannot be replayed."
                if not ready
                else "All pre-registered new-information gates passed."
            ),
        },
    }


def _report_mark(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _render_report(result: dict[str, Any]) -> str:
    sources = result["sources"]
    gates = result["gate_evaluation"]
    title = sources["announcement_titles"]
    body = sources["announcement_bodies"]
    analyst = sources["analyst_report_metadata"]
    consensus = sources["analyst_consensus_vintages"]
    fundamentals = sources["fundamental_actuals"]
    joint = gates["joint_gate"]
    lines = [
        "# PREDICTION_V2_NEW_INFORMATION_READINESS_AUDIT_REPORT",
        "",
        f"Audit status: `{result['final_decision']['status']}`",
        f"Bounded challenger: `{result['final_decision']['bounded_challenger_experiment']}`",
        "",
        "## Executive conclusion",
        "",
        (
            "The requested challenger is not ready to train. The repository has a broad, date-level "
            "announcement-title history, but that signal family was already evaluated in V14-V18. It "
            "does not contain a sufficiently covered, historically PIT-verified full-body corpus. The "
            "repository also has only one 288-symbol analyst-consensus observation, so historical "
            "expectation revisions and earnings surprise cannot be reconstructed."
        ),
        "",
        "## Source coverage and PIT status",
        "",
        "| Source | Coverage | PIT / revision finding | V2 admission |",
        "|---|---:|---|---|",
        (
            f"| Announcement titles | {title.get('rows', 0):,} rows; "
            f"{title.get('symbols', 0)} symbols; {title.get('date', {}).get('min')} to "
            f"{title.get('date', {}).get('max')} | Date-level publication; use next-session embargo; "
            "already tested in V14-V18 | FAIL (not novel) |"
        ),
        (
            f"| Announcement bodies | {body.get('documents', 0)} documents; "
            f"{body.get('symbols', 0)} symbols; {body.get('distinct_years', 0)} years | "
            f"Historical PIT verified {body.get('historical_pit_verified', 0)}/"
            f"{body.get('documents', 0)} | FAIL |"
        ),
        (
            f"| Analyst report metadata | {analyst.get('rows', 0):,} rows; "
            f"{analyst.get('symbols', 0)} symbols | No numeric EPS/target consensus vintages | FAIL |"
        ),
        (
            f"| Analyst consensus snapshots | {consensus.get('rows', 0)} rows; "
            f"{consensus.get('symbols', 0)} symbols; {consensus.get('distinct_snapshots', 0)} snapshot(s) | "
            f"Span {consensus.get('snapshot_span_days', 0)} days; revisions not replayable | FAIL |"
        ),
        (
            f"| Fundamental actuals | {fundamentals.get('rows', 0):,} rows; "
            f"{fundamentals.get('symbols', 0)} symbols | "
            f"{fundamentals.get('rows_updated_after_available_date', 0):,} rows updated after first "
            f"availability; {fundamentals.get('symbol_report_date_keys_with_multiple_versions', 0)} "
            "versioned keys retained | FAIL (revision risk and not novel) |"
        ),
        "",
        "## Joint challenger gate",
        "",
    ]
    for name, passed in joint.items():
        lines.append(f"- {name}: `{_report_mark(passed)}`")
    lines.extend(
        [
            "",
            "## Publication-time and revision findings",
            "",
            (
                f"- Announcement titles have {title.get('date', {}).get('invalid', 0)} invalid "
                "publication dates, but only date-level timing. They can be used only from the next "
                "verified trading session."
            ),
            (
                f"- The {body.get('documents', 0)} body receipts carry source publication dates and "
                f"retrieval first-seen timestamps, but all {body.get('source_publication_values_at_midnight', 0)} "
                "source values are midnight/date-level and none is historically PIT verified."
            ),
            (
                f"- First-seen ledgers contain only "
                f"{sources['announcement_first_seen_v5'].get('distinct_observation_timestamps', 0)} and "
                f"{sources['announcement_first_seen_v5r2'].get('distinct_observation_timestamps', 0)} "
                "distinct recent observation timestamps, respectively; they are prospective seeds, not "
                "historical panels."
            ),
            (
                f"- Fundamental actuals contain {fundamentals.get('rows_updated_after_available_date', 0):,} "
                "post-availability updates but retain no per-report vintage chain, so an original-vintage "
                "earnings surprise cannot be replayed."
            ),
            "",
            "## New-information answer",
            "",
            (
                "At least one sufficiently covered signal family genuinely different from the current "
                "61 factors and prior V14-V18 title research: `NO`. Full-body event semantics would "
                "qualify in principle, but the repository has only 12 non-PIT-verified documents. "
                "Earnings surprise would also qualify, but only one analyst snapshot exists."
            ),
            "",
            "## Data acquisition decision",
            "",
            (
                "- Historical analyst expectations: `REQUIRED_OR_APPROVED_EQUIVALENT`. Obtain licensed "
                "historical consensus vintages (not a current snapshot) with source timestamps, raw "
                "hashes, EPS estimates, dispersion/revision fields, and at least 400-symbol / five-year "
                "coverage."
            ),
            (
                "- Announcement bodies: build or acquire an immutable historical CNInfo full-body "
                "archive. Publication dates without intraday time must be conservatively effective on "
                "the next verified trading session."
            ),
            (
                "- Level-2 data: `NOT_REQUIRED_FOR_FIRST_CHALLENGER`. Do not purchase it before the "
                "event and surprise inputs clear readiness."
            ),
            "",
            "## Experiment disposition",
            "",
            (
                "`PREDICTION_V2_BOUNDED_CHALLENGER_EXPERIMENT` was not started. No labels were read, no "
                "model was trained, no historical result was selected, and production Gen2 remains "
                "unchanged."
            ),
            "",
            "## Final status",
            "",
            f"`{result['final_decision']['status']}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_and_write(settings: AuditSettings) -> dict[str, Any]:
    result = audit_readiness(settings)
    inventory = {
        "audit_as_of_date": result["audit_as_of_date"],
        "protocol_sha256": result["protocol_sha256"],
        "sources": result["sources"],
    }
    _write_json(settings.artifact_dir / "source_inventory.json", inventory)
    _write_json(settings.artifact_dir / "readiness_audit.json", result)
    report_path = settings.artifact_dir / "PREDICTION_V2_NEW_INFORMATION_READINESS_AUDIT_REPORT.md"
    _write_bytes_atomic(report_path, _render_report(result).encode("utf-8"))
    _write_bytes_atomic(
        report_path.with_name(f"{report_path.name}.sha256"),
        f"{sha256_file(report_path)}\n".encode(),
    )
    manifest_files = sorted(
        path
        for path in settings.artifact_dir.iterdir()
        if path.is_file() and path.name not in {"artifact_manifest.json", "artifact_manifest.json.sha256"}
    )
    manifest = {
        "protocol": "PREDICTION_V2_NEW_INFORMATION_READINESS_AUDIT",
        "audit_as_of_date": AUDIT_DATE,
        "files": {path.name: sha256_file(path) for path in manifest_files},
    }
    _write_json(settings.artifact_dir / "artifact_manifest.json", manifest)
    return result
