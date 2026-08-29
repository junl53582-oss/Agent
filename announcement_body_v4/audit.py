from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from announcement_body.core import CATEGORIES, clean_title, sha_file, validate_detail


TARGET_YEARS = tuple(range(2015, 2026))
REVISION_MARKERS = {
    "correction": ("修正", "更正", "纠正"),
    "supplement": ("补充",),
    "cancellation": ("取消", "撤销", "撤回", "作废"),
    "termination": ("终止",),
}
REQUIRED_COLUMNS = {"symbol", "announcement_id", "announcement_date", "title", "org_id"}


def categories_for_title(title):
    return tuple(name for name, terms in CATEGORIES.items() if any(term in title for term in terms))


def revision_type(title):
    return next((kind for kind, terms in REVISION_MARKERS.items() if any(term in title for term in terms)), None)


def _valid_date(value):
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def audit_source(path):
    path = Path(path)
    total = invalid = conflicting = duplicate_rows = 0
    min_date = max_date = None
    date_has_non_midnight_time = 0
    counts = Counter()
    securities = defaultdict(set)
    revision_counts = Counter()
    revision_without_explicit_lineage = 0
    seen = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing_columns = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(f"missing source columns: {missing_columns}")
        for row in reader:
            total += 1
            symbol, announcement_id = row["symbol"], row["announcement_id"]
            day = _valid_date(row["announcement_date"])
            title, org_id = row["title"].strip(), row["org_id"].strip()
            if not re.fullmatch(r"\d{6}", symbol) or not re.fullmatch(r"\d{8,15}", announcement_id) or day is None or not title or not org_id:
                invalid += 1
                continue
            date_value = day.date().isoformat()
            min_date = date_value if min_date is None or date_value < min_date else min_date
            max_date = date_value if max_date is None or date_value > max_date else max_date
            if day.time().isoformat() != "00:00:00":
                date_has_non_midnight_time += 1
            identity = (symbol, date_value, clean_title(title), org_id)
            key = announcement_id
            if key in seen:
                duplicate_rows += 1
                if seen[key] != identity:
                    conflicting += 1
            else:
                seen[key] = identity
            year = day.year
            if year in TARGET_YEARS:
                securities[year].add(symbol)
                for category in categories_for_title(title):
                    counts[(year, category)] += 1
            kind = revision_type(title)
            if kind:
                revision_counts[kind] += 1
                # The frozen CSV has no explicit parent/related-announcement id column.
                revision_without_explicit_lineage += 1
    coverage = {str(year): {category: counts[(year, category)] for category in CATEGORIES} for year in TARGET_YEARS}
    empty_cells = [[year, category] for year in TARGET_YEARS for category in CATEGORIES if counts[(year, category)] == 0]
    return {"source_path": path.as_posix(), "source_sha256": sha_file(path), "rows": total,
            "date_min": min_date, "date_max": max_date, "invalid_required_fields": invalid,
            "duplicate_announcement_id_rows": duplicate_rows, "conflicting_announcement_ids": conflicting,
            "date_rows_with_non_midnight_time": date_has_non_midnight_time,
            "publication_time_granularity": "day_only" if date_has_non_midnight_time == 0 else "mixed",
            "coverage_2015_2025": coverage,
            "unique_securities_by_year": {str(year): len(securities[year]) for year in TARGET_YEARS},
            "empty_year_category_cells": empty_cells, "revision_titles": dict(revision_counts),
            "revision_titles_without_explicit_lineage": revision_without_explicit_lineage,
            "source_has_historical_first_seen_field": False,
            "source_has_explicit_revision_parent_field": False}


def audit_pilot(root, selection_path):
    root, selection_path = Path(root), Path(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    records, problems = [], []
    for selected in selection["records"]:
        document_id = selected["symbol"] + "_" + selected["announcement_id"]
        folder = root / document_id
        receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
        detail = json.loads((folder / "detail.json").read_text(encoding="utf-8"))
        try:
            pdf_url, published_at = validate_detail(selected, detail)
            identity_ok = (receipt["symbol"] == selected["symbol"] and receipt["announcement_id"] == selected["announcement_id"]
                           and clean_title(receipt["title"]) == clean_title(selected["title"])
                           and receipt["published_at_source"] == published_at
                           and Path(pdf_url).stem == selected["announcement_id"])
        except Exception as error:
            identity_ok, published_at, pdf_url = False, None, None
            problems.append({"document_id": document_id, "problem": str(error)})
        retrieved = datetime.fromisoformat(receipt["first_seen_at_utc"])
        published = datetime.fromisoformat(receipt["published_at_source"])
        historical_first_seen_proven = retrieved <= published
        announcement = detail.get("announcement") or {}
        relation = announcement.get("associateAnnouncement")
        records.append({"document_id": document_id, "identity_ok": identity_ok,
                        "published_at_source": published_at, "pdf_url": pdf_url,
                        "retrieval_first_seen_at_utc": receipt["first_seen_at_utc"],
                        "retrieved_after_publication": retrieved > published,
                        "historical_first_seen_proven": historical_first_seen_proven,
                        "explicit_associate_announcement": bool(relation),
                        "revision_type": revision_type(selected["title"]),
                        "body_extraction_passed": receipt["body_extraction_passed"]})
    return {"documents": records, "document_count": len(records),
            "identity_matches": sum(item["identity_ok"] for item in records),
            "historical_first_seen_proven_count": sum(item["historical_first_seen_proven"] for item in records),
            "revision_documents": sum(item["revision_type"] is not None for item in records),
            "revision_documents_with_explicit_lineage": sum(item["revision_type"] is not None and item["explicit_associate_announcement"] for item in records),
            "problems": problems}


def build_report(source, pilot, expected_source_sha256, expected_rows):
    source_identity_pass = (source["source_sha256"] == expected_source_sha256 and source["rows"] == expected_rows
                            and source["invalid_required_fields"] == 0 and source["conflicting_announcement_ids"] == 0)
    pilot_identity_pass = pilot["document_count"] == 12 and pilot["identity_matches"] == 12 and not pilot["problems"]
    coverage_pass = not source["empty_year_category_cells"]
    historical_first_seen_proven = source["source_has_historical_first_seen_field"] and pilot["historical_first_seen_proven_count"] == 12
    revision_lineage_verified = (source["revision_titles_without_explicit_lineage"] == 0
                                 and pilot["revision_documents"] == pilot["revision_documents_with_explicit_lineage"])
    historical_pit_verified = all((source_identity_pass, pilot_identity_pass, coverage_pass,
                                   historical_first_seen_proven, revision_lineage_verified))
    return {"status": "historical_pit_verified" if historical_pit_verified else "historical_pit_not_verified",
            "gates": {"source_identity_pass": source_identity_pass, "pilot_identity_pass": pilot_identity_pass,
                      "coverage_pass": coverage_pass, "historical_first_seen_proven": historical_first_seen_proven,
                      "revision_lineage_verified": revision_lineage_verified},
            "source": source, "pilot": pilot, "historical_pit_verified": historical_pit_verified,
            "model_training_ready": False, "replacement_approved": False, "execution_authorized": False,
            "decision": "quarantine_from_training_until_first_seen_and_revision_lineage_are_defensible"}

