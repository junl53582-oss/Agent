from __future__ import annotations

import json
from pathlib import Path

from announcement_body.core import fetch_document, verify_cached


def select_prospective_events(ledger_root):
    selected = {}
    for path in sorted((Path(ledger_root) / "events").glob("*/*.json")):
        event = json.loads(path.read_text(encoding="utf-8"))
        if not event.get("prospective_metadata_eligible"):
            continue
        if event.get("status") != "new" or not event.get("published_after_v5_freeze") or event.get("lineage_quarantined"):
            raise ValueError("prospective eligibility contradicts frozen event fields")
        announcement_id = str(event["announcement_id"])
        identity = event["identity_sha256"]
        if announcement_id in selected and selected[announcement_id]["identity_sha256"] != identity:
            raise ValueError("eligible event identity changed; quarantine lineage")
        selected[announcement_id] = event
    return [selected[key] for key in sorted(selected)]


def source_record(event):
    return {"symbol": event["symbol"], "announcement_id": event["announcement_id"], "org_id": event["org_id"],
            "title": event["title"], "announcement_date": event["published_at_source"][:10]}


def ingest_events(events, output_root, *, document_fetcher=fetch_document):
    output_root = Path(output_root)
    results = []
    for event in events:
        folder = output_root / f"{event['symbol']}_{event['announcement_id']}"
        if (folder / "failure.json").exists() and not (folder / "receipt.json").exists():
            results.append({"announcement_id": event["announcement_id"], "status": "prior_failure_quarantined"})
            continue
        receipt = document_fetcher(source_record(event), output_root)
        if (folder / "receipt.json").exists():
            verify_cached(folder)
        status = "text_extracted" if receipt["body_extraction_passed"] else "scan_or_text_quality_quarantined"
        results.append({"announcement_id": event["announcement_id"], "status": status,
                        "first_seen_at": event["first_seen_at"], "identity_sha256": event["identity_sha256"],
                        "body_extraction_passed": bool(receipt["body_extraction_passed"])})
    return results


def body_gate(results):
    approved = sum(item.get("status") == "text_extracted" for item in results)
    quarantined = len(results) - approved
    return {"eligible_events": len(results), "text_extracted": approved, "quarantined": quarantined,
            "body_training_approved": False, "model_training_ready": False,
            "replacement_approved": False, "execution_authorized": False}

