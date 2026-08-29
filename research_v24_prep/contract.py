from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal


CURRENT_ROLES = {"current_guidance", "current_preliminary"}
PRIOR_ROLES = {"prior_year_comparator", "period_begin"}


def _midpoint(rows):
    values = [Decimal(str(row["value_base"])) for row in rows]
    lower = [value for value, row in zip(values, rows) if row["bound"] == "lower"]
    upper = [value for value, row in zip(values, rows) if row["bound"] == "upper"]
    points = [value for value, row in zip(values, rows) if row["bound"] == "point"]
    if lower and upper:
        return (min(lower) + max(upper)) / Decimal(2)
    if points:
        return sum(points) / Decimal(len(points))
    return sum(values) / Decimal(len(values)) if values else None


def _relative_delta(current, reference):
    if current is None or reference is None:
        return None
    scale = max(abs(reference), Decimal("1e-12"))
    return float((current - reference) / scale)


def build_document_features(payload):
    if payload.get("gold_binding_passed") is not True:
        raise ValueError("gold binding must pass before feature construction")
    facts = payload.get("facts")
    if not isinstance(facts, list):
        raise ValueError("facts must be a list")
    documents = defaultdict(list)
    for fact in facts:
        if fact.get("binding_approved") is not True:
            raise ValueError("unapproved fact cannot enter features")
        documents[fact["document_id"]].append(fact)
    output = []
    for document_id, rows in sorted(documents.items()):
        grouped = defaultdict(list)
        for row in rows:
            grouped[(row["metric"], row["role"])].append(row)
        revisions, comparisons = [], []
        for metric in sorted({row["metric"] for row in rows}):
            current = _midpoint([row for role in CURRENT_ROLES for row in grouped.get((metric, role), [])])
            previous = _midpoint(grouped.get((metric, "previous_guidance"), []))
            prior = _midpoint([row for role in PRIOR_ROLES for row in grouped.get((metric, role), [])])
            revision = _relative_delta(current, previous)
            comparison = _relative_delta(current, prior)
            if revision is not None and math.isfinite(revision):
                revisions.append(revision)
            if comparison is not None and math.isfinite(comparison):
                comparisons.append(comparison)
        ratios = [float(Decimal(str(row["value_base"]))) for row in rows if row["currency"] == "ratio"]
        feature = {
            "document_id": document_id,
            "fact_count": len(rows),
            "metric_count": len({row["metric"] for row in rows}),
            "uncertain_fact_ratio": sum(row.get("qualifier") is not None for row in rows) / len(rows),
            "ratio_fact_count": len(ratios),
            "mean_reported_ratio": sum(ratios) / len(ratios) if ratios else None,
            "guidance_revision_delta": sum(revisions) / len(revisions) if revisions else None,
            "current_vs_prior_delta": sum(comparisons) / len(comparisons) if comparisons else None,
            "contract_to_revenue": max(
                (float(Decimal(str(row["value_base"]))) for row in rows if row["metric"] == "contract_to_revenue"),
                default=None,
            ),
            "repurchased_ratio": max(
                (float(Decimal(str(row["value_base"]))) for row in rows if row["metric"] == "repurchased_ratio"),
                default=None,
            ),
        }
        output.append(feature)
    return output


def next_session_after(available_at, trading_sessions):
    timestamp = datetime.fromisoformat(available_at)
    if timestamp.tzinfo is None:
        raise ValueError("availability timestamp must include timezone")
    sessions = sorted(date.fromisoformat(value) for value in trading_sessions)
    return next((value.isoformat() for value in sessions if value > timestamp.date()), None)


def label_schedule(available_at, trading_sessions, horizons=(5, 20)):
    sessions = sorted(date.fromisoformat(value) for value in trading_sessions)
    entry = next_session_after(available_at, trading_sessions)
    if entry is None:
        return {"entry_date": None, "label_end_dates": {str(h): None for h in horizons}}
    index = sessions.index(date.fromisoformat(entry))
    ends = {str(horizon): sessions[index + horizon].isoformat() if index + horizon < len(sessions) else None
            for horizon in horizons}
    return {"entry_date": entry, "label_end_dates": ends}


def training_gate(*, prospective_events, approved_bodies, mature_5d, mature_20d):
    checks = {
        "prospective_events_positive": prospective_events > 0,
        "all_events_have_approved_bodies": prospective_events > 0 and approved_bodies == prospective_events,
        "all_5d_labels_mature": prospective_events > 0 and mature_5d == prospective_events,
        "all_20d_labels_mature": prospective_events > 0 and mature_20d == prospective_events,
    }
    return {"checks": checks, "model_training_ready": all(checks.values()),
            "replacement_approved": False, "execution_authorized": False}

