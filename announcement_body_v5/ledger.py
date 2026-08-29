from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

from announcement_body.core import SHANGHAI, clean_title, write_json_new
from announcement_body_v4.audit import revision_type


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def parse_source_record(raw):
    symbol = str(raw.get("secCode") or "")
    announcement_id = str(raw.get("announcementId") or "")
    org_id = str(raw.get("orgId") or "")
    title = clean_title(str(raw.get("announcementTitle") or ""))
    timestamp = raw.get("announcementTime")
    adjunct = str(raw.get("adjunctUrl") or "")
    if not re.fullmatch(r"\d{6}", symbol) or not re.fullmatch(r"\d{8,15}", announcement_id):
        raise ValueError("invalid announcement identity")
    if not org_id or not title or timestamp is None:
        raise ValueError("missing required announcement metadata")
    published = datetime.fromtimestamp(int(timestamp) / 1000, timezone.utc).astimezone(SHANGHAI)
    if adjunct:
        pdf_url = "https://static.cninfo.com.cn/" + adjunct.lstrip("/")
        parsed = urlsplit(pdf_url)
        if parsed.hostname != "static.cninfo.com.cn" or Path(parsed.path).stem != announcement_id:
            raise ValueError("official PDF identity mismatch")
    else:
        pdf_url = None
    identity = {"symbol": symbol, "announcement_id": announcement_id, "org_id": org_id,
                "title": title, "published_at_source": published.isoformat(), "pdf_url": pdf_url}
    digest = sha_bytes(json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return {**identity, "identity_sha256": digest, "revision_type": revision_type(title)}


def _event_files(data_root, announcement_id):
    return sorted((Path(data_root) / "events" / announcement_id).glob("*.json"))


def prior_events(data_root, announcement_id):
    return [json.loads(path.read_text(encoding="utf-8")) for path in _event_files(data_root, announcement_id)]


def observation_id(observed_at):
    value = datetime.fromisoformat(observed_at)
    if value.tzinfo is None:
        raise ValueError("observed_at must include timezone")
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def record_observation(data_root, artifact_root, *, observed_at, target_date, freeze_created_at, raw_pages, query):
    observed = datetime.fromisoformat(observed_at)
    frozen = datetime.fromisoformat(freeze_created_at)
    if observed.tzinfo is None or frozen.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    shanghai_day = observed.astimezone(SHANGHAI).date().isoformat()
    if target_date != shanghai_day:
        raise ValueError("target date must equal actual Shanghai observation date; no backfill")
    obs_id = observation_id(observed_at)
    data_root, artifact_root = Path(data_root), Path(artifact_root)
    receipt_path = artifact_root / "observations" / f"{obs_id}.json"
    if receipt_path.exists():
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    parsed_pages, raw_hashes = [], []
    for index, raw in enumerate(raw_pages, start=1):
        raw_hashes.append(sha_bytes(raw))
        body = json.loads(raw)
        if not isinstance(body.get("announcements"), list):
            raise ValueError("invalid official response shape")
        parsed_pages.append(body)
    rows = [item for page in parsed_pages for item in page["announcements"]]
    identities = {}
    for raw in rows:
        item = parse_source_record(raw)
        key = item["announcement_id"]
        if key in identities and identities[key]["identity_sha256"] != item["identity_sha256"]:
            raise ValueError("conflicting identity in one observation")
        identities[key] = item
    raw_dir = data_root / "raw" / obs_id
    for index, raw in enumerate(raw_pages, start=1):
        path = raw_dir / f"page_{index:04d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    statuses = {"new": 0, "repeat_same": 0, "identity_hash_changed": 0}
    prospective_new = 0
    for announcement_id, item in sorted(identities.items()):
        previous = prior_events(data_root, announcement_id)
        same = any(event["identity_sha256"] == item["identity_sha256"] for event in previous)
        if same:
            statuses["repeat_same"] += 1
            continue
        status = "identity_hash_changed" if previous else "new"
        statuses[status] += 1
        published = datetime.fromisoformat(item["published_at_source"])
        post_freeze_publication = published.astimezone(timezone.utc) >= frozen.astimezone(timezone.utc)
        lineage_quarantined = item["revision_type"] is not None
        prospective_eligible = status == "new" and post_freeze_publication and not lineage_quarantined
        prospective_new += int(prospective_eligible)
        event = {**item, "observation_id": obs_id, "observed_at": observed.isoformat(),
                 "first_seen_at": observed.isoformat() if not previous else previous[0]["first_seen_at"],
                 "status": status, "published_after_v5_freeze": post_freeze_publication,
                 "prospective_metadata_eligible": prospective_eligible,
                 "lineage_quarantined": lineage_quarantined,
                 "historical_pit_verified": False, "body_training_approved": False,
                 "earliest_effective_date": None,
                 "effective_date_reason": "requires_next_verified_trading_session_after_max_publication_and_first_seen"}
        event_path = data_root / "events" / announcement_id / f"{obs_id}.json"
        write_json_new(event_path, event)
    receipt = {"observation_id": obs_id, "observed_at": observed.isoformat(), "target_date": target_date,
               "query": query, "raw_page_sha256": raw_hashes, "pages": len(raw_pages),
               "records_returned": len(rows), "unique_announcements": len(identities), "statuses": statuses,
               "prospective_new_eligible": prospective_new,
               "prospective_first_seen_verified": prospective_new > 0,
               "prospective_pit_verified": False,
               "historical_pit_verified": False, "model_training_ready": False,
               "execution_authorized": False}
    write_json_new(receipt_path, receipt)
    return receipt


def summarize_ledger(data_root):
    files = sorted((Path(data_root) / "events").glob("*/*.json"))
    events = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    ids = {event["announcement_id"] for event in events}
    prospective = {event["announcement_id"] for event in events if event["prospective_metadata_eligible"]}
    changes = sum(event["status"] == "identity_hash_changed" for event in events)
    return {"event_records": len(events), "announcement_ids": len(ids), "prospective_metadata_eligible_ids": len(prospective),
            "identity_hash_changes": changes, "historical_pit_verified": False,
            "prospective_first_seen_verified": len(prospective) > 0, "prospective_pit_verified": False,
            "model_training_ready": False, "execution_authorized": False}
