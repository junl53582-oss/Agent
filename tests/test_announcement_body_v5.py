import json
from datetime import datetime, timezone

import pytest

from announcement_body_v5.ledger import parse_source_record, record_observation, summarize_ledger
from announcement_body_v5.source import query_payload


def raw_page(records):
    return json.dumps({"totalAnnouncement": len(records), "announcements": records}, ensure_ascii=False).encode("utf-8")


def record(announcement_id="1226000001", title="关于经营进展的公告", timestamp=None):
    timestamp = timestamp if timestamp is not None else int(datetime.fromisoformat("2026-08-29T13:25:00+08:00").timestamp() * 1000)
    return {"secCode": "000001", "announcementId": announcement_id, "orgId": "gssz0000001",
            "announcementTitle": title, "announcementTime": timestamp,
            "adjunctUrl": f"finalpage/2026-08-29/{announcement_id}.PDF"}


def test_parse_requires_official_identity_and_preserves_source_time():
    item = parse_source_record(record())
    assert item["symbol"] == "000001"
    assert item["announcement_id"] == "1226000001"
    assert item["pdf_url"].startswith("https://static.cninfo.com.cn/")
    broken = record()
    broken["adjunctUrl"] = "finalpage/2026-08-29/9999999999.PDF"
    with pytest.raises(ValueError, match="identity mismatch"):
        parse_source_record(broken)


def test_append_only_idempotency_hash_change_and_no_backdating(tmp_path):
    data, artifacts = tmp_path / "data", tmp_path / "artifacts"
    freeze = "2026-08-29T05:20:00+00:00"
    observed1 = "2026-08-29T13:30:00+08:00"
    first = record_observation(data, artifacts, observed_at=observed1, target_date="2026-08-29",
                               freeze_created_at=freeze, raw_pages=[raw_page([record()])], query={"page": 1})
    assert first["statuses"] == {"new": 1, "repeat_same": 0, "identity_hash_changed": 0}
    assert first["prospective_first_seen_verified"] is True
    assert first["prospective_pit_verified"] is False
    again = record_observation(data, artifacts, observed_at=observed1, target_date="2026-08-29",
                               freeze_created_at=freeze, raw_pages=[raw_page([record()])], query={"page": 1})
    assert again == first
    observed2 = "2026-08-29T13:31:00+08:00"
    repeat = record_observation(data, artifacts, observed_at=observed2, target_date="2026-08-29",
                                freeze_created_at=freeze, raw_pages=[raw_page([record()])], query={"page": 1})
    assert repeat["statuses"]["repeat_same"] == 1
    observed3 = "2026-08-29T13:32:00+08:00"
    changed = record(announcement_id="1226000001", title="关于经营进展的补充公告")
    changed_receipt = record_observation(data, artifacts, observed_at=observed3, target_date="2026-08-29",
                                         freeze_created_at=freeze, raw_pages=[raw_page([changed])], query={"page": 1})
    assert changed_receipt["statuses"]["identity_hash_changed"] == 1
    summary = summarize_ledger(data)
    assert summary["event_records"] == 2
    assert summary["identity_hash_changes"] == 1
    events = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((data / "events" / "1226000001").glob("*.json"))]
    assert events[1]["first_seen_at"] == observed1
    assert events[1]["lineage_quarantined"] is True


def test_pre_freeze_publication_and_revision_never_become_eligible(tmp_path):
    data, artifacts = tmp_path / "data", tmp_path / "artifacts"
    old = record(timestamp=1756425600000)
    receipt = record_observation(data, artifacts, observed_at="2026-08-29T13:30:00+08:00", target_date="2026-08-29",
                                 freeze_created_at="2026-08-29T05:20:00+00:00", raw_pages=[raw_page([old])], query={})
    assert receipt["prospective_new_eligible"] == 0
    event = json.loads(next((data / "events" / "1226000001").glob("*.json")).read_text(encoding="utf-8"))
    assert event["first_seen_at"] == "2026-08-29T13:30:00+08:00"
    assert event["published_after_v5_freeze"] is False
    assert event["historical_pit_verified"] is False


def test_target_date_must_be_actual_observation_date(tmp_path):
    with pytest.raises(ValueError, match="no backfill"):
        record_observation(tmp_path / "data", tmp_path / "artifacts", observed_at="2026-08-29T13:30:00+08:00",
                           target_date="2026-08-28", freeze_created_at="2026-08-29T05:20:00+00:00",
                           raw_pages=[raw_page([])], query={})


def test_query_is_current_day_only_shape_without_hidden_filters():
    payload = query_payload("2026-08-29", 2)
    assert payload["seDate"] == "2026-08-29~2026-08-29"
    assert payload["pageNum"] == "2" and payload["pageSize"] == "30"
    assert payload["stock"] == "" and payload["category"] == "" and payload["searchkey"] == ""
