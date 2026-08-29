import json

import pytest

from research_v24_body.pipeline import body_gate, ingest_events, select_prospective_events


def event(announcement_id="1226000001", *, eligible=True, changed=False):
    return {"symbol": "000001", "announcement_id": announcement_id, "org_id": "gssz0000001",
            "title": "关于经营进展的公告", "published_at_source": "2026-08-30T08:00:00+08:00",
            "first_seen_at": "2026-08-30T09:00:00+08:00", "status": "new",
            "published_after_v5_freeze": True, "prospective_metadata_eligible": eligible,
            "lineage_quarantined": False, "identity_sha256": "b" * 64 if changed else "a" * 64,
            "pdf_url": f"https://static.cninfo.com.cn/finalpage/2026-08-30/{announcement_id}.PDF"}


def write_event(root, item, suffix="one"):
    path = root / "events" / item["announcement_id"] / f"{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")


def test_selection_only_admits_consistent_prospective_events(tmp_path):
    write_event(tmp_path, event(eligible=False), "old")
    write_event(tmp_path, event("1226000002"), "new")
    selected = select_prospective_events(tmp_path)
    assert [item["announcement_id"] for item in selected] == ["1226000002"]
    broken = event("1226000003")
    broken["lineage_quarantined"] = True
    write_event(tmp_path, broken, "broken")
    with pytest.raises(ValueError, match="contradicts"):
        select_prospective_events(tmp_path)


def test_identity_change_is_quarantined(tmp_path):
    write_event(tmp_path, event(), "first")
    write_event(tmp_path, event(changed=True), "second")
    with pytest.raises(ValueError, match="identity changed"):
        select_prospective_events(tmp_path)


def test_ingestion_uses_only_selected_identity_and_classifies_text(tmp_path):
    calls = []
    def fake_fetch(record, root):
        calls.append(record)
        folder = root / f"{record['symbol']}_{record['announcement_id']}"
        folder.mkdir(parents=True)
        for name, body in (("detail.json", b"{}"), ("body.pdf", b"%PDF-x"), ("parsed.json", b"{}")):
            (folder / name).write_bytes(body)
        import hashlib
        sha = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in ("detail.json", "body.pdf", "parsed.json")}
        (folder / "receipt.json").write_text(json.dumps({"sha256": sha, "body_extraction_passed": True}), encoding="utf-8")
        return {"body_extraction_passed": True, "sha256": sha}
    results = ingest_events([event()], tmp_path, document_fetcher=fake_fetch)
    assert len(calls) == 1 and results[0]["status"] == "text_extracted"
    assert body_gate(results)["body_training_approved"] is False


def test_zero_events_is_successful_noop_but_never_training_ready(tmp_path):
    assert ingest_events([], tmp_path) == []
    gate = body_gate([])
    assert gate["eligible_events"] == 0 and gate["model_training_ready"] is False

