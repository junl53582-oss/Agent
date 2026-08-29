import csv

from announcement_body_v4.audit import audit_source, build_report, categories_for_title, revision_type


def test_title_classification_is_fixed_and_revision_is_separate():
    assert categories_for_title("2024年度业绩预告修正公告") == ("earnings_forecast",)
    assert revision_type("2024年度业绩预告修正公告") == "correction"
    assert categories_for_title("关于回购股份的进展公告") == ("repurchase",)
    assert revision_type("关于回购股份的进展公告") is None


def test_source_audit_detects_conflicting_ids_and_missing_lineage(tmp_path):
    path = tmp_path / "source.csv"
    rows = [
        {"symbol": "000001", "announcement_id": "12345678", "announcement_date": "2020-01-02", "title": "2020年度业绩预告", "org_id": "o1"},
        {"symbol": "000001", "announcement_id": "12345679", "announcement_date": "2020-01-03", "title": "2020年度业绩预告修正公告", "org_id": "o1"},
        {"symbol": "000002", "announcement_id": "12345678", "announcement_date": "2020-01-04", "title": "关于回购股份公告", "org_id": "o2"},
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    audit = audit_source(path)
    assert audit["rows"] == 3
    assert audit["conflicting_announcement_ids"] == 1
    assert audit["revision_titles_without_explicit_lineage"] == 1
    assert audit["source_has_historical_first_seen_field"] is False


def test_report_cannot_pass_without_first_seen_and_revision_lineage():
    source = {"source_sha256": "abc", "rows": 10, "invalid_required_fields": 0,
              "conflicting_announcement_ids": 0, "empty_year_category_cells": [],
              "source_has_historical_first_seen_field": False,
              "revision_titles_without_explicit_lineage": 2}
    pilot = {"document_count": 12, "identity_matches": 12, "problems": [],
             "historical_first_seen_proven_count": 0, "revision_documents": 1,
             "revision_documents_with_explicit_lineage": 0}
    report = build_report(source, pilot, "abc", 10)
    assert report["gates"]["source_identity_pass"] is True
    assert report["gates"]["pilot_identity_pass"] is True
    assert report["historical_pit_verified"] is False
    assert report["model_training_ready"] is False

