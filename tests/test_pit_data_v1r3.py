from datetime import datetime, timezone

import pytest

from pit_data_v1r3.core import normalize_exact_duplicate_expectations


def _record(eps: float = 1.0) -> dict:
    return {
        "SECURITY_CODE": "000001",
        "SECURITY_NAME_ABBR": "A",
        "YEAR1": 2026,
        "EPS1": eps,
        "YEAR2": 2027,
        "EPS2": 1.2,
    }


def _page(raw: bytes, records: list[dict]):
    return raw, {"result": {"data": records}}


def test_exact_cross_page_duplicate_is_admitted_with_all_provenance():
    record = _record()
    frame, audit = normalize_exact_duplicate_expectations(
        [_page(b"page-one", [record]), _page(b"page-two", [dict(record)])],
        {"000001"},
        datetime(2026, 8, 30, tzinfo=timezone.utc),
    )
    assert len(frame) == 1
    assert frame.loc[0, "raw_duplicate_count"] == 1
    assert frame.loc[0, "raw_page_count"] == 2
    assert len(frame.loc[0, "raw_page_sha256"].split(";")) == 2
    assert audit["duplicate_symbols"] == 1
    assert audit["duplicate_rows_removed"] == 1


def test_conflicting_duplicate_fails_closed():
    with pytest.raises(ValueError, match="conflicting expectation records"):
        normalize_exact_duplicate_expectations(
            [_page(b"page-one", [_record(1.0)]), _page(b"page-two", [_record(1.1)])],
            {"000001"},
            datetime(2026, 8, 30, tzinfo=timezone.utc),
        )


def test_empty_watchlist_intersection_fails_closed():
    with pytest.raises(ValueError, match="no PIT-watchlist intersection"):
        normalize_exact_duplicate_expectations(
            [_page(b"page-one", [_record()])],
            {"000002"},
            datetime(2026, 8, 30, tzinfo=timezone.utc),
        )


def test_admission_does_not_promote_readiness():
    from pit_data_v1r3.core import AdmissionSettings

    settings = AdmissionSettings()
    assert settings.minimum_training_observations == 20
