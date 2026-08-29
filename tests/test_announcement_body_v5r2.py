import pytest

from announcement_body_v5r2.source import fetch_watchlist, normalize_body


def test_null_is_empty_only_when_total_is_zero():
    normalized = normalize_body({"totalAnnouncement": 0, "announcements": None})
    assert normalized["announcements"] == []
    with pytest.raises(ValueError, match="shape invalid"):
        normalize_body({"totalAnnouncement": 1, "announcements": None})
    with pytest.raises(ValueError, match="shape invalid"):
        normalize_body({"totalAnnouncement": 0, "announcements": {}})


def test_sequential_watchlist_keeps_sorted_partitions_and_zero_results():
    symbols = ["000002", "000001"]
    orgs = {symbol: "org" for symbol in symbols}
    def fetcher(target_date, symbol, org_id):
        raw = b'{"totalAnnouncement":0,"announcements":[]}'
        return [raw], {"symbol": symbol, "org_id": org_id, "reported_total": 0, "pages": 1}
    raws, manifest = fetch_watchlist("2026-08-29", symbols, orgs, symbol_fetcher=fetcher)
    assert len(raws) == 2
    assert manifest["workers"] == 1 and manifest["automatic_retries"] == 0
    assert [item["symbol"] for item in manifest["partitions"]] == sorted(symbols)

