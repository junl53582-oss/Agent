import json
from datetime import datetime

import pytest

from announcement_body_v5r1.source import fetch_symbol, fetch_watchlist, load_watchlist, query_payload


def ms(value):
    return int(datetime.fromisoformat(value).timestamp() * 1000)


def item(symbol, announcement_id):
    return {"secCode": symbol, "announcementId": announcement_id, "orgId": "org",
            "announcementTitle": "公告", "announcementTime": ms("2026-08-29T12:00:00+08:00")}


def page(records, total=None):
    body = {"totalAnnouncement": len(records) if total is None else total, "announcements": records}
    return json.dumps(body).encode(), body


def test_watchlist_uses_latest_snapshot_and_requires_300(tmp_path):
    path = tmp_path / "membership.csv"
    rows = ["snapshot_date,index_code,symbol,weight,source"]
    for value in range(300):
        rows.append(f"2026-06-30,000300,{value:06d},0.1,test")
    rows.append("2025-12-31,000300,999999,0.1,test")
    path.write_text("\n".join(rows), encoding="utf-8")
    snapshot, symbols = load_watchlist(path, "2026-08-29")
    assert snapshot == "2026-06-30" and len(symbols) == 300 and "999999" not in symbols


def test_symbol_query_contains_exact_frozen_identity():
    payload = query_payload("2026-08-29", 1, "000001", "gssz0000001")
    assert payload["stock"] == "000001,gssz0000001"
    assert payload["seDate"] == "2026-08-29~2026-08-29"
    assert payload["category"] == "" and payload["searchkey"] == ""


def test_per_symbol_pagination_requires_exact_total_unique_symbol_and_date():
    records = [item("000001", f"1226000{x:03d}") for x in range(31)]
    def fetcher(target_date, page_number, symbol, org_id):
        return page(records[:30], total=31) if page_number == 1 else page(records[30:], total=31)
    raws, manifest = fetch_symbol("2026-08-29", "000001", "org", page_fetcher=fetcher)
    assert len(raws) == 2 and manifest["reported_total"] == 31
    def repeated(target_date, page_number, symbol, org_id):
        return page(records[:30], total=31)
    with pytest.raises(ValueError, match="pagination mismatch"):
        fetch_symbol("2026-08-29", "000001", "org", page_fetcher=repeated)


def test_watchlist_aggregation_is_sorted_and_complete():
    symbols = ["000002", "000001"]
    orgs = {symbol: "org" for symbol in symbols}
    def fetcher(target_date, symbol, org_id):
        raw, _ = page([item(symbol, "1226" + symbol)], total=1)
        return [raw], {"symbol": symbol, "org_id": org_id, "reported_total": 1, "pages": 1}
    raws, manifest = fetch_watchlist("2026-08-29", symbols, orgs, workers=2, symbol_fetcher=fetcher)
    assert manifest["reported_total"] == 2 and manifest["returned_rows"] == 2
    assert [partition["symbol"] for partition in manifest["partitions"]] == sorted(symbols)

