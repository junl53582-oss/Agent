from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, build_opener

from announcement_body_v5.source import ENDPOINT, MAX_PAGE_BYTES, MAX_PAGES, OfficialRedirect
from announcement_body.core import SHANGHAI


def load_watchlist(membership_path, target_date):
    rows = []
    with Path(membership_path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["snapshot_date"] <= target_date:
                rows.append(row)
    if not rows:
        raise ValueError("no frozen membership snapshot on or before target date")
    snapshot = max(row["snapshot_date"] for row in rows)
    symbols = sorted({row["symbol"].zfill(6) for row in rows if row["snapshot_date"] == snapshot})
    if len(symbols) != 300:
        raise ValueError(f"frozen watchlist must contain exactly 300 unique securities, got {len(symbols)}")
    return snapshot, symbols


def load_org_ids(metadata_path, symbols):
    wanted, latest = set(symbols), {}
    with Path(metadata_path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            symbol = row["symbol"].zfill(6)
            if symbol not in wanted or not row["org_id"]:
                continue
            key = row["announcement_date"]
            if symbol not in latest or key > latest[symbol][0]:
                latest[symbol] = (key, row["org_id"])
    missing = sorted(wanted - set(latest))
    if missing:
        raise ValueError(f"missing frozen CNINFO org ids: {missing[:10]}")
    return {symbol: latest[symbol][1] for symbol in sorted(wanted)}


def query_payload(target_date, page, symbol, org_id):
    return {"pageNum": str(page), "pageSize": "30", "column": "szse", "tabName": "fulltext",
            "plate": "", "stock": f"{symbol},{org_id}", "searchkey": "", "secid": "", "category": "", "trade": "",
            "seDate": f"{target_date}~{target_date}", "sortName": "", "sortType": "", "isHLtitle": "true"}


def fetch_page(target_date, page, symbol, org_id, *, opener=None):
    opener = opener or build_opener(OfficialRedirect())
    data = urlencode(query_payload(target_date, page, symbol, org_id)).encode("ascii")
    request = Request(ENDPOINT, data=data, method="POST",
                      headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/",
                               "Content-Type": "application/x-www-form-urlencoded"})
    with opener.open(request, timeout=30) as response:
        final = urlsplit(response.geturl())
        if final.scheme != "https" or final.hostname != "www.cninfo.com.cn" or final.path != "/new/hisAnnouncement/query":
            raise ValueError("response came from unapproved endpoint")
        if "json" not in response.headers.get("Content-Type", "").lower():
            raise ValueError("official query did not return JSON")
        raw = response.read(MAX_PAGE_BYTES + 1)
        if len(raw) > MAX_PAGE_BYTES:
            raise ValueError("official response exceeds page limit")
        body = json.loads(raw)
        if not isinstance(body.get("announcements"), list):
            raise ValueError("official response shape invalid")
        return raw, body


def fetch_symbol(target_date, symbol, org_id, *, page_fetcher=fetch_page):
    first_raw, first = page_fetcher(target_date, 1, symbol, org_id)
    total = int(first.get("totalAnnouncement") or 0)
    pages = max(1, (total + 29) // 30)
    if pages > MAX_PAGES:
        raise ValueError(f"per-security result exceeds page cap for {symbol}")
    raws, seen_ids = [first_raw], []
    for item in first["announcements"]:
        if str(item.get("secCode") or "").zfill(6) != symbol:
            raise ValueError(f"partition returned another security for {symbol}")
        published = datetime.fromtimestamp(int(item["announcementTime"]) / 1000, timezone.utc).astimezone(SHANGHAI).date().isoformat()
        if published != target_date:
            raise ValueError(f"partition returned another publication date for {symbol}: {published}")
        seen_ids.append(str(item.get("announcementId")))
    for page in range(2, pages + 1):
        raw, body = page_fetcher(target_date, page, symbol, org_id)
        raws.append(raw)
        for item in body["announcements"]:
            if str(item.get("secCode") or "").zfill(6) != symbol:
                raise ValueError(f"partition returned another security for {symbol}")
            published = datetime.fromtimestamp(int(item["announcementTime"]) / 1000, timezone.utc).astimezone(SHANGHAI).date().isoformat()
            if published != target_date:
                raise ValueError(f"partition returned another publication date for {symbol}: {published}")
            seen_ids.append(str(item.get("announcementId")))
    if len(seen_ids) != total or len(set(seen_ids)) != total:
        raise ValueError(f"per-security pagination mismatch for {symbol}: total={total}, rows={len(seen_ids)}, unique={len(set(seen_ids))}")
    return raws, {"symbol": symbol, "org_id": org_id, "reported_total": total, "pages": pages}


def fetch_watchlist(target_date, symbols, org_ids, *, workers=4, symbol_fetcher=fetch_symbol):
    if workers < 1 or workers > 4:
        raise ValueError("workers must stay between 1 and 4")
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(symbol_fetcher, target_date, symbol, org_ids[symbol]): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            results[symbol] = future.result()
    raw_pages, partitions = [], []
    for symbol in sorted(results):
        raws, manifest = results[symbol]
        raw_pages.extend(raws)
        partitions.append(manifest)
    total = sum(item["reported_total"] for item in partitions)
    returned = sum(len(json.loads(raw)["announcements"]) for raw in raw_pages)
    if returned != total:
        raise ValueError(f"partition aggregate mismatch: total={total}, rows={returned}")
    return raw_pages, {"endpoint": ENDPOINT, "target_date": target_date, "partition": "frozen_csi300_symbol_org_id",
                       "watchlist_size": len(symbols), "workers": workers, "automatic_retries": 0,
                       "reported_total": total, "returned_rows": returned, "partitions": partitions}
