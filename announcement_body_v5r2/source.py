from __future__ import annotations

import json
import time
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, build_opener

from announcement_body_v5.source import ENDPOINT, MAX_PAGE_BYTES, OfficialRedirect
from announcement_body_v5r1 import source as parent


def normalize_body(body):
    total = int(body.get("totalAnnouncement") or 0)
    announcements = body.get("announcements")
    if announcements is None and total == 0:
        body = dict(body)
        body["announcements"] = []
        return body
    if not isinstance(announcements, list):
        raise ValueError("official response shape invalid")
    return body


def fetch_page(target_date, page, symbol, org_id, *, opener=None, delay_seconds=0.2):
    if delay_seconds < 0.1:
        raise ValueError("frozen throttle cannot be disabled")
    time.sleep(delay_seconds)
    opener = opener or build_opener(OfficialRedirect())
    data = urlencode(parent.query_payload(target_date, page, symbol, org_id)).encode("ascii")
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
        original = json.loads(raw)
        body = normalize_body(original)
        normalized_raw = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8") if body is not original else raw
        return normalized_raw, body


def fetch_symbol(target_date, symbol, org_id, *, page_fetcher=fetch_page):
    return parent.fetch_symbol(target_date, symbol, org_id, page_fetcher=page_fetcher)


def fetch_watchlist(target_date, symbols, org_ids, *, symbol_fetcher=fetch_symbol):
    raw_pages, partitions = [], []
    for symbol in sorted(symbols):
        raws, manifest = symbol_fetcher(target_date, symbol, org_ids[symbol])
        raw_pages.extend(raws)
        partitions.append(manifest)
    total = sum(item["reported_total"] for item in partitions)
    returned = sum(len(json.loads(raw)["announcements"]) for raw in raw_pages)
    if returned != total:
        raise ValueError(f"partition aggregate mismatch: total={total}, rows={returned}")
    return raw_pages, {"endpoint": ENDPOINT, "target_date": target_date, "partition": "frozen_csi300_symbol_org_id",
                       "watchlist_size": len(symbols), "workers": 1, "inter_request_delay_seconds": 0.2,
                       "automatic_retries": 0, "reported_total": total, "returned_rows": returned,
                       "partitions": partitions}


load_watchlist = parent.load_watchlist
load_org_ids = parent.load_org_ids

