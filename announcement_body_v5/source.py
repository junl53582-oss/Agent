from __future__ import annotations

import json
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


ENDPOINT = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
MAX_PAGE_BYTES = 8 * 1024 * 1024
MAX_PAGES = 500


class OfficialRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlsplit(newurl)
        if parsed.scheme != "https" or parsed.hostname != "www.cninfo.com.cn" or parsed.path != "/new/hisAnnouncement/query":
            raise ValueError("official query redirected outside approved endpoint")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def query_payload(target_date, page):
    return {"pageNum": str(page), "pageSize": "30", "column": "szse", "tabName": "fulltext",
            "plate": "", "stock": "", "searchkey": "", "secid": "", "category": "", "trade": "",
            "seDate": f"{target_date}~{target_date}", "sortName": "", "sortType": "", "isHLtitle": "true"}


def fetch_page(target_date, page, *, opener=None):
    opener = opener or build_opener(OfficialRedirect())
    data = urlencode(query_payload(target_date, page)).encode("ascii")
    request = Request(ENDPOINT, data=data, method="POST",
                      headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/",
                               "Content-Type": "application/x-www-form-urlencoded"})
    with opener.open(request, timeout=30) as response:
        final = urlsplit(response.geturl())
        if final.scheme != "https" or final.hostname != "www.cninfo.com.cn" or final.path != "/new/hisAnnouncement/query":
            raise ValueError("response came from unapproved endpoint")
        content_type = response.headers.get("Content-Type", "")
        if "json" not in content_type.lower():
            raise ValueError("official query did not return JSON")
        raw = response.read(MAX_PAGE_BYTES + 1)
        if len(raw) > MAX_PAGE_BYTES:
            raise ValueError("official response exceeds page limit")
        body = json.loads(raw)
        if not isinstance(body.get("announcements"), list):
            raise ValueError("official response shape invalid")
        return raw, body


def fetch_day(target_date, *, page_fetcher=fetch_page):
    first_raw, first = page_fetcher(target_date, 1)
    total = int(first.get("totalAnnouncement") or 0)
    pages = max(1, (total + 29) // 30)
    if pages > MAX_PAGES:
        raise ValueError("daily result exceeds frozen page limit; preserve failure and revise protocol")
    raw_pages = [first_raw]
    for page in range(2, pages + 1):
        raw, _ = page_fetcher(target_date, page)
        raw_pages.append(raw)
    return raw_pages, {"endpoint": ENDPOINT, "target_date": target_date, "page_size": 30,
                       "reported_total": total, "reported_pages": pages, "automatic_retries": 0}

