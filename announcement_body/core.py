import csv
import hashlib
import html
import json
import re
import unicodedata
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


SHANGHAI = timezone(timedelta(hours=8))
YEARS = (2018, 2021, 2024)
CATEGORIES = {
    "earnings_forecast": ("业绩预告",),
    "earnings_flash": ("业绩快报",),
    "contract": ("重大合同", "中标", "重大经营合同"),
    "repurchase": ("回购",),
}
MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PAGES = 40


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def validate_record(record):
    if not re.fullmatch(r"\d{6}", record["symbol"]) or not re.fullmatch(r"\d{8,15}", record["announcement_id"]):
        raise ValueError("invalid security/document identifier")
    datetime.fromisoformat(record["announcement_date"])
    if not record["title"].strip():
        raise ValueError("missing title")


def select_pilot(source):
    """One minimum SHA256 id per year/category, independent of prices/returns."""
    selected = {}
    count = 0
    with Path(source).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            count += 1
            year = int(row["announcement_date"][:4])
            if year not in YEARS:
                continue
            for category, terms in CATEGORIES.items():
                if not any(term in row["title"] for term in terms):
                    continue
                key = (year, category)
                rank = sha_bytes((row["symbol"] + ":" + row["announcement_id"]).encode())
                if key not in selected or rank < selected[key][0]:
                    validate_record(row)
                    selected[key] = (rank, {**row, "selection_year": year, "selection_category": category})
    if len(selected) != len(YEARS) * len(CATEGORIES):
        raise ValueError("pilot strata missing; do not silently replace years/categories")
    records = [selected[key][1] for key in sorted(selected)]
    keys = [(record["symbol"], record["announcement_id"]) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("pilot categories selected duplicate documents; revise selection explicitly")
    return records, count


def clean_title(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", html.unescape(re.sub(r"<[^>]*>", "", text))))


def allowed_url(url):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in {"www.cninfo.com.cn", "static.cninfo.com.cn"}:
        raise ValueError("only official HTTPS announcement hosts allowed")
    if parsed.username or parsed.password or parsed.port not in (None, 443) or parsed.fragment:
        raise ValueError("invalid announcement URL")
    if parsed.hostname == "www.cninfo.com.cn" and parsed.path != "/new/announcement/bulletin_detail":
        raise ValueError("only read-only bulletin-detail endpoint allowed")
    if parsed.hostname == "static.cninfo.com.cn" and not re.fullmatch(r"/finalpage/\d{4}-\d{2}-\d{2}/\d{8,15}\.PDF", parsed.path):
        raise ValueError("invalid official PDF path")
    return url


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        allowed_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class AccessLimited(RuntimeError):
    pass


def fetch_bytes(url, *, post=False, limit=2 * 1024 * 1024):
    allowed_url(url)
    request = Request(url, data=b"" if post else None, method="POST" if post else "GET",
                      headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"})
    try:
        with build_opener(SafeRedirect()).open(request, timeout=20) as response:
            allowed_url(response.geturl())
            length = response.headers.get("Content-Length")
            if length and int(length) > limit:
                raise ValueError("response exceeds size limit")
            raw = response.read(limit + 1)
            if len(raw) > limit:
                raise ValueError("response exceeds size limit")
            return raw, {"url": response.geturl(), "status": response.status,
                         "content_type": response.headers.get("Content-Type", "")}
    except HTTPError as error:
        if error.code in (401, 403, 429):
            raise AccessLimited(f"source access/rate limit {error.code}; stop, no bypass or automatic retry") from error
        raise


def detail_request_url(record):
    validate_record(record)
    flag = "true" if record["symbol"].startswith(("0", "3")) else "false"
    return "https://www.cninfo.com.cn/new/announcement/bulletin_detail?" + urlencode(
        {"announceId": record["announcement_id"], "flag": flag, "announceTime": ""})


def validate_detail(record, body):
    announcement = body["announcement"]
    if str(announcement["announcementId"]) != record["announcement_id"]:
        raise ValueError("document id mismatch")
    symbols = re.split(r"[,;\s]+", announcement["secCode"])
    if record["symbol"] not in symbols or announcement["orgId"] != record["org_id"]:
        raise ValueError("issuer mismatch")
    if clean_title(announcement["announcementTitle"]) != clean_title(record["title"]):
        raise ValueError("title changed or wrong document; quarantine")
    published = datetime.fromtimestamp(int(announcement["announcementTime"]) / 1000, timezone.utc).astimezone(SHANGHAI)
    if published.date().isoformat() != record["announcement_date"][:10]:
        raise ValueError("publication date mismatch")
    adjunct = announcement["adjunctUrl"]
    url = allowed_url("https://static.cninfo.com.cn/" + adjunct)
    if Path(urlsplit(url).path).stem != record["announcement_id"] or announcement["adjunctType"].upper() != "PDF":
        raise ValueError("PDF identity/type mismatch")
    return url, published.isoformat()


def numeric_mentions(text, page_number):
    """Literal candidate mentions; deliberately no automatic current/prior-year binding."""
    normalized = unicodedata.normalize("NFKC", text).replace("\u2212", "-")
    result = []
    pattern = r"(?<![\d.,])(?P<number>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(?P<unit>亿元|万元|元|%)"
    factors = {"亿元": Decimal(100000000), "万元": Decimal(10000), "元": Decimal(1), "%": Decimal("0.01")}
    for match in re.finditer(pattern, normalized):
        number = Decimal(match["number"].replace(",", ""))
        result.append({"page": page_number, "text": match[0], "value": str(number * factors[match["unit"]]),
                       "unit": "ratio" if match["unit"] == "%" else "CNY",
                       "context": normalized[max(0, match.start() - 70):match.end() + 70],
                       "start": match.start(), "end": match.end(),
                       "binding": "unresolved_current_prior_or_forecast", "training_approved": False})
    return result


def parse_pdf(path, symbol):
    # Bundled PDF environment only; no modifications to the running model venv.
    import pdfplumber
    pages, mentions = [], []
    with pdfplumber.open(path) as document:
        if not 1 <= len(document.pages) <= MAX_PAGES:
            raise ValueError("document exceeds pilot page limit; never silently truncate")
        for index, page in enumerate(document.pages, start=1):
            text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
            pages.append({"page": index, "text": text, "tables": page.extract_tables()})
            mentions.extend(numeric_mentions(text, index))
    full_text = "\n\n".join(page["text"] for page in pages)
    chinese = len(re.findall(r"[\u3400-\u9fff]", full_text))
    header = re.sub(r"\s+", "", pages[0]["text"][:2500])
    code_present = re.search(r"(?<!\d)" + re.escape(symbol) + r"(?!\d)", header) is not None
    quality = {"text_chars": len(full_text), "chinese_chars": chinese,
               "security_code_in_first_page": code_present,
               "replacement_chars": full_text.count("\ufffd"),
               "all_pages_have_text": all(len(page["text"].strip()) >= 20 for page in pages)}
    accepted = chinese >= 50 and len(full_text) >= 150 and code_present and quality["replacement_chars"] == 0 and quality["all_pages_have_text"]
    return {"pages": pages, "full_text": full_text, "numeric_mentions": mentions, "quality": quality,
            "body_extraction_passed": accepted, "ocr_used": False,
            "numeric_features_ready": False, "historical_pit_verified": False, "model_training_ready": False,
            "reason": "Official archive body reconstructed today; numerical column/period binding and archival-version review still required"}


def next_session_after(day, trading_dates):
    day = datetime.fromisoformat(str(day)[:10]).date() + timedelta(days=1)
    dates = sorted(set(str(value)[:10] for value in trading_dates))
    index = bisect_left(dates, day.isoformat())
    return dates[index] if index < len(dates) else None


def effective_dates(record, observed_at, trading_dates):
    observed = datetime.fromisoformat(observed_at)
    if observed.tzinfo is None:
        raise ValueError("first-seen timestamp must include timezone")
    publication = record["announcement_date"][:10]
    seen_date = observed.astimezone(SHANGHAI).date().isoformat()
    return {"archive_reconstructed_effective_date": next_session_after(publication, trading_dates),
            "prospective_effective_date": next_session_after(max(publication, seen_date), trading_dates),
            "historical_pit_verified": False}


def verify_cached(folder):
    receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
    expected = {"detail.json", "body.pdf", "parsed.json"}
    if set(receipt["sha256"]) != expected:
        raise ValueError("incomplete document receipt")
    for name, digest in receipt["sha256"].items():
        if sha_file(folder / name) != digest:
            raise ValueError("cached document changed; refuse to reuse")
    return receipt


def fetch_document(record, root, fetcher=fetch_bytes, parser=parse_pdf):
    validate_record(record)
    folder = Path(root) / (record["symbol"] + "_" + record["announcement_id"])
    if (folder / "receipt.json").exists():
        receipt = verify_cached(folder)
        validate_detail(record, json.loads((folder / "detail.json").read_bytes()))
        return receipt
    if folder.exists():
        raise ValueError("partial/failed document directory exists; preserve and diagnose, no overwrite")
    folder.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    try:
        detail_url = detail_request_url(record)
        detail_raw, detail_http = fetcher(detail_url, post=True)
        with (folder / "detail.json").open("xb") as stream:
            stream.write(detail_raw)
        pdf_url, published_at = validate_detail(record, json.loads(detail_raw))
        pdf_raw, pdf_http = fetcher(pdf_url, limit=MAX_PDF_BYTES)
        if not pdf_raw.startswith(b"%PDF-"):
            raise ValueError("download is not a PDF; never parse HTML as body")
        with (folder / "body.pdf").open("xb") as stream:
            stream.write(pdf_raw)
        parsed = parser(folder / "body.pdf", record["symbol"])
        write_json_new(folder / "parsed.json", parsed)
        completed = datetime.now(timezone.utc).isoformat()
        receipt = {"symbol": record["symbol"], "announcement_id": record["announcement_id"], "title": record["title"],
                   "announcement_date": record["announcement_date"], "published_at_source": published_at,
                   "retrieval_started_at_utc": started, "first_seen_at_utc": completed,
                   "detail_response": detail_http, "pdf_response": pdf_http,
                   "sha256": {name: sha_file(folder / name) for name in ("detail.json", "body.pdf", "parsed.json")},
                   "body_extraction_passed": parsed["body_extraction_passed"],
                   "historical_pit_verified": False, "model_training_ready": False, "execution_authorized": False}
        write_json_new(folder / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        write_json_new(folder / "failure.json", {"started_at_utc": started, "error": str(error), "automatic_retry": False})
        raise
