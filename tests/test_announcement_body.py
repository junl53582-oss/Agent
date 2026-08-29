import copy
import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from announcement_body.core import (CATEGORIES, YEARS, allowed_url, clean_title, effective_dates, fetch_document,
                                    next_session_after, numeric_mentions, parse_pdf, select_pilot, validate_detail)


def record():
    return dict(symbol="000001", announcement_id="1201234567", org_id="gssz0000001",
                announcement_date="2024-01-05 00:00:00", title="2023年度业绩预告")


def detail(row=None):
    row = row or record()
    return {"announcement": dict(announcementId=row["announcement_id"], secCode=row["symbol"], orgId=row["org_id"],
             announcementTitle=row["title"], adjunctType="PDF", adjunctUrl=f"finalpage/2024-01-05/{row['announcement_id']}.PDF",
             announcementTime=int(datetime(2024, 1, 4, 16, tzinfo=timezone.utc).timestamp() * 1000))}


class BodyTests(unittest.TestCase):
    def test_selection_is_order_independent_and_does_not_need_returns(self):
        with tempfile.TemporaryDirectory() as directory:
            files = [Path(directory) / "forward.csv", Path(directory) / "reverse.csv"]
            rows = []
            number = 1200000000
            for year in YEARS:
                for category, terms in CATEGORIES.items():
                    for candidate in range(2):
                        number += 1
                        rows.append({**record(), "announcement_date": f"{year}-01-05 00:00:00",
                                     "announcement_id": str(number), "title": terms[0] + str(candidate)})
            for path, ordered in zip(files, (rows, list(reversed(rows)))):
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(ordered)
            first, count = select_pilot(files[0])
            second, _ = select_pilot(files[1])
            self.assertEqual(first, second)
            self.assertEqual(len(first), 12)
            self.assertEqual(count, 24)

    def test_url_rejects_nonofficial_redirects_and_traversal(self):
        for url in ("http://static.cninfo.com.cn/finalpage/2024-01-05/1201234567.PDF",
                    "https://evil.example/file.PDF", "https://static.cninfo.com.cn/../secret",
                    "https://www.cninfo.com.cn/login", "https://name:password@static.cninfo.com.cn/file.PDF",
                    "https://static.cninfo.com.cn:8888/finalpage/2024-01-05/1201234567.PDF"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                allowed_url(url)

    def test_identity_date_and_title_checks(self):
        url, timestamp = validate_detail(record(), detail())
        self.assertTrue(url.endswith("1201234567.PDF"))
        self.assertTrue(timestamp.startswith("2024-01-05"))
        for key, value in (("announcementId", "1207654321"), ("secCode", "600001"),
                           ("orgId", "wrong"), ("announcementTitle", "changed"),
                           ("announcementTime", 0), ("adjunctUrl", "https://evil.example/x.PDF")):
            data = detail()
            data["announcement"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_detail(record(), data)

    def test_title_normalization_does_not_remove_semantic_changes(self):
        self.assertEqual(clean_title("<em>公告</em>（2024）"), clean_title("公告 (2024)"))
        self.assertNotEqual(clean_title("业绩预告"), clean_title("业绩预告更正公告"))

    def test_prospective_date_cannot_backfill_historical_publication(self):
        calendar = ["2024-01-05", "2024-01-08", "2026-08-28", "2026-08-31", "2026-09-01"]
        result = effective_dates(record(), "2026-08-29T00:00:00+00:00", calendar)
        self.assertEqual(result["archive_reconstructed_effective_date"], "2024-01-08")
        self.assertEqual(result["prospective_effective_date"], "2026-08-31")
        self.assertFalse(result["historical_pit_verified"])
        self.assertIsNone(next_session_after("2026-09-01", calendar))
        with self.assertRaises(ValueError):
            effective_dates(record(), "2026-08-29T00:00:00", calendar)

    def test_currency_percent_preserved_without_unsafe_column_binding(self):
        text = "本期合同1.23亿元，上期-2,000万元，同比增长25.5%，下降−3%。"
        facts = numeric_mentions(text, 2)
        self.assertEqual([Decimal(x["value"]) for x in facts], [Decimal("123000000"), Decimal("-20000000"), Decimal(".255"), Decimal("-.03")])
        self.assertTrue(all(x["page"] == 2 and not x["training_approved"] for x in facts))
        self.assertTrue(all(x["binding"] == "unresolved_current_prior_or_forecast" for x in facts))
        self.assertEqual(numeric_mentions("错误格式1,23万元", 1), [])

    def test_pdf_quality_rejects_scans_or_missing_issuer(self):
        for text in ("", "上市公司业绩预告" * 30):
            document = MagicMock()
            document.__enter__.return_value.pages = [SimpleNamespace(extract_text=lambda **kwargs: text, extract_tables=lambda: [])]
            with patch.dict("sys.modules", {"pdfplumber": SimpleNamespace(open=lambda path: document)}):
                self.assertFalse(parse_pdf("unused.pdf", "000001")["body_extraction_passed"])

    def test_pdf_captures_all_pages_tables_and_keeps_not_training_ready(self):
        text = "证券代码000001 " + "上市公司业绩预告，数据未经审计。" * 15
        document = MagicMock()
        document.__enter__.return_value.pages = [SimpleNamespace(extract_text=lambda **kwargs: text, extract_tables=lambda: [[['本期', '上年同期'], ['100', '90']]])] * 2
        with patch.dict("sys.modules", {"pdfplumber": SimpleNamespace(open=lambda path: document)}):
            result = parse_pdf("unused.pdf", "000001")
        self.assertEqual(len(result["pages"]), 2)
        self.assertTrue(result["body_extraction_passed"])
        self.assertFalse(result["model_training_ready"])
        self.assertFalse(result["numeric_features_ready"])

    def test_pdf_never_silently_truncates_long_document(self):
        page = SimpleNamespace(extract_text=lambda **kwargs: "证券代码000001" * 20, extract_tables=lambda: [])
        document = MagicMock()
        document.__enter__.return_value.pages = [page] * 41
        with patch.dict("sys.modules", {"pdfplumber": SimpleNamespace(open=lambda path: document)}), \
             self.assertRaisesRegex(ValueError, "page limit"):
            parse_pdf("unused.pdf", "000001")

    def test_cache_resumes_without_network_and_detects_tampering(self):
        calls = []
        def fetcher(url, **kwargs):
            calls.append(url)
            raw = json.dumps(detail()).encode() if kwargs.get("post") else b"%PDF-test"
            return raw, {"url": url, "status": 200, "content_type": "test"}
        parser = lambda *args: {"body_extraction_passed": True}
        with tempfile.TemporaryDirectory() as directory:
            first = fetch_document(record(), directory, fetcher, parser)
            second = fetch_document(record(), directory, fetcher, parser)
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 2)
            path = Path(directory) / "000001_1201234567/body.pdf"
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "cached document changed"):
                fetch_document(record(), directory, fetcher, parser)

    def test_html_response_and_partial_state_never_overwritten(self):
        def fetcher(url, **kwargs):
            return (json.dumps(detail()).encode() if kwargs.get("post") else b"<html>access denied</html>"), {}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "not a PDF"):
                fetch_document(record(), directory, fetcher, lambda *a: {})
            folder = Path(directory) / "000001_1201234567"
            previous = (folder / "failure.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "partial/failed"):
                fetch_document(record(), directory, fetcher, lambda *a: {})
            self.assertEqual(previous, (folder / "failure.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
