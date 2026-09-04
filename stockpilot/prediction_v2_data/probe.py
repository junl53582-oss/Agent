from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

import requests

ENDPOINT = "https://reportapi.eastmoney.com/report/list"
MAX_BYTES = 2_000_000


def probe_eastmoney_report_schema(raw_path: Path, session: requests.Session | None = None) -> dict:
    """Perform one schema-only request. Raw rows remain ignored and are never training inputs."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    requests_made = 0
    sidecar = raw_path.with_name(f"{raw_path.name}.sha256")
    if raw_path.exists():
        raw = raw_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if not sidecar.exists() or sidecar.read_text(encoding="ascii").strip() != digest:
            raise RuntimeError("schema-probe raw response is not bound by a valid SHA256 sidecar")
        source = "IMMUTABLE_LOCAL_PROBE_REUSE"
    else:
        if urlsplit(ENDPOINT).hostname != "reportapi.eastmoney.com":
            raise ValueError("unapproved schema-probe host")
        client = session or requests.Session()
        params = {
            "industryCode": "*",
            "pageSize": "5",
            "beginTime": "2017-01-01",
            "endTime": "2026-09-05",
            "pageNo": "1",
            "qType": "0",
            "code": "000001",
        }
        response = client.get(
            ENDPOINT,
            params=params,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        requests_made = 1
        response.raise_for_status()
        if len(response.content) > MAX_BYTES:
            raise ValueError("schema probe exceeded byte limit")
        raw = response.content
        digest = hashlib.sha256(raw).hexdigest()
        with raw_path.open("xb") as stream:
            stream.write(raw)
        with sidecar.open("x", encoding="ascii") as stream:
            stream.write(f"{digest}\n")
        source = "LIVE_SCHEMA_PROBE"
    payload = json.loads(raw)
    rows = payload.get("data") or []
    fields = sorted({str(key) for row in rows for key in row})
    forecast_fields = sorted(field for field in fields if "predict" in field.lower())
    return {
        "endpoint": ENDPOINT,
        "source": source,
        "network_requests": requests_made,
        "network_requests_total": 1,
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_bytes": len(raw),
        "rows_observed": len(rows),
        "rows_retained_in_committed_artifacts": 0,
        "top_level_fields": sorted(payload),
        "record_fields": fields,
        "forecast_fields": forecast_fields,
        "current_year": payload.get("currentYear"),
        "has_stable_report_id": "infoCode" in fields,
        "has_report_publication_date": "publishDate" in fields,
        "has_explicit_forecast_period_per_value": any(
            name in fields for name in ("forecastPeriod", "forecastYear", "reportPeriod")
        ),
        "has_revision_or_supersession_link": any(
            name in fields for name in ("revisionId", "supersedesId", "revisionStatus")
        ),
        "training_admissible": False,
        "reason": (
            "Dynamic predictThisYear/NextYear fields are tied to a response-level currentYear and the "
            "record schema lacks explicit per-value forecast-period and revision lineage."
        ),
    }
