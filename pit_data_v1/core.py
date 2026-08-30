from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import requests
from zoneinfo import ZoneInfo


SHANGHAI = ZoneInfo("Asia/Shanghai")
EXPECTATION_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
FLOW_URL = "https://push2.eastmoney.com/api/qt/clist/get"
APPROVED_HOSTS = {"datacenter-web.eastmoney.com", "push2.eastmoney.com"}
MAX_PAGE_BYTES = 8_000_000
PAGE_SIZE = 500


@dataclass(frozen=True)
class ObservationSettings:
    version: str = "pit-data-v1"
    data_root: Path = Path("data/pit_observations_v1")
    artifact_root: Path = Path("artifacts/pit_data_v1")
    membership_path: Path = Path("data/universes/000300/history_v10.csv")
    industry_path: Path = Path("data/industry_history_v10.csv")
    minimum_expectation_coverage: float = 0.60
    minimum_flow_coverage: float = 0.95
    minimum_training_observations: int = 20


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: dict) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(raw)


def _approved_url(url: str) -> bool:
    parsed = urlsplit(url)
    return parsed.scheme == "https" and parsed.hostname in APPROVED_HOSTS and not parsed.username


def _get_json(session: requests.Session, url: str, params: dict) -> tuple[bytes, dict]:
    if not _approved_url(url):
        raise ValueError(f"unapproved source URL: {url}")
    response = session.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    if len(response.content) > MAX_PAGE_BYTES:
        raise ValueError("source response exceeds frozen byte limit")
    content_type = response.headers.get("Content-Type", "").lower()
    if "json" not in content_type and "text/plain" not in content_type:
        raise ValueError(f"source did not return JSON: {content_type}")
    body = response.json()
    return response.content, body


def fetch_expectation_pages(session: requests.Session | None = None) -> list[tuple[bytes, dict]]:
    session = session or requests.Session()
    base = {
        "reportName": "RPT_WEB_RESPREDICT",
        "columns": "WEB_RESPREDICT",
        "pageNumber": "1",
        "pageSize": str(PAGE_SIZE),
        "sortTypes": "-1",
        "sortColumns": "RATING_ORG_NUM",
    }
    first = _get_json(session, EXPECTATION_URL, base)
    result = first[1].get("result")
    if not first[1].get("success") or not isinstance(result, dict):
        raise ValueError("expectation response shape invalid")
    pages = int(result.get("pages") or 0)
    output = [first]
    for page in range(2, pages + 1):
        output.append(_get_json(session, EXPECTATION_URL, {**base, "pageNumber": str(page)}))
    expected = int(result.get("count") or 0)
    actual = sum(len(item[1]["result"].get("data") or []) for item in output)
    if actual != expected:
        raise ValueError(f"expectation pagination mismatch: expected={expected}, actual={actual}")
    return output


def fetch_flow_pages(session: requests.Session | None = None) -> list[tuple[bytes, dict]]:
    session = session or requests.Session()
    base = {
        "fid": "f62",
        "po": "1",
        "pz": str(PAGE_SIZE),
        "pn": "1",
        "np": "1",
        "fltt": "2",
        "invt": "2",
        "ut": "b2884a393a59ad64002292a3e90d46a5",
        "fs": "m:0+t:6+f:!2,m:0+t:13+f:!2,m:0+t:80+f:!2,m:1+t:2+f:!2,m:1+t:23+f:!2,m:0+t:7+f:!2,m:1+t:3+f:!2",
        "fields": "f12,f14,f2,f3,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87,f124",
    }
    first = _get_json(session, FLOW_URL, base)
    data = first[1].get("data")
    if not isinstance(data, dict) or not isinstance(data.get("diff"), list):
        raise ValueError("fund-flow response shape invalid")
    total = int(data.get("total") or 0)
    pages = int(math.ceil(total / PAGE_SIZE))
    output = [first]
    for page in range(2, pages + 1):
        output.append(_get_json(session, FLOW_URL, {**base, "pn": str(page)}))
    actual = sum(len(item[1]["data"].get("diff") or []) for item in output)
    if actual != total:
        raise ValueError(f"fund-flow pagination mismatch: expected={total}, actual={actual}")
    return output


def load_watchlist(path: str | Path, as_of: str | pd.Timestamp) -> tuple[str, set[str]]:
    frame = pd.read_csv(path, dtype={"symbol": str})
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"]).dt.normalize()
    cutoff = pd.Timestamp(as_of).normalize()
    eligible = frame[frame["snapshot_date"] <= cutoff]
    if eligible.empty:
        raise ValueError("no PIT membership snapshot is available")
    snapshot = eligible["snapshot_date"].max()
    symbols = set(eligible.loc[eligible["snapshot_date"].eq(snapshot), "symbol"].str.zfill(6))
    if len(symbols) != 300:
        raise ValueError(f"frozen CSI300 snapshot must contain 300 symbols, got {len(symbols)}")
    return str(snapshot.date()), symbols


def _page_rows(pages: list[tuple[bytes, dict]], root: str) -> list[tuple[dict, str]]:
    output: list[tuple[dict, str]] = []
    for raw, body in pages:
        rows = body["result"]["data"] if root == "result" else body["data"]["diff"]
        page_hash = sha256_bytes(raw)
        output.extend((row, page_hash) for row in rows)
    return output


def normalize_expectations(
    pages: list[tuple[bytes, dict]], watchlist: set[str], observed_at: datetime
) -> pd.DataFrame:
    rows = []
    for raw, page_hash in _page_rows(pages, "result"):
        symbol = str(raw.get("SECURITY_CODE") or "").zfill(6)
        if symbol not in watchlist:
            continue
        item = {
            "symbol": symbol,
            "name": raw.get("SECURITY_NAME_ABBR"),
            "rating_org_count": raw.get("RATING_ORG_NUM"),
            "rating_buy_count": raw.get("RATING_BUY_NUM"),
            "rating_add_count": raw.get("RATING_ADD_NUM"),
            "rating_neutral_count": raw.get("RATING_NEUTRAL_NUM"),
            "rating_reduce_count": raw.get("RATING_REDUCE_NUM"),
            "rating_sell_count": raw.get("RATING_SALE_NUM"),
            "forecast_year_1": raw.get("YEAR1"),
            "forecast_eps_1": raw.get("EPS1"),
            "forecast_year_2": raw.get("YEAR2"),
            "forecast_eps_2": raw.get("EPS2"),
            "forecast_year_3": raw.get("YEAR3"),
            "forecast_eps_3": raw.get("EPS3"),
            "forecast_year_4": raw.get("YEAR4"),
            "forecast_eps_4": raw.get("EPS4"),
            "target_price_min": raw.get("DEC_AIMPRICEMIN"),
            "target_price_max": raw.get("DEC_AIMPRICEMAX"),
            "provider_industry": raw.get("INDUSTRY_BOARD"),
            "raw_page_sha256": page_hash,
        }
        item["identity_sha256"] = canonical_hash(item)
        rows.append(item)
    result = pd.DataFrame(rows)
    if result.empty or result["symbol"].duplicated().any():
        raise ValueError("expectation snapshot is empty or contains duplicate symbols")
    result.insert(0, "observed_at_utc", observed_at.astimezone(timezone.utc).isoformat())
    return result.sort_values("symbol").reset_index(drop=True)


def normalize_flows(
    pages: list[tuple[bytes, dict]], watchlist: set[str], observed_at: datetime
) -> pd.DataFrame:
    rows = []
    for raw, page_hash in _page_rows(pages, "data"):
        symbol = str(raw.get("f12") or "").zfill(6)
        if symbol not in watchlist:
            continue
        source_epoch = pd.to_numeric(raw.get("f124"), errors="coerce")
        source_time = (
            datetime.fromtimestamp(float(source_epoch), tz=timezone.utc).isoformat()
            if pd.notna(source_epoch) and float(source_epoch) > 0
            else None
        )
        item = {
            "symbol": symbol,
            "name": raw.get("f14"),
            "source_timestamp_utc": source_time,
            "close": raw.get("f2"),
            "change_pct": raw.get("f3"),
            "main_net_inflow": raw.get("f62"),
            "main_net_inflow_ratio": raw.get("f184"),
            "super_large_net_inflow": raw.get("f66"),
            "super_large_net_inflow_ratio": raw.get("f69"),
            "large_net_inflow": raw.get("f72"),
            "large_net_inflow_ratio": raw.get("f75"),
            "medium_net_inflow": raw.get("f78"),
            "medium_net_inflow_ratio": raw.get("f81"),
            "small_net_inflow": raw.get("f84"),
            "small_net_inflow_ratio": raw.get("f87"),
            "raw_page_sha256": page_hash,
        }
        item["identity_sha256"] = canonical_hash(item)
        rows.append(item)
    result = pd.DataFrame(rows)
    if result.empty or result["symbol"].duplicated().any():
        raise ValueError("fund-flow snapshot is empty or contains duplicate symbols")
    result.insert(0, "observed_at_utc", observed_at.astimezone(timezone.utc).isoformat())
    return result.sort_values("symbol").reset_index(drop=True)


def attach_pit_industry(
    expectations: pd.DataFrame, industry_path: str | Path, as_of: str | pd.Timestamp
) -> pd.DataFrame:
    history = pd.read_csv(industry_path, dtype={"symbol": str})
    history["symbol"] = history["symbol"].str.zfill(6)
    history["industry_effective_date"] = pd.to_datetime(history["industry_effective_date"]).dt.normalize()
    cutoff = pd.Timestamp(as_of).normalize()
    history = history[history["industry_effective_date"] <= cutoff].sort_values(
        ["symbol", "industry_effective_date"]
    )
    latest = history.groupby("symbol", as_index=False).tail(1)[["symbol", "industry", "industry_effective_date"]]
    result = expectations.merge(latest, on="symbol", how="left", validate="one_to_one")
    if (result["industry_effective_date"] > cutoff).any():
        raise ValueError("future industry classification leaked into observation")
    return result


def industry_prosperity(expectations: pd.DataFrame, previous: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = expectations.copy()
    frame["forecast_eps_1"] = pd.to_numeric(frame["forecast_eps_1"], errors="coerce")
    frame["forecast_eps_2"] = pd.to_numeric(frame["forecast_eps_2"], errors="coerce")
    frame["forward_eps_slope"] = (frame["forecast_eps_2"] - frame["forecast_eps_1"]) / (
        frame["forecast_eps_1"].abs() + 0.05
    )
    frame["forward_eps_slope"] = frame["forward_eps_slope"].clip(-5, 5)
    if previous is not None and not previous.empty:
        prior = previous[["symbol", "forecast_year_1", "forecast_eps_1"]].copy()
        prior["forecast_eps_1"] = pd.to_numeric(prior["forecast_eps_1"], errors="coerce")
        prior = prior.rename(columns={"forecast_year_1": "prior_year_1", "forecast_eps_1": "prior_eps_1"})
        frame = frame.merge(prior, on="symbol", how="left", validate="one_to_one")
        same_year = frame["forecast_year_1"].astype(str).eq(frame["prior_year_1"].astype(str))
        frame["eps_revision"] = (frame["forecast_eps_1"] - frame["prior_eps_1"]).where(same_year)
    else:
        frame["eps_revision"] = float("nan")
    grouped = frame.groupby("industry", dropna=False)
    result = grouped.agg(
        symbol_count=("symbol", "nunique"),
        median_forward_eps_slope=("forward_eps_slope", "median"),
        mean_forward_eps_slope=("forward_eps_slope", "mean"),
        revision_coverage=("eps_revision", "count"),
        median_eps_revision=("eps_revision", "median"),
        positive_revision_breadth=("eps_revision", lambda value: (value.dropna() > 0).mean() if value.notna().any() else float("nan")),
    ).reset_index()
    return result.sort_values("industry", na_position="last").reset_index(drop=True)


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _completed_manifests(settings: ObservationSettings) -> list[Path]:
    return sorted(settings.data_root.glob("*/manifest.json"))


def _previous_expectations(settings: ObservationSettings) -> pd.DataFrame | None:
    manifests = _completed_manifests(settings)
    if not manifests:
        return None
    previous = manifests[-1].parent / "expectations.csv"
    return pd.read_csv(previous, dtype={"symbol": str}) if previous.exists() else None


def observe(
    target_date: str | None = None,
    *,
    now: datetime | None = None,
    settings: ObservationSettings | None = None,
    expectation_fetcher=fetch_expectation_pages,
    flow_fetcher=fetch_flow_pages,
) -> dict:
    from .freeze import verify_lock

    settings = settings or ObservationSettings()
    lock = verify_lock(settings)
    now = now or datetime.now(timezone.utc)
    shanghai_date = now.astimezone(SHANGHAI).date().isoformat()
    target_date = target_date or shanghai_date
    if target_date != shanghai_date:
        raise ValueError("historical backfill is forbidden; target must equal current Shanghai date")
    for manifest_path in _completed_manifests(settings):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["observed_date_shanghai"] == target_date:
            raise RuntimeError(f"PIT observation already exists for {target_date}")
    observation_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    destination = settings.data_root / observation_id
    destination.mkdir(parents=True, exist_ok=False)
    try:
        snapshot, watchlist = load_watchlist(settings.membership_path, target_date)
        expectation_pages = expectation_fetcher()
        flow_pages = flow_fetcher()
        for source, pages in (("expectations", expectation_pages), ("flows", flow_pages)):
            for index, (raw, _) in enumerate(pages, 1):
                _write_new(destination / "raw" / source / f"page_{index:04d}.json", raw)
        expectations = normalize_expectations(expectation_pages, watchlist, now)
        flows = normalize_flows(flow_pages, watchlist, now)
        expectations = attach_pit_industry(expectations, settings.industry_path, target_date)
        previous = _previous_expectations(settings)
        prosperity = industry_prosperity(expectations, previous)
        expectations.to_csv(destination / "expectations.csv", index=False, encoding="utf-8-sig")
        flows.to_csv(destination / "fund_flows.csv", index=False, encoding="utf-8-sig")
        prosperity.to_csv(destination / "industry_prosperity.csv", index=False, encoding="utf-8-sig")
        expectation_coverage = expectations["symbol"].nunique() / len(watchlist)
        flow_coverage = flows["symbol"].nunique() / len(watchlist)
        source_times = pd.to_datetime(flows["source_timestamp_utc"], utc=True, errors="coerce")
        future_source_times = int((source_times > pd.Timestamp(now)).sum())
        completed_before = len(_completed_manifests(settings))
        manifest = {
            "version": settings.version,
            "observation_id": observation_id,
            "observed_at_utc": now.astimezone(timezone.utc).isoformat(),
            "observed_date_shanghai": target_date,
            "membership_snapshot": snapshot,
            "watchlist_size": len(watchlist),
            "expectations": {
                "rows": len(expectations),
                "coverage": expectation_coverage,
                "source": EXPECTATION_URL,
                "raw_page_sha256": [sha256_bytes(item[0]) for item in expectation_pages],
            },
            "fund_flows": {
                "rows": len(flows),
                "coverage": flow_coverage,
                "source": FLOW_URL,
                "source_timestamp_min": source_times.min().isoformat() if source_times.notna().any() else None,
                "source_timestamp_max": source_times.max().isoformat() if source_times.notna().any() else None,
                "future_source_timestamps": future_source_times,
                "raw_page_sha256": [sha256_bytes(item[0]) for item in flow_pages],
            },
            "industry_prosperity": {
                "rows": len(prosperity),
                "has_prior_snapshot": previous is not None,
                "revision_rows": int(prosperity["revision_coverage"].sum()),
                "industry_source": str(settings.industry_path),
                "future_industry_dates": 0,
            },
            "prospective_capture_verified": bool(
                expectation_coverage >= settings.minimum_expectation_coverage
                and flow_coverage >= settings.minimum_flow_coverage
                and future_source_times == 0
            ),
            "historical_pit_verified": False,
            "distinct_completed_observations_before": completed_before,
            "minimum_training_observations": settings.minimum_training_observations,
            "effective_date_verified": False,
            "labels_matured": False,
            "model_training_ready": False,
            "production_prediction_ready": False,
            "execution_authorized": False,
            "frozen_inputs_intact": True,
            "lock_sha256": lock["lock_sha256"],
        }
        _write_new(destination / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
        report_path = settings.artifact_root / "observations" / f"{observation_id}.json"
        _write_new(report_path, json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
        return manifest
    except BaseException as error:
        failure = {
            "observation_id": observation_id,
            "observed_at_utc": now.astimezone(timezone.utc).isoformat(),
            "target_date": target_date,
            "error": f"{type(error).__name__}: {error}",
            "automatic_retry": False,
            "model_training_ready": False,
            "execution_authorized": False,
        }
        failure_path = settings.artifact_root / "failures" / f"{observation_id}.json"
        _write_new(failure_path, json.dumps(failure, ensure_ascii=False, indent=2).encode("utf-8"))
        raise
