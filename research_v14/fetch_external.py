from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import akshare as ak
import pandas as pd
import requests

from .data_config import V14DataSettings


ANALYST_COLUMNS = [
    "symbol", "report_date", "title", "rating", "institution", "industry", "report_url"
]
NORTHBOUND_COLUMNS = [
    "symbol", "holding_date", "close", "change_pct", "holding_shares",
    "holding_market_cap", "holding_ratio", "added_shares", "added_amount", "market_cap_change",
]
ANNOUNCEMENT_COLUMNS = [
    "symbol", "announcement_date", "title", "announcement_id", "org_id", "announcement_url"
]


def _symbols(settings: V14DataSettings) -> list[str]:
    data = pd.read_csv(settings.membership_path, dtype={"symbol": str})
    return sorted(data["symbol"].astype(str).str.zfill(6).unique())


def _write_cache(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.reindex(columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def _fetch_analyst(symbol: str, path: Path) -> tuple[str, int, str | None]:
    if path.exists():
        return symbol, len(pd.read_csv(path)), None
    try:
        raw = ak.stock_research_report_em(symbol=symbol)
        if raw.empty:
            result = pd.DataFrame(columns=ANALYST_COLUMNS)
        else:
            result = pd.DataFrame({
                "symbol": raw.iloc[:, 1].astype(str).str.zfill(6),
                "report_date": pd.to_datetime(raw.iloc[:, -2], errors="coerce"),
                "title": raw.iloc[:, 3].astype(str),
                "rating": raw.iloc[:, 4].fillna("").astype(str),
                "institution": raw.iloc[:, 5].fillna("").astype(str),
                "industry": raw.iloc[:, -3].fillna("").astype(str),
                "report_url": raw.iloc[:, -1].fillna("").astype(str),
            }).dropna(subset=["report_date"])
        _write_cache(result.drop_duplicates(["symbol", "report_date", "title", "institution"]), path, ANALYST_COLUMNS)
        return symbol, len(result), None
    except Exception as exc:
        return symbol, 0, f"{type(exc).__name__}: {exc}"


def _fetch_northbound(symbol: str, path: Path) -> tuple[str, int, str | None]:
    if path.exists():
        return symbol, len(pd.read_csv(path)), None
    try:
        raw = ak.stock_hsgt_individual_em(symbol=symbol)
        if raw.empty:
            result = pd.DataFrame(columns=NORTHBOUND_COLUMNS)
        else:
            result = pd.DataFrame({
                "symbol": symbol,
                "holding_date": pd.to_datetime(raw.iloc[:, 0], errors="coerce"),
                "close": pd.to_numeric(raw.iloc[:, 1], errors="coerce"),
                "change_pct": pd.to_numeric(raw.iloc[:, 2], errors="coerce"),
                "holding_shares": pd.to_numeric(raw.iloc[:, 3], errors="coerce"),
                "holding_market_cap": pd.to_numeric(raw.iloc[:, 4], errors="coerce"),
                "holding_ratio": pd.to_numeric(raw.iloc[:, 5], errors="coerce"),
                "added_shares": pd.to_numeric(raw.iloc[:, 6], errors="coerce"),
                "added_amount": pd.to_numeric(raw.iloc[:, 7], errors="coerce"),
                "market_cap_change": pd.to_numeric(raw.iloc[:, 8], errors="coerce"),
            }).dropna(subset=["holding_date"])
        _write_cache(result.drop_duplicates(["symbol", "holding_date"]), path, NORTHBOUND_COLUMNS)
        return symbol, len(result), None
    except Exception as exc:
        # Missing connect eligibility is valid missingness, not a reason to fabricate zero holdings.
        _write_cache(pd.DataFrame(columns=NORTHBOUND_COLUMNS), path, NORTHBOUND_COLUMNS)
        return symbol, 0, f"{type(exc).__name__}: {exc}"


def _cninfo_map() -> dict[str, str]:
    from akshare.stock_feature import stock_disclosure_cninfo as module

    return module.__get_stock_json("沪深京")


def _fetch_announcements(
    symbol: str,
    org_id: str | None,
    path: Path,
    start_date: str,
    end_date: str,
) -> tuple[str, int, str | None]:
    if path.exists():
        return symbol, len(pd.read_csv(path)), None
    if not org_id:
        _write_cache(pd.DataFrame(columns=ANNOUNCEMENT_COLUMNS), path, ANNOUNCEMENT_COLUMNS)
        return symbol, 0, "missing_cninfo_org_id"
    payload = {
        "pageNum": "1", "pageSize": "30", "column": "szse", "tabName": "fulltext",
        "plate": "", "stock": f"{symbol},{org_id}", "searchkey": "", "secid": "",
        "category": "", "trade": "",
        "seDate": f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}~{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}",
        "sortName": "", "sortType": "", "isHLtitle": "true",
    }
    try:
        rows = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer": "http://www.cninfo.com.cn/",
        }
        with requests.Session() as session:
            def request_page(page: int):
                payload["pageNum"] = str(page)
                last_error = None
                for attempt in range(4):
                    try:
                        response = session.post(
                            "http://www.cninfo.com.cn/new/hisAnnouncement/query",
                            data=payload,
                            headers=headers,
                            timeout=45,
                        )
                        response.raise_for_status()
                        return response.json()
                    except Exception as exc:
                        last_error = exc
                        time.sleep(1.0 + attempt * 1.5)
                raise last_error

            body = request_page(1)
            total = int(body.get("totalAnnouncement") or 0)
            pages = max(1, (total + 29) // 30)
            for page in range(1, pages + 1):
                page_body = body if page == 1 else request_page(page)
                rows.extend(page_body.get("announcements") or [])
        raw = pd.DataFrame(rows)
        if raw.empty:
            result = pd.DataFrame(columns=ANNOUNCEMENT_COLUMNS)
        else:
            date = pd.to_datetime(raw["announcementTime"], unit="ms", utc=True, errors="coerce").dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
            result = pd.DataFrame({
                "symbol": raw["secCode"].astype(str).str.zfill(6),
                "announcement_date": date,
                "title": raw["announcementTitle"].astype(str).str.replace(r"<[^>]+>", "", regex=True),
                "announcement_id": raw["announcementId"].astype(str),
                "org_id": raw["orgId"].astype(str),
            }).dropna(subset=["announcement_date"])
            result["announcement_url"] = (
                "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=" + result["symbol"]
                + "&announcementId=" + result["announcement_id"] + "&orgId=" + result["org_id"]
            )
        _write_cache(result.drop_duplicates(["symbol", "announcement_id"]), path, ANNOUNCEMENT_COLUMNS)
        return symbol, len(result), None
    except Exception as exc:
        return symbol, 0, f"{type(exc).__name__}: {exc}"


def _run_parallel(label, symbols, worker, workers):
    failures, completed = [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol, rows, error = future.result()
            completed += 1
            if error:
                failures.append({"symbol": symbol, "error": error})
            if completed % 25 == 0 or completed == len(symbols):
                print(f"{label}: {completed}/{len(symbols)}, failures={len(failures)}", flush=True)
    return failures


def _combine(cache: Path, output: Path, columns: list[str]) -> dict:
    frames = [pd.read_csv(path, dtype={"symbol": str}) for path in sorted(cache.glob("*.csv"))]
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    result = result.reindex(columns=columns)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    return {"rows": len(result), "symbols": int(result["symbol"].nunique()) if len(result) else 0}


def _combine_announcements(cache: Path, output: Path) -> dict:
    frames = []
    for path in sorted(cache.glob("*.csv")):
        frame = pd.read_csv(path, dtype={"symbol": str})
        requested_symbol = path.stem.zfill(6)
        frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
        frames.append(frame[frame["symbol"] == requested_symbol])
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ANNOUNCEMENT_COLUMNS)
    result = result.reindex(columns=ANNOUNCEMENT_COLUMNS)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    return {"rows": len(result), "symbols": int(result["symbol"].nunique()) if len(result) else 0}


def fetch_analyst(settings: V14DataSettings) -> dict:
    symbols, cache = _symbols(settings), settings.cache_dir / "analyst"
    failures = _run_parallel(
        "analyst", symbols,
        lambda symbol: _fetch_analyst(symbol, cache / f"{symbol}.csv"),
        settings.workers,
    )
    result = _combine(cache, settings.analyst_output, ANALYST_COLUMNS)
    pd.DataFrame(failures).to_csv(settings.analyst_output.with_suffix(".failures.csv"), index=False, encoding="utf-8-sig")
    return {**result, "failures": len(failures)}


def fetch_northbound(settings: V14DataSettings) -> dict:
    symbols, cache = _symbols(settings), settings.cache_dir / "northbound"
    failures = _run_parallel(
        "northbound", symbols,
        lambda symbol: _fetch_northbound(symbol, cache / f"{symbol}.csv"),
        settings.workers,
    )
    result = _combine(cache, settings.northbound_output, NORTHBOUND_COLUMNS)
    pd.DataFrame(failures).to_csv(settings.northbound_output.with_suffix(".failures.csv"), index=False, encoding="utf-8-sig")
    return {**result, "failures": len(failures)}


def fetch_announcements(settings: V14DataSettings) -> dict:
    # The first audit cache used an invalid assumed page size and is preserved.
    symbols, cache = _symbols(settings), settings.cache_dir / "announcements_complete"
    org_map = _cninfo_map()
    start, end = settings.start_date.replace("-", ""), settings.end_date.replace("-", "")
    failures = _run_parallel(
        "announcements", symbols,
        lambda symbol: _fetch_announcements(symbol, org_map.get(symbol), cache / f"{symbol}.csv", start, end),
        min(4, settings.workers),
    )
    result = _combine_announcements(cache, settings.announcement_output)
    pd.DataFrame(failures).to_csv(settings.announcement_output.with_suffix(".failures.csv"), index=False, encoding="utf-8-sig")
    return {**result, "failures": len(failures)}


def main() -> None:
    parser = argparse.ArgumentParser(description="V14新增PIT数据抓取")
    parser.add_argument("source", choices=["analyst", "northbound", "announcements", "all"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    settings = V14DataSettings(workers=args.workers)
    result = {}
    if args.source in {"analyst", "all"}:
        result["analyst"] = fetch_analyst(settings)
    if args.source in {"northbound", "all"}:
        result["northbound"] = fetch_northbound(settings)
    if args.source in {"announcements", "all"}:
        result["announcements"] = fetch_announcements(settings)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
