from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

DOLT_API = "https://www.dolthub.com/api/v1alpha1/chenditc/investment_data/master"
INDEX_CODE_MAP = {
    "000300": "399300.SZ",
    "000905": "000905.SH",
    "000906": "000906.SH",
    "000852": "000852.SH",
    "000985": "000985.SH",
}
MEMBERSHIP_COLUMNS = ["snapshot_date", "index_code", "symbol", "weight", "source"]


def _query_dolt(sql: str, timeout: int = 90, retries: int = 3) -> list[dict]:
    url = f"{DOLT_API}?q={urllib.parse.quote(sql)}"
    request = urllib.request.Request(url, headers={"User-Agent": "StockPilot-CN/0.1"})
    payload = None
    last_error: Exception | None = None
    retryable = (
        urllib.error.URLError,
        TimeoutError,
        ConnectionError,
        http.client.RemoteDisconnected,
        json.JSONDecodeError,
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except retryable as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    if payload is None:
        raise RuntimeError(f"历史成分查询失败，重试{retries}次: {last_error}") from last_error
    status = payload.get("query_execution_status")
    if status != "Success":
        message = payload.get("query_execution_message") or status
        raise RuntimeError(f"历史成分查询失败: {message}")
    return payload.get("rows", [])


def _history_dates(dolt_code: str, start_date: str, end_date: str) -> list[str]:
    quote = "'"
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    dates: set[str] = set()
    for year in range(start.year, end.year + 1):
        chunk_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        chunk_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        sql = (
            "SELECT DISTINCT trade_date FROM ts_index_weight "
            f"WHERE index_code={quote}{dolt_code}{quote} "
            f"AND trade_date >= {quote}{chunk_start.date()}{quote} "
            f"AND trade_date <= {quote}{chunk_end.date()}{quote} ORDER BY trade_date"
        )
        dates.update(str(row["trade_date"]) for row in _query_dolt(sql))
    return sorted(dates)


def fetch_membership_history(
    index_code: str,
    start_date: str,
    end_date: str,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch historical index weights in <=900-row batches and retain membership changes."""
    normalized_code = str(index_code).zfill(6)
    dolt_code = INDEX_CODE_MAP.get(normalized_code)
    if dolt_code is None:
        raise ValueError(f"暂不支持指数 {normalized_code}")
    dates = _history_dates(dolt_code, start_date, end_date)
    if not dates:
        raise ValueError("指定区间没有历史成分数据")

    rows: list[dict] = []
    quote = "'"
    for offset in range(0, len(dates), 3):
        batch = dates[offset : offset + 3]
        date_sql = ",".join(f"{quote}{date}{quote}" for date in batch)
        sql = (
            "SELECT trade_date,index_code,stock_code,weight FROM ts_index_weight "
            f"WHERE index_code={quote}{dolt_code}{quote} "
            f"AND trade_date IN ({date_sql}) ORDER BY trade_date,stock_code"
        )
        rows.extend(_query_dolt(sql))
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise ValueError("历史成分查询返回空数据")
    raw["snapshot_date"] = pd.to_datetime(raw["trade_date"])
    raw["index_code"] = normalized_code
    raw["symbol"] = raw["stock_code"].astype(str).str[:6].str.zfill(6)
    raw["weight"] = pd.to_numeric(raw["weight"], errors="coerce")
    raw["source"] = "chenditc/investment_data:ts_index_weight"
    raw = raw[MEMBERSHIP_COLUMNS].drop_duplicates(["snapshot_date", "symbol"])

    signatures = raw.groupby("snapshot_date")["symbol"].apply(
        lambda values: "|".join(sorted(values))
    )
    change_dates = signatures.index[signatures.ne(signatures.shift())]
    history = raw[raw["snapshot_date"].isin(change_dates)].copy()
    history = history.sort_values(["snapshot_date", "symbol"]).reset_index(drop=True)
    if cache_path is not None:
        save_membership_history(history, cache_path)
    return history


def save_membership_history(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, encoding="utf-8-sig")
    return target


def load_membership_history(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"symbol": str, "index_code": str})
    missing = sorted(set(MEMBERSHIP_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"历史成分文件缺少字段: {', '.join(missing)}")
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"], errors="raise")
    data["symbol"] = data["symbol"].str.zfill(6)
    data["index_code"] = data["index_code"].str.zfill(6)
    return data.sort_values(["snapshot_date", "symbol"]).reset_index(drop=True)


def attach_point_in_time_membership(panel: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Mark membership using only the most recent snapshot available on each market date."""
    if history.empty:
        raise ValueError("历史成分数据为空")
    result = panel.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["in_universe"] = False
    result["membership_known"] = False
    result["membership_snapshot_date"] = pd.NaT
    result["membership_universe_size"] = 0
    result["membership_available_size"] = 0
    snapshot_dates = [
        pd.Timestamp(value) for value in history["snapshot_date"].drop_duplicates().sort_values()
    ]
    for position, snapshot_date in enumerate(snapshot_dates):
        next_date = snapshot_dates[position + 1] if position + 1 < len(snapshot_dates) else None
        date_mask = result["date"] >= snapshot_date
        if next_date is not None:
            date_mask &= result["date"] < next_date
        members = set(history.loc[history["snapshot_date"] == snapshot_date, "symbol"])
        available = set(result.loc[date_mask, "symbol"]) & members
        result.loc[date_mask, "membership_known"] = True
        result.loc[date_mask, "membership_snapshot_date"] = snapshot_date
        result.loc[date_mask, "membership_universe_size"] = len(members)
        result.loc[date_mask, "membership_available_size"] = len(available)
        result.loc[date_mask & result["symbol"].isin(members), "in_universe"] = True
    return result


def export_qlib_intervals(
    history: pd.DataFrame, path: str | Path, end_date: str | pd.Timestamp | None = None
) -> Path:
    """Export Qlib-compatible symbol/start/end membership intervals."""
    dates = [
        pd.Timestamp(value) for value in history["snapshot_date"].drop_duplicates().sort_values()
    ]
    rows: list[dict] = []
    active: dict[str, pd.Timestamp] = {}
    previous: set[str] = set()
    for date in dates:
        current = set(history.loc[history["snapshot_date"] == date, "symbol"])
        for symbol in current - previous:
            active[symbol] = date
        for symbol in previous - current:
            prefix = "SH" if symbol.startswith(("5", "6", "9")) else "SZ"
            rows.append(
                {
                    "instrument": f"{prefix}{symbol}",
                    "start": active.pop(symbol),
                    "end": date - pd.offsets.Day(1),
                }
            )
        previous = current
    final_end = (
        pd.Timestamp(end_date).normalize()
        if end_date is not None
        else pd.Timestamp.today().normalize()
    )
    for symbol, start in active.items():
        prefix = "SH" if symbol.startswith(("5", "6", "9")) else "SZ"
        rows.append({"instrument": f"{prefix}{symbol}", "start": start, "end": final_end})
    intervals = pd.DataFrame(rows).sort_values(["instrument", "start"])
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(target, sep="\t", header=False, index=False, date_format="%Y-%m-%d")
    return target
