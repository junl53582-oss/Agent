from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

RAW_COLUMNS = {
    "SECURITY_CODE": "symbol",
    "REPORT_DATE": "report_date",
    "NOTICE_DATE": "available_date",
    "UPDATE_DATE": "update_date",
    "BPS": "book_value_per_share",
    "EPSJB": "earnings_per_share",
    "ROEJQ": "roe",
    "ROIC": "roic",
    "ZCFZL": "debt_ratio",
    "TOTALOPERATEREVETZ": "revenue_growth",
    "PARENTNETPROFITTZ": "profit_growth",
    "JYXJLYYSR": "operating_cash_margin",
    "XSMLL": "gross_margin",
}
FUNDAMENTAL_COLUMNS = [
    "symbol",
    "report_date",
    "available_date",
    "update_date",
    "book_value_per_share",
    "earnings_per_share",
    "roe",
    "roic",
    "debt_ratio",
    "revenue_growth",
    "profit_growth",
    "operating_cash_margin",
    "gross_margin",
]


def _market_symbol(symbol: str) -> str:
    code = str(symbol).zfill(6)
    suffix = "SH" if code.startswith(("5", "6", "9")) else "SZ"
    return f"{code}.{suffix}"


def normalize_fundamentals(raw: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    data = raw.rename(columns=RAW_COLUMNS).copy()
    if "symbol" not in data and symbol is not None:
        data["symbol"] = str(symbol).zfill(6)
    missing = [
        column for column in ["symbol", "report_date", "available_date"] if column not in data
    ]
    if missing:
        raise ValueError(f"基本面数据缺少字段: {', '.join(missing)}")
    for column in FUNDAMENTAL_COLUMNS:
        if column not in data:
            data[column] = pd.NA
    data["symbol"] = data["symbol"].astype(str).str.zfill(6)
    for column in ["report_date", "available_date", "update_date"]:
        data[column] = pd.to_datetime(data[column], errors="coerce").dt.normalize()
    numeric = [
        column
        for column in FUNDAMENTAL_COLUMNS
        if column not in {"symbol", "report_date", "available_date", "update_date"}
    ]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["available_date", "report_date"])
    data = data[data["available_date"] >= data["report_date"]]
    return (
        data[FUNDAMENTAL_COLUMNS]
        .sort_values(["symbol", "available_date", "report_date", "update_date"])
        .drop_duplicates(["symbol", "available_date", "report_date"], keep="last")
        .reset_index(drop=True)
    )


def fetch_fundamentals(
    symbols: pd.Series | list[str],
    output_path: str | Path = "data/fundamentals_pit.csv",
    cache_dir: str | Path = "data/fundamental_cache",
    workers: int = 4,
) -> tuple[pd.DataFrame, list[str]]:
    try:
        import akshare as ak
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("请安装应用依赖 akshare") from exc

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    normalized_symbols = sorted({str(symbol).zfill(6) for symbol in symbols})

    def load_one(symbol: str) -> tuple[str, pd.DataFrame | None]:
        target = cache / f"{symbol}.csv"
        if target.exists():
            return symbol, pd.read_csv(target)
        try:
            raw = ak.stock_financial_analysis_indicator_em(
                symbol=_market_symbol(symbol), indicator="按报告期"
            )
        except (
            requests.RequestException,
            ConnectionError,
            TimeoutError,
            RuntimeError,
            ValueError,
            KeyError,
            TypeError,
        ):  # public provider failures are reported per symbol
            return symbol, None
        if raw.empty:
            return symbol, None
        raw.to_csv(target, index=False, encoding="utf-8-sig")
        return symbol, raw

    pieces: list[pd.DataFrame] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 6))) as executor:
        futures = [executor.submit(load_one, symbol) for symbol in normalized_symbols]
        for future in as_completed(futures):
            symbol, raw = future.result()
            if raw is None:
                failures.append(symbol)
            else:
                pieces.append(normalize_fundamentals(raw, symbol))
    if not pieces:
        raise RuntimeError("没有取得任何PIT基本面数据")
    result = pd.concat(pieces, ignore_index=True).sort_values(
        ["symbol", "available_date", "report_date"]
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(target, index=False, encoding="utf-8-sig")
    pd.DataFrame({"symbol": failures}).to_csv(
        target.with_suffix(".failures.csv"), index=False, encoding="utf-8-sig"
    )
    return result.reset_index(drop=True), failures


def load_fundamentals(path: str | Path) -> pd.DataFrame:
    return normalize_fundamentals(pd.read_csv(path, dtype={"symbol": str}))


def attach_fundamentals_asof(panel: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    left = panel.copy()
    left["date"] = pd.to_datetime(left["date"]).dt.normalize()
    left["symbol"] = left["symbol"].astype(str).str.zfill(6)
    right = fundamentals.sort_values(["available_date", "symbol", "report_date"]).copy()
    pieces = []
    value_columns = [column for column in FUNDAMENTAL_COLUMNS if column != "symbol"]
    for symbol, group in left.groupby("symbol", sort=False):
        filings = right[right["symbol"] == symbol]
        if filings.empty:
            missing = group.copy()
            for column in value_columns:
                missing[column] = pd.NA
            pieces.append(missing)
            continue
        pieces.append(
            pd.merge_asof(
                group.sort_values("date"),
                filings.drop(columns="symbol").sort_values("available_date"),
                left_on="date",
                right_on="available_date",
                direction="backward",
                allow_exact_matches=True,
            )
        )
    result = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"])
    result["fundamental_age_days"] = (result["date"] - result["available_date"]).dt.days
    pit_violation = result["available_date"].notna() & (result["available_date"] > result["date"])
    if pit_violation.any():
        raise RuntimeError("检测到基本面公告日未来数据泄漏")
    return result.reset_index(drop=True)
