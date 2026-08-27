from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

EXPOSURE_COLUMNS = [
    "date",
    "symbol",
    "float_market_cap",
    "outstanding_share",
    "industry",
    "industry_code",
    "industry_effective_date",
    "exposure_source",
]


def _market_symbol(symbol: str) -> str:
    prefix = "sh" if symbol.startswith(("5", "6", "9")) else "sz"
    return prefix + symbol


def normalize_market_cap(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    required = {"date", "close", "outstanding_share"}
    if raw.empty or not required.issubset(raw.columns):
        return pd.DataFrame(columns=["date", "symbol", "float_market_cap", "outstanding_share"])
    result = raw[["date", "close", "outstanding_share"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result["outstanding_share"] = pd.to_numeric(result["outstanding_share"], errors="coerce")
    result["symbol"] = str(symbol).zfill(6)
    result["float_market_cap"] = result["close"] * result["outstanding_share"]
    return result.dropna(subset=["date", "float_market_cap"])[
        ["date", "symbol", "float_market_cap", "outstanding_share"]
    ]


def normalize_industry_history(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    columns = ["industry_effective_date", "industry", "industry_code", "industry_source"]
    if raw.empty or "变更日期" not in raw.columns or "分类标准" not in raw.columns:
        return pd.DataFrame(columns=columns)
    standards = raw["分类标准"].fillna("").astype(str)
    result = raw[standards.str.contains("申银万国", regex=False)].copy()
    if result.empty:
        return pd.DataFrame(columns=columns)
    result["standard_priority"] = (~result["分类标准"].astype(str).str.contains("旧")).astype(int)
    result["industry_effective_date"] = pd.to_datetime(result["变更日期"], errors="coerce")
    industry = pd.Series(index=result.index, dtype=object)
    for source in ["行业门类", "行业大类", "行业中类", "行业次类"]:
        if source in result:
            industry = industry.fillna(result[source].replace("", pd.NA))
    result["industry"] = industry
    result["industry_code"] = result.get("行业编码", pd.Series(index=result.index, dtype=object))
    result["industry_source"] = "cninfo:申银万国行业分类"
    result["symbol"] = str(symbol).zfill(6)
    return (
        result.dropna(subset=["industry_effective_date", "industry"])
        .sort_values(["industry_effective_date", "standard_priority"])
        .drop_duplicates("industry_effective_date", keep="last")[columns]
        .reset_index(drop=True)
    )


def combine_exposure(market_cap: pd.DataFrame, industry: pd.DataFrame) -> pd.DataFrame:
    if market_cap.empty:
        return pd.DataFrame(columns=EXPOSURE_COLUMNS)
    market = market_cap.sort_values("date").copy()
    market["symbol"] = market["symbol"].astype(str).str.zfill(6)
    if industry.empty:
        market["industry"] = pd.NA
        market["industry_code"] = pd.NA
        market["industry_effective_date"] = pd.NaT
        industry_available = False
    else:
        market = pd.merge_asof(
            market,
            industry.sort_values("industry_effective_date"),
            left_on="date",
            right_on="industry_effective_date",
            direction="backward",
        )
        industry_available = True
    market["exposure_source"] = "sina:outstanding_share"
    if industry_available:
        market.loc[market["industry"].notna(), "exposure_source"] += "+cninfo:industry_change"
    return market[EXPOSURE_COLUMNS].sort_values(["date", "symbol"]).reset_index(drop=True)


def fetch_exposures(
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    output_path: str | Path,
    cache_dir: str | Path = "data/exposure_cache",
    workers: int = 1,
) -> pd.DataFrame:
    """Fetch PIT float market cap and SW industry changes with per-symbol resume caches."""
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("请先安装应用依赖: pip install -e .[app]") from exc
    if workers < 1 or workers > 2:
        raise ValueError("为降低公开接口封禁风险，workers 必须在1到2之间")
    cache = Path(cache_dir)
    market_cache = cache / "market_cap"
    industry_cache = cache / "industry"
    market_cache.mkdir(parents=True, exist_ok=True)
    industry_cache.mkdir(parents=True, exist_ok=True)
    compact_start = start_date.replace("-", "")
    compact_end = end_date.replace("-", "")

    def load_one(raw_symbol: str) -> tuple[str, pd.DataFrame, str | None]:
        symbol = str(raw_symbol).zfill(6)
        market_path = market_cache / f"{symbol}_{compact_start}_{compact_end}.csv"
        industry_path = industry_cache / f"{symbol}_history_to_{compact_end}.csv"
        try:
            if market_path.exists():
                market = pd.read_csv(market_path, dtype={"symbol": str}, parse_dates=["date"])
            else:
                raw_market = ak.stock_zh_a_daily(
                    symbol=_market_symbol(symbol),
                    start_date=compact_start,
                    end_date=compact_end,
                    adjust="",
                )
                market = normalize_market_cap(raw_market, symbol)
                market.to_csv(market_path, index=False, encoding="utf-8-sig")
            if industry_path.exists():
                industry = pd.read_csv(industry_path, parse_dates=["industry_effective_date"])
            else:
                raw_industry = ak.stock_industry_change_cninfo(
                    symbol=symbol, start_date="19900101", end_date=compact_end
                )
                industry = normalize_industry_history(raw_industry, symbol)
                industry.to_csv(industry_path, index=False, encoding="utf-8-sig")
            return symbol, combine_exposure(market, industry), None
        except Exception as exc:  # noqa: BLE001 - a failed public endpoint must not abort resume
            return symbol, pd.DataFrame(columns=EXPOSURE_COLUMNS), type(exc).__name__

    normalized = sorted({str(symbol).zfill(6) for symbol in symbols})
    results: list[tuple[str, pd.DataFrame, str | None]] = []
    if workers == 1:
        results = [load_one(symbol) for symbol in normalized]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(load_one, symbol) for symbol in normalized]
            for future in as_completed(futures):
                results.append(future.result())
    pieces = [frame for _, frame, _ in results if not frame.empty]
    failures = [(symbol, error) for symbol, _, error in results if error]
    if not pieces:
        detail = ", ".join(f"{symbol}:{error}" for symbol, error in failures[:10])
        raise ValueError(f"没有取得暴露数据：{detail}")
    exposure = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"])
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    exposure.to_csv(target, index=False, encoding="utf-8-sig")
    failure_path = target.with_suffix(".failures.csv")
    pd.DataFrame(failures, columns=["symbol", "error"]).to_csv(
        failure_path, index=False, encoding="utf-8-sig"
    )
    return exposure.reset_index(drop=True)


def load_exposures(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"symbol": str})
    missing = sorted(set(EXPOSURE_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"暴露文件缺少字段: {', '.join(missing)}")
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data["industry_effective_date"] = pd.to_datetime(
        data["industry_effective_date"], errors="coerce"
    )
    data["symbol"] = data["symbol"].str.zfill(6)
    return data.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"])


def attach_exposures(panel: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in EXPOSURE_COLUMNS if column not in {"date", "symbol"}]
    result = panel.drop(columns=[column for column in columns if column in panel], errors="ignore")
    result = result.merge(exposure[["date", "symbol", *columns]], on=["date", "symbol"], how="left")
    invalid_pit = result["industry_effective_date"].notna() & (
        result["industry_effective_date"] > result["date"]
    )
    if invalid_pit.any():
        raise ValueError("行业暴露包含未来生效记录")
    return result


def attach_exposures_asof(panel: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    """Carry the last actually observed exposure forward for append-only shadow dates."""
    exposure_columns = [column for column in EXPOSURE_COLUMNS if column not in {"date", "symbol"}]
    pieces: list[pd.DataFrame] = []
    for symbol, group in panel.groupby("symbol", sort=False):
        history = exposure[exposure["symbol"] == symbol].copy()
        left = group.drop(columns=exposure_columns, errors="ignore").sort_values("date")
        if history.empty:
            left["float_market_cap"] = float("nan")
            left["outstanding_share"] = float("nan")
            left["industry"] = pd.Series(None, index=left.index, dtype=object)
            left["industry_code"] = pd.Series(None, index=left.index, dtype=object)
            left["industry_effective_date"] = pd.NaT
            left["exposure_source"] = pd.Series(None, index=left.index, dtype=object)
            left["exposure_observed_date"] = pd.NaT
            pieces.append(left)
            continue
        history = history[["date", *exposure_columns]].rename(
            columns={"date": "exposure_observed_date"}
        )
        merged = pd.merge_asof(
            left,
            history.sort_values("exposure_observed_date"),
            left_on="date",
            right_on="exposure_observed_date",
            direction="backward",
        )
        pieces.append(merged)
    result = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"])
    result["exposure_age_days"] = (
        result["date"] - pd.to_datetime(result["exposure_observed_date"])
    ).dt.days
    invalid_pit = result["industry_effective_date"].notna() & (
        pd.to_datetime(result["industry_effective_date"]) > result["date"]
    )
    if invalid_pit.any():
        raise ValueError("行业暴露包含未来生效记录")
    return result.reset_index(drop=True)


def exposure_coverage(panel: pd.DataFrame) -> dict:
    scope = (
        panel["in_universe"].fillna(False)
        if "in_universe" in panel
        else pd.Series(True, index=panel.index)
    )
    scoped = panel[scope]
    rows = len(scoped)
    return {
        "rows": rows,
        "float_market_cap_coverage": float(scoped["float_market_cap"].notna().mean())
        if rows and "float_market_cap" in scoped
        else 0.0,
        "industry_coverage": float(scoped["industry"].notna().mean())
        if rows and "industry" in scoped
        else 0.0,
        "industry_point_in_time": bool(
            "industry_effective_date" in scoped
            and not (
                scoped["industry_effective_date"].notna()
                & (scoped["industry_effective_date"] > scoped["date"])
            ).any()
        ),
    }
