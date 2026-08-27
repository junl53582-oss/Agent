from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from stockpilot.data import REQUIRED_COLUMNS, validate_panel


MAPPING = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}


def _normalize_hfq(raw: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
    data = raw.rename(columns=MAPPING).copy()
    data["symbol"] = symbol
    if source == "tencent":
        volume = pd.to_numeric(data.get("amount"), errors="coerce")
        data["volume"] = volume
        mean_price = (
            pd.to_numeric(data.get("open"), errors="coerce")
            + pd.to_numeric(data.get("close"), errors="coerce")
        ) / 2
        data["amount"] = volume * mean_price
    for column in REQUIRED_COLUMNS:
        if column not in data:
            data[column] = pd.NA
    result = validate_panel(data[REQUIRED_COLUMNS])
    if (result[["open", "close", "high", "low"]] <= 0).any().any():
        raise ValueError("后复权数据仍包含非正价格")
    return result


def fetch_hfq_history(
    symbols: list[str] | pd.Series,
    start_date: str,
    end_date: str,
    output_path: str | Path = "data/market_history_v10_hfq.csv",
    cache_dir: str | Path = "data/raw_v10_hfq",
    workers: int = 8,
) -> tuple[pd.DataFrame, list[dict]]:
    try:
        import akshare as ak
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("请安装应用依赖 akshare") from exc
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    normalized = sorted({str(symbol).zfill(6) for symbol in symbols})
    compact_start = start_date.replace("-", "")
    compact_end = end_date.replace("-", "")

    def load_one(symbol: str) -> tuple[str, pd.DataFrame | None, str, str | None]:
        target = cache / f"{symbol}_{start_date}_{end_date}_hfq.csv"
        if target.exists():
            try:
                return symbol, validate_panel(pd.read_csv(target, dtype={"symbol": str})), "cache", None
            except Exception as exc:  # noqa: BLE001
                return symbol, None, "cache", type(exc).__name__
        errors = []
        try:
            raw = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=compact_start,
                end_date=compact_end,
                adjust="hfq",
                timeout=20,
            )
            if not raw.empty:
                frame = _normalize_hfq(raw, symbol, "eastmoney")
                frame.to_csv(target, index=False, encoding="utf-8-sig")
                return symbol, frame, "eastmoney", None
        except Exception as exc:  # noqa: BLE001
            errors.append(f"eastmoney:{type(exc).__name__}")
        market_symbol = ("sh" if symbol.startswith(("5", "6", "9")) else "sz") + symbol
        try:
            raw = ak.stock_zh_a_hist_tx(
                symbol=market_symbol,
                start_date=compact_start,
                end_date=compact_end,
                adjust="hfq",
                timeout=30,
            )
            if not raw.empty:
                frame = _normalize_hfq(raw, symbol, "tencent")
                frame.to_csv(target, index=False, encoding="utf-8-sig")
                return symbol, frame, "tencent", None
        except Exception as exc:  # noqa: BLE001
            errors.append(f"tencent:{type(exc).__name__}")
        return symbol, None, "failed", ";".join(errors) or "empty"

    results = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as executor:
        futures = [executor.submit(load_one, symbol) for symbol in normalized]
        for future in as_completed(futures):
            results.append(future.result())
    pieces = [frame for _, frame, _, _ in results if frame is not None and not frame.empty]
    failures = [
        {"symbol": symbol, "source": source, "error": error}
        for symbol, frame, source, error in results
        if frame is None or frame.empty
    ]
    if not pieces:
        raise RuntimeError("没有取得任何V10后复权历史行情")
    panel = validate_panel(pd.concat(pieces, ignore_index=True))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output, index=False, encoding="utf-8-sig")
    pd.DataFrame(failures).to_csv(
        output.with_suffix(".failures.csv"), index=False, encoding="utf-8-sig"
    )
    sources = pd.Series([source for _, frame, source, _ in results if frame is not None]).value_counts()
    manifest = {
        "requested_symbols": len(normalized),
        "output_symbols": int(panel["symbol"].nunique()),
        "output_rows": len(panel),
        "failures": len(failures),
        "date_min": str(panel["date"].min().date()),
        "date_max": str(panel["date"].max().date()),
        "price_positive": bool((panel[["open", "high", "low", "close"]] > 0).all().all()),
        "volume_coverage": float(panel["volume"].notna().mean()),
        "amount_coverage": float(panel["amount"].notna().mean()),
        "sources": {str(key): int(value) for key, value in sources.items()},
    }
    pd.Series(manifest).to_json(
        output.with_suffix(".manifest.json"), force_ascii=False, indent=2
    )
    return panel, failures

