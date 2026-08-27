from __future__ import annotations

from pathlib import Path

import pandas as pd

from stockpilot.data import REQUIRED_COLUMNS, validate_panel


def normalize_cached_market(
    raw_dir: str | Path = "data/raw",
    output_path: str | Path = "data/market_history_v10.csv",
    pattern: str = "*_2015-01-01_2026-08-21_auto.csv",
) -> tuple[pd.DataFrame, dict]:
    """Normalize mixed Eastmoney/Tencent caches without altering frozen V9 data.

    Tencent's ``stock_zh_a_hist_tx`` response calls traded volume ``amount`` and
    does not provide monetary turnover.  Treating a union of both schemas as a
    single table silently leaves Tencent volume null.  V10 maps that field to
    volume and derives a conservative turnover proxy from the daily mean price.
    """
    pieces = []
    eastmoney_symbols = 0
    tencent_symbols = 0
    raw_rows = 0
    negative_price_rows = 0
    for path in sorted(Path(raw_dir).glob(pattern)):
        raw = pd.read_csv(path, dtype={"symbol": str, "股票代码": str})
        raw_rows += len(raw)
        symbol = str(raw.get("symbol", pd.Series([path.name[:6]])).iloc[0]).zfill(6)
        raw["symbol"] = symbol
        if "volume" not in raw.columns:
            tencent_symbols += 1
            traded_volume = pd.to_numeric(raw.get("amount"), errors="coerce")
            raw["volume"] = traded_volume
            mean_price = (
                pd.to_numeric(raw.get("open"), errors="coerce")
                + pd.to_numeric(raw.get("close"), errors="coerce")
            ) / 2
            raw["amount"] = traded_volume * mean_price
        else:
            eastmoney_symbols += 1
        for column in REQUIRED_COLUMNS:
            if column not in raw:
                raw[column] = pd.NA
        open_price = pd.to_numeric(raw["open"], errors="coerce")
        close_price = pd.to_numeric(raw["close"], errors="coerce")
        negative_price_rows += int(((open_price <= 0) | (close_price <= 0)).sum())
        pieces.append(raw[REQUIRED_COLUMNS])
    if not pieces:
        raise RuntimeError(f"没有找到V10原始行情缓存: {raw_dir}/{pattern}")
    panel = validate_panel(pd.concat(pieces, ignore_index=True))
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(target, index=False, encoding="utf-8-sig")
    report = {
        "raw_files": eastmoney_symbols + tencent_symbols,
        "eastmoney_schema_symbols": eastmoney_symbols,
        "tencent_schema_symbols": tencent_symbols,
        "raw_rows": raw_rows,
        "negative_price_rows_removed": negative_price_rows,
        "output_rows": len(panel),
        "output_symbols": int(panel["symbol"].nunique()),
        "volume_coverage": float(panel["volume"].notna().mean()),
        "amount_coverage": float(panel["amount"].notna().mean()),
        "date_min": str(panel["date"].min().date()),
        "date_max": str(panel["date"].max().date()),
    }
    pd.Series(report).to_json(
        target.with_suffix(".normalization.json"), force_ascii=False, indent=2
    )
    return panel, report

