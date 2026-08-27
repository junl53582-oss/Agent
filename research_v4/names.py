from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def normalize_stock_names(data: pd.DataFrame) -> pd.DataFrame:
    required = {"code", "name"}
    if not required.issubset(data.columns):
        raise ValueError("股票名称数据必须包含code和name字段")
    result = data[["code", "name"]].copy()
    result["symbol"] = result.pop("code").astype(str).str.strip().str.zfill(6)
    result["name"] = result["name"].astype(str).str.strip()
    result = result[
        result["symbol"].str.fullmatch(r"\d{6}")
        & result["name"].ne("")
        & result["name"].ne("nan")
    ]
    return result.drop_duplicates("symbol", keep="last").sort_values("symbol").reset_index(drop=True)


def fetch_stock_names(output: str | Path = "data/stock_names.csv") -> dict:
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("缺少akshare，无法刷新股票名称") from exc
    names = normalize_stock_names(ak.stock_info_a_code_name())
    names["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    names.to_csv(target, index=False, encoding="utf-8-sig")
    return {
        "rows": len(names),
        "output": str(target),
        "fetched_at_utc": names["fetched_at_utc"].iloc[0],
    }
