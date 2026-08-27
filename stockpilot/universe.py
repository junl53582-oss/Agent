from __future__ import annotations

from pathlib import Path

import pandas as pd

UNIVERSE_COLUMNS = ["snapshot_date", "index_code", "symbol", "name", "exchange", "weight"]


def fetch_index_snapshot(index_code: str = "000300") -> pd.DataFrame:
    """Fetch and normalize the latest CSI index constituent snapshot."""
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("请先安装应用依赖: pip install -e .[app]") from exc

    raw = ak.index_stock_cons_weight_csindex(symbol=index_code)
    if raw.empty:
        raise ValueError(f"指数 {index_code} 没有返回成分股")
    mapping = {
        "日期": "snapshot_date",
        "指数代码": "index_code",
        "成分券代码": "symbol",
        "成分券名称": "name",
        "交易所": "exchange",
        "权重": "weight",
    }
    missing = sorted(set(mapping) - set(raw.columns))
    if missing:
        raise ValueError(f"指数接口字段变化，缺少: {', '.join(missing)}")
    result = raw.rename(columns=mapping)[list(mapping.values())].copy()
    result["snapshot_date"] = pd.to_datetime(result["snapshot_date"])
    result["index_code"] = result["index_code"].astype(str).str.zfill(6)
    result["symbol"] = result["symbol"].astype(str).str.zfill(6)
    result["weight"] = pd.to_numeric(result["weight"], errors="coerce")
    return result.sort_values("weight", ascending=False).reset_index(drop=True)


def save_universe(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, encoding="utf-8-sig")
    return target


def load_universe(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"symbol": str, "index_code": str})
    missing = sorted(set(UNIVERSE_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"股票池缺少字段: {', '.join(missing)}")
    data["symbol"] = data["symbol"].str.zfill(6)
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"])
    return data
