from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]


def validate_panel(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"行情缺少字段: {', '.join(missing)}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["symbol"] = data["symbol"].astype(str).str.zfill(6)
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["date", "symbol", "open", "close"])
    data = data[(data["open"] > 0) & (data["close"] > 0)]
    data = data.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"])
    if data.empty:
        raise ValueError("行情数据为空")
    return data.reset_index(drop=True)


def make_demo_panel(
    symbols: int = 30,
    periods: int = 900,
    end: str | pd.Timestamp | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Create deterministic A-share-like data for offline smoke tests and UI demos."""
    if symbols < 6 or periods < 320:
        raise ValueError("演示数据至少需要6只股票和320个交易日")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp(end or "2025-12-31"), periods=periods)
    market = rng.normal(0.00025, 0.010, periods)
    regime = np.sin(np.linspace(0, 8 * np.pi, periods)) * 0.0015
    rows: list[pd.DataFrame] = []
    for i in range(symbols):
        symbol = f"{600000 + i:06d}" if i % 2 == 0 else f"{1 + i:06d}"
        quality = (i % 7 - 3) * 0.00008
        beta = 0.75 + (i % 6) * 0.1
        noise = rng.normal(0, 0.013 + (i % 5) * 0.001, periods)
        momentum = np.zeros(periods)
        for t in range(20, periods):
            momentum[t] = 0.08 * np.mean(noise[t - 20 : t])
        returns = beta * market + regime * ((i % 4) - 1.5) + quality + momentum + noise
        close = (8 + i * 0.7) * np.exp(np.cumsum(returns))
        overnight = rng.normal(0, 0.003, periods)
        open_ = close / np.exp(returns) * np.exp(overnight)
        spread = np.abs(rng.normal(0.008, 0.004, periods))
        high = np.maximum(open_, close) * (1 + spread)
        low = np.minimum(open_, close) * (1 - spread)
        volume = rng.lognormal(15.3 + (i % 4) * 0.2, 0.35, periods).astype(int)
        amount = volume * (open_ + close) / 2
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "amount": amount,
                    "name": f"演示股{i + 1:02d}",
                }
            )
        )
    return validate_panel(pd.concat(rows, ignore_index=True))


def fetch_akshare(
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    cache_dir: str | Path = "data/raw",
    provider: str = "auto",
    workers: int = 1,
) -> pd.DataFrame:
    """Download adjusted daily A-share bars. Each symbol is cached independently."""
    try:
        import akshare as ak
        import requests
    except ImportError as exc:
        raise RuntimeError("请先安装应用依赖: pip install -e .[app]") from exc

    provider_errors = (
        requests.RequestException,
        ConnectionError,
        TimeoutError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
    )

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    pieces: list[pd.DataFrame] = []
    mapping = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    if provider not in {"auto", "eastmoney", "tencent"}:
        raise ValueError("provider 必须是 auto、eastmoney 或 tencent")
    if workers < 1 or workers > 8:
        raise ValueError("workers 必须在1到8之间")

    def load_one(raw_symbol: str) -> tuple[str, pd.DataFrame, list[str]]:
        symbol = str(raw_symbol).strip().zfill(6)
        path = cache / f"{symbol}_{start_date}_{end_date}_{provider}.csv"
        local_errors: list[str] = []
        if path.exists():
            part = pd.read_csv(path)
        else:
            part = pd.DataFrame()
            if provider in {"auto", "eastmoney"}:
                try:
                    part = ak.stock_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=start_date.replace("-", ""),
                        end_date=end_date.replace("-", ""),
                        adjust="qfq",
                        timeout=15,
                    )
                except provider_errors as exc:
                    local_errors.append(f"{symbol}/eastmoney: {type(exc).__name__}")
                    if provider == "eastmoney":
                        return symbol, pd.DataFrame(), local_errors
            if part.empty and provider in {"auto", "tencent"}:
                market_symbol = ("sh" if symbol.startswith(("5", "6", "9")) else "sz") + symbol
                try:
                    part = ak.stock_zh_a_hist_tx(
                        symbol=market_symbol,
                        start_date=start_date.replace("-", ""),
                        end_date=end_date.replace("-", ""),
                        adjust="qfq",
                        timeout=15,
                    )
                except provider_errors as exc:
                    local_errors.append(f"{symbol}/tencent: {type(exc).__name__}")
            if part.empty:
                return symbol, part, local_errors
            part = part.rename(columns=mapping)
            part["symbol"] = symbol
            part.to_csv(path, index=False, encoding="utf-8-sig")
        part["symbol"] = symbol
        return symbol, part, local_errors

    normalized_symbols = [str(symbol).strip().zfill(6) for symbol in symbols]
    errors: list[str] = []
    if workers == 1:
        results = [load_one(symbol) for symbol in normalized_symbols]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(load_one, symbol) for symbol in normalized_symbols]
            for future in as_completed(futures):
                results.append(future.result())
    for _, part, local_errors in results:
        errors.extend(local_errors)
        if not part.empty:
            pieces.append(part)
    if not pieces:
        detail = "; ".join(errors[-5:])
        raise ValueError(f"没有下载到行情，请检查代码、日期或网络。{detail}")
    return validate_panel(pd.concat(pieces, ignore_index=True))


def save_panel(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    validate_panel(frame).to_csv(target, index=False, encoding="utf-8-sig")
    return target


def load_panel(path: str | Path) -> pd.DataFrame:
    return validate_panel(pd.read_csv(path))
