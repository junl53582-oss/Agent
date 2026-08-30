from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .immutable import sha256_file, write_new_json


HORIZONS = (1, 5, 20)


def _calendar_index(market: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(market["date"]).dt.normalize().drop_duplicates().sort_values())


def _price_row(frame: pd.DataFrame, date: pd.Timestamp, symbol: str) -> pd.Series | None:
    rows = frame[(frame["date"].eq(date)) & (frame["symbol"].eq(symbol))]
    if len(rows) > 1:
        raise ValueError("market prices contain duplicate date/symbol rows")
    return None if rows.empty else rows.iloc[0]


def settle_mature_labels(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    price_source_path: str | Path,
    benchmark_source_path: str | Path,
    ledger_root: str | Path,
    as_of: str | pd.Timestamp,
    corporate_action_handling: str,
) -> list[dict]:
    """Append only records whose exit trading date has genuinely arrived."""
    prices = market.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices["symbol"] = prices["symbol"].astype(str).str.zfill(6)
    bench = benchmark.copy()
    bench["date"] = pd.to_datetime(bench["date"]).dt.normalize()
    calendar = _calendar_index(prices)
    as_of_date = pd.Timestamp(as_of).normalize()
    if calendar.empty or calendar.max() > as_of_date:
        raise ValueError("market input contains prices after settlement as_of")
    bench_lookup = bench.set_index("date")["open"]
    source_hash = sha256_file(price_source_path)
    benchmark_hash = sha256_file(benchmark_source_path)
    written: list[dict] = []
    for prediction in predictions.itertuples(index=False):
        prediction_date = pd.Timestamp(prediction.date).normalize()
        location = calendar.get_indexer([prediction_date])[0]
        if location < 0:
            continue
        symbol = str(prediction.symbol).zfill(6)
        for horizon in HORIZONS:
            entry_pos, exit_pos = location + 1, location + horizon + 1
            if exit_pos >= len(calendar) or calendar[exit_pos] > as_of_date:
                continue
            entry_date, maturity_date = calendar[entry_pos], calendar[exit_pos]
            entry = _price_row(prices, entry_date, symbol)
            exit_row = _price_row(prices, maturity_date, symbol)
            status = "SETTLED"
            entry_open = np.nan if entry is None else pd.to_numeric(entry.get("open"), errors="coerce")
            exit_open = np.nan if exit_row is None else pd.to_numeric(exit_row.get("open"), errors="coerce")
            if entry is None or not np.isfinite(entry_open):
                status = "MISSING_ENTRY_PRICE"
            elif exit_row is None or not np.isfinite(exit_open):
                status = "MISSING_EXIT_PRICE"
            elif bool(exit_row.get("is_delisted", False)):
                status = "DELISTED_AT_EXIT"
            elif bool(entry.get("is_suspended", False)) or bool(exit_row.get("is_suspended", False)):
                status = "SUSPENDED_PRICE_UNAVAILABLE"
            benchmark_entry = pd.to_numeric(bench_lookup.get(entry_date, np.nan), errors="coerce")
            benchmark_exit = pd.to_numeric(bench_lookup.get(maturity_date, np.nan), errors="coerce")
            if status == "SETTLED" and (not np.isfinite(benchmark_entry) or not np.isfinite(benchmark_exit)):
                status = "MISSING_BENCHMARK_PRICE"
            forward_return = float(exit_open / entry_open - 1) if status == "SETTLED" else None
            benchmark_return = float(benchmark_exit / benchmark_entry - 1) if status == "SETTLED" else None
            record = {
                "prediction_date": str(prediction_date.date()),
                "maturity_date": str(maturity_date.date()),
                "symbol": symbol,
                "horizon": horizon,
                "entry_date": str(entry_date.date()),
                "entry_price": float(entry_open) if np.isfinite(entry_open) else None,
                "exit_price": float(exit_open) if np.isfinite(exit_open) else None,
                "corporate_action_handling": corporate_action_handling,
                "forward_return": forward_return,
                "benchmark_return": benchmark_return,
                "excess_return": forward_return - benchmark_return if status == "SETTLED" else None,
                "status": status,
                "price_source_sha256": source_hash,
                "benchmark_source_sha256": benchmark_hash,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "execution_authorized": False,
            }
            target = Path(ledger_root) / str(prediction_date.date()) / f"{symbol}_{horizon}d.json"
            if target.exists():
                existing = json.loads(target.read_text(encoding="utf-8"))
                comparable = {key: value for key, value in record.items() if key != "recorded_at_utc"}
                old = {key: value for key, value in existing.items() if key != "recorded_at_utc"}
                if comparable != old:
                    raise RuntimeError(f"mature label is immutable: {target}")
                written.append(existing)
                continue
            write_new_json(target, record)
            written.append(record)
    return written


def load_label_records(root: str | Path) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(Path(root).glob("*/*.json"))]
