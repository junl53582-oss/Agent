from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .integrity import canonical_frame_bytes, read_verified_json, sha256_bytes, sha256_file, write_immutable_json


HORIZONS = (1, 5, 20)


def _normalize_market(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol", "open"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"market source missing {sorted(missing)}")
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"]).dt.normalize()
    output["symbol"] = output["symbol"].astype(str).str.zfill(6)
    if output.duplicated(["date", "symbol"]).any():
        raise ValueError("market source contains duplicate date/symbol rows")
    return output


def _normalize_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "open"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"benchmark source missing {sorted(missing)}")
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"]).dt.normalize()
    if output["date"].duplicated().any():
        raise ValueError("benchmark source contains duplicate dates")
    return output


def _frame_fingerprint(frame: pd.DataFrame, keys: list[str]) -> str:
    return sha256_bytes(canonical_frame_bytes(frame, keys))


def _read_csv(path: Path, kind: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"symbol": str})
    return _normalize_market(frame) if kind == "market" else _normalize_benchmark(frame)


def _verify_supplied_frame(supplied: pd.DataFrame | None, parsed: pd.DataFrame, keys: list[str], name: str) -> pd.DataFrame:
    if supplied is None:
        return parsed
    normalized = _normalize_market(supplied) if name == "market" else _normalize_benchmark(supplied)
    if _frame_fingerprint(normalized, keys) != _frame_fingerprint(parsed, keys):
        raise RuntimeError(f"{name} DataFrame provenance mismatch")
    return normalized


def verify_corporate_action_provenance(dataset_path: str | Path, manifest_path: str | Path) -> dict:
    dataset = Path(dataset_path)
    manifest = Path(manifest_path)
    actual = sha256_file(dataset)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    mapping = payload.get("sha256") or payload.get("files") or {}
    key = dataset.as_posix()
    expected = mapping.get(key) or mapping.get(str(dataset).replace("/", "\\"))
    if expected is None or str(expected).lower() != actual.lower():
        raise RuntimeError("corporate action dataset is not bound by its manifest")
    return {
        "price_adjustment_mode": "HFQ_PIT_GOVERNED",
        "corporate_action_dataset_hash": actual,
        "corporate_action_manifest_hash": sha256_file(manifest),
        "corporate_action_verified": True,
    }


def settle_mature_labels(
    predictions: pd.DataFrame,
    *,
    market_source_path: str | Path,
    benchmark_source_path: str | Path,
    corporate_action_dataset_path: str | Path,
    corporate_action_manifest_path: str | Path,
    ledger_root: str | Path,
    as_of: str | pd.Timestamp,
    expected_universe_by_date: dict[str, set[str]],
    market: pd.DataFrame | None = None,
    benchmark: pd.DataFrame | None = None,
) -> list[dict]:
    market_path, benchmark_path = Path(market_source_path), Path(benchmark_source_path)
    parsed_market = _read_csv(market_path, "market")
    parsed_benchmark = _read_csv(benchmark_path, "benchmark")
    prices = _verify_supplied_frame(market, parsed_market, ["date", "symbol"], "market")
    bench = _verify_supplied_frame(benchmark, parsed_benchmark, ["date"], "benchmark")
    action = verify_corporate_action_provenance(
        corporate_action_dataset_path, corporate_action_manifest_path
    )
    calendar = pd.DatetimeIndex(prices["date"].drop_duplicates().sort_values())
    as_of_date = pd.Timestamp(as_of).normalize()
    if calendar.empty or calendar.max() > as_of_date or bench["date"].max() > as_of_date:
        raise ValueError("settlement source contains data after as_of")
    benchmark_lookup = bench.set_index("date")["open"]
    written: list[dict] = []
    for prediction in predictions.itertuples(index=False):
        prediction_date = pd.Timestamp(prediction.date).normalize()
        prediction_key = str(prediction_date.date())
        expected = {str(value).zfill(6) for value in expected_universe_by_date.get(prediction_key, set())}
        if not expected:
            raise ValueError(f"prediction universe provenance missing for {prediction_key}")
        position = calendar.get_indexer([prediction_date])[0]
        if position < 0:
            continue
        symbol = str(prediction.symbol).zfill(6)
        if symbol not in expected:
            raise ValueError("prediction symbol is outside its PIT universe proof")
        for horizon in HORIZONS:
            entry_pos, exit_pos = position + 1, position + horizon + 1
            if exit_pos >= len(calendar) or calendar[exit_pos] > as_of_date:
                continue
            entry_date, maturity_date = calendar[entry_pos], calendar[exit_pos]
            entry_rows = prices[prices["date"].eq(entry_date) & prices["symbol"].eq(symbol)]
            exit_rows = prices[prices["date"].eq(maturity_date) & prices["symbol"].eq(symbol)]
            entry_open = pd.to_numeric(entry_rows["open"].iloc[0], errors="coerce") if len(entry_rows) else np.nan
            exit_open = pd.to_numeric(exit_rows["open"].iloc[0], errors="coerce") if len(exit_rows) else np.nan
            status = "SETTLED"
            if not np.isfinite(entry_open):
                status = "MISSING_ENTRY_PRICE"
            elif not np.isfinite(exit_open):
                status = "MISSING_EXIT_PRICE"
            elif len(exit_rows) and bool(exit_rows.iloc[0].get("is_delisted", False)):
                status = "DELISTED_AT_EXIT"
            elif (len(entry_rows) and bool(entry_rows.iloc[0].get("is_suspended", False))) or (
                len(exit_rows) and bool(exit_rows.iloc[0].get("is_suspended", False))
            ):
                status = "SUSPENDED_PRICE_UNAVAILABLE"
            benchmark_entry = pd.to_numeric(benchmark_lookup.get(entry_date, np.nan), errors="coerce")
            benchmark_exit = pd.to_numeric(benchmark_lookup.get(maturity_date, np.nan), errors="coerce")
            if status == "SETTLED" and (not np.isfinite(benchmark_entry) or not np.isfinite(benchmark_exit)):
                status = "MISSING_BENCHMARK_PRICE"
            forward_return = float(exit_open / entry_open - 1) if status == "SETTLED" else None
            benchmark_return = float(benchmark_exit / benchmark_entry - 1) if status == "SETTLED" else None
            record = {
                "prediction_date": prediction_key,
                "maturity_date": str(maturity_date.date()),
                "symbol": symbol,
                "horizon": horizon,
                "entry_date": str(entry_date.date()),
                "entry_price": float(entry_open) if np.isfinite(entry_open) else None,
                "exit_price": float(exit_open) if np.isfinite(exit_open) else None,
                "forward_return": forward_return,
                "benchmark_return": benchmark_return,
                "excess_return": forward_return - benchmark_return if status == "SETTLED" else None,
                "status": status,
                "expected_universe_size": len(expected),
                "price_source_sha256": sha256_file(market_path),
                "benchmark_source_sha256": sha256_file(benchmark_path),
                "price_provenance_verified": True,
                "benchmark_provenance_verified": True,
                **action,
                "label_fully_verified": status == "SETTLED" and action["corporate_action_verified"],
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "execution_authorized": False,
            }
            target = Path(ledger_root) / prediction_key / f"{symbol}_{horizon}d.json"
            if target.exists():
                existing = read_verified_json(target)
                comparable = {key: value for key, value in record.items() if key != "recorded_at_utc"}
                old = {key: value for key, value in existing.items() if key != "recorded_at_utc"}
                if comparable != old:
                    raise RuntimeError(f"mature label is immutable: {target}")
                written.append(existing)
                continue
            write_immutable_json(target, record)
            written.append(record)
    return written


def load_verified_label_records(root: str | Path) -> list[dict]:
    return [read_verified_json(path) for path in sorted(Path(root).glob("*/*.json"))]
