from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from stockpilot.exposure import normalize_industry_history


INDUSTRY_COLUMNS = [
    "symbol",
    "industry_effective_date",
    "industry",
    "industry_code",
    "industry_source",
]


def fetch_industry_history(
    symbols: Iterable[str],
    end_date: str,
    output_path: str | Path,
    cache_dir: str | Path = "data/exposure_cache/industry",
) -> tuple[pd.DataFrame, list[str]]:
    """Fetch PIT industry changes sequentially.

    AkShare's CNInfo JavaScript decoder is not thread safe, so V9 deliberately
    performs this small request stream in one process and resumes per symbol.
    """
    try:
        import akshare as ak
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("请安装应用依赖 akshare") from exc

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    compact_end = end_date.replace("-", "")
    pieces: list[pd.DataFrame] = []
    failures: list[str] = []
    normalized = sorted({str(symbol).zfill(6) for symbol in symbols})
    for symbol in normalized:
        target = cache / f"{symbol}_history_to_{compact_end}.csv"
        try:
            if target.exists():
                frame = pd.read_csv(target)
                frame["industry_effective_date"] = pd.to_datetime(
                    frame["industry_effective_date"], errors="coerce"
                )
            else:
                raw = ak.stock_industry_change_cninfo(
                    symbol=symbol, start_date="19900101", end_date=compact_end
                )
                frame = normalize_industry_history(raw, symbol)
                frame.to_csv(target, index=False, encoding="utf-8-sig")
            if frame.empty:
                failures.append(symbol)
                continue
            frame["symbol"] = symbol
            pieces.append(frame[INDUSTRY_COLUMNS])
        except Exception:  # noqa: BLE001 - provider failures are audited below
            failures.append(symbol)

    if not pieces:
        raise RuntimeError("没有取得任何历史行业分类")
    result = (
        pd.concat(pieces, ignore_index=True)
        .drop_duplicates(["symbol", "industry_effective_date"], keep="last")
        .sort_values(["symbol", "industry_effective_date"])
        .reset_index(drop=True)
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    pd.DataFrame({"symbol": failures}).to_csv(
        output.with_suffix(".failures.csv"), index=False, encoding="utf-8-sig"
    )
    return result, failures


def load_industry_history(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"symbol": str})
    missing = sorted(set(INDUSTRY_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"行业历史缺少字段: {', '.join(missing)}")
    data["symbol"] = data["symbol"].str.zfill(6)
    data["industry_effective_date"] = pd.to_datetime(
        data["industry_effective_date"], errors="raise"
    )
    return data.sort_values(["symbol", "industry_effective_date"]).reset_index(drop=True)


def attach_industry_asof(panel: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    left = panel.drop(
        columns=["industry", "industry_code", "industry_source", "industry_effective_date"],
        errors="ignore",
    ).copy()
    left["date"] = pd.to_datetime(left["date"])
    left["symbol"] = left["symbol"].astype(str).str.zfill(6)
    pieces = []
    values = ["industry_effective_date", "industry", "industry_code", "industry_source"]
    for symbol, group in left.groupby("symbol", sort=False):
        changes = history[history["symbol"] == symbol].drop(columns="symbol")
        if changes.empty:
            missing = group.copy()
            missing["industry_effective_date"] = pd.NaT
            for column in values[1:]:
                missing[column] = pd.NA
            pieces.append(missing)
            continue
        pieces.append(
            pd.merge_asof(
                group.sort_values("date"),
                changes.sort_values("industry_effective_date"),
                left_on="date",
                right_on="industry_effective_date",
                direction="backward",
            )
        )
    result = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"])
    invalid = result["industry_effective_date"].notna() & (
        result["industry_effective_date"] > result["date"]
    )
    if invalid.any():
        raise RuntimeError("检测到行业分类未来数据泄漏")
    return result.reset_index(drop=True)


def attach_membership_weight(panel: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest known official index weight without backward filling."""
    result = panel.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["benchmark_weight"] = 0.0
    result["membership_snapshot_date"] = pd.NaT
    dates = list(pd.to_datetime(history["snapshot_date"].drop_duplicates()).sort_values())
    for position, snapshot in enumerate(dates):
        until = dates[position + 1] if position + 1 < len(dates) else None
        date_mask = result["date"] >= snapshot
        if until is not None:
            date_mask &= result["date"] < until
        weights = history.loc[
            pd.to_datetime(history["snapshot_date"]) == snapshot, ["symbol", "weight"]
        ].copy()
        weights["symbol"] = weights["symbol"].astype(str).str.zfill(6)
        weight_map = weights.set_index("symbol")["weight"]
        result.loc[date_mask, "benchmark_weight"] = (
            result.loc[date_mask, "symbol"].map(weight_map).fillna(0.0).to_numpy()
        )
        result.loc[date_mask, "membership_snapshot_date"] = snapshot
    totals = result.groupby("date")["benchmark_weight"].transform("sum")
    result["benchmark_weight"] = result["benchmark_weight"].div(totals.where(totals > 0)).fillna(0)
    return result

