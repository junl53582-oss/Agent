from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research_v12.features import build_v12_dataset
from research_v14.features import _effective_dates, load_announcements


EVENT_TARGETS = ["event_target_1", "event_target_5", "event_target_20"]
EVENT_LABEL_ENDS = ["event_label_end_1", "event_label_end_5", "event_label_end_20"]


def _official_sector_return(data: pd.DataFrame, return_column: str) -> pd.Series:
    result = pd.Series(np.nan, index=data.index, dtype=float)
    for indexes in data.groupby(["date", "broad_sector"], sort=False).groups.values():
        group = data.loc[indexes]
        valid = group["eligible"].fillna(False) & group[return_column].notna()
        if valid.sum() < 1:
            continue
        idx = group.index[valid]
        weights = pd.to_numeric(group.loc[idx, "benchmark_weight"], errors="coerce").clip(lower=0)
        target = pd.to_numeric(group.loc[idx, return_column], errors="coerce")
        benchmark = float(np.average(target, weights=weights)) if weights.sum() > 0 else float(target.mean())
        result.loc[idx] = benchmark
    return result


def build_v15_dataset(panel: pd.DataFrame) -> pd.DataFrame:
    data = build_v12_dataset(panel).sort_values(["symbol", "date"]).copy()
    grouped = data.groupby("symbol", sort=False)
    data["event_entry_open_1"] = grouped["open"].shift(-1)
    data["event_exit_open_1"] = grouped["open"].shift(-2)
    data["event_label_end_1"] = grouped["date"].shift(-2)
    data["event_future_return_1"] = data["event_exit_open_1"] / data["event_entry_open_1"] - 1
    data["event_sector_return_1"] = _official_sector_return(data, "event_future_return_1")
    data["event_sector_return_5"] = _official_sector_return(data, "future_return_5")
    cost = pd.to_numeric(data["estimated_round_trip_cost"], errors="coerce")
    data["event_target_1"] = data["event_future_return_1"] - data["event_sector_return_1"] - cost
    data["event_target_5"] = data["future_return_5"] - data["event_sector_return_5"] - cost
    data["event_target_20"] = data["v12_net_marginal_target"]
    data["event_label_end_5"] = pd.to_datetime(data["label_end_date_5"])
    data["event_label_end_20"] = pd.to_datetime(data["label_end_date_20"])
    return data.sort_values(["date", "symbol"]).reset_index(drop=True)


def build_event_documents(
    dataset: pd.DataFrame,
    announcements: pd.DataFrame,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(dataset["date"]).drop_duplicates().sort_values())
    events = announcements[["symbol", "announcement_date", "title", "announcement_id"]].copy()
    events["effective_date"] = _effective_dates(events, dates)
    events["title"] = events["title"].fillna("").astype(str).str.strip()
    events = events.dropna(subset=["effective_date"])
    events = events[events["title"].str.len() > 0]
    events = events.drop_duplicates(["symbol", "effective_date", "title"])
    documents = (
        events.groupby(["symbol", "effective_date"], sort=True)
        .agg(
            document=("title", lambda values: "。".join(values.astype(str))),
            event_count=("announcement_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"effective_date": "date"})
    )
    labels = dataset[
        [
            "symbol", "date", "eligible", "broad_sector", "benchmark_weight",
            *EVENT_TARGETS, *EVENT_LABEL_ENDS,
        ]
    ].copy()
    labels["date"] = pd.to_datetime(labels["date"])
    result = documents.merge(labels, on=["symbol", "date"], how="inner", validate="one_to_one")
    result["symbol"] = result["symbol"].astype(str).str.zfill(6)
    return result.sort_values(["date", "symbol"]).reset_index(drop=True)


def save_event_documents(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, encoding="utf-8-sig")


def load_event_documents(path: str | Path = "data/event_documents_pit_v15.csv") -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"symbol": str})
    data["symbol"] = data["symbol"].astype(str).str.zfill(6)
    data["date"] = pd.to_datetime(data["date"], format="mixed", errors="raise")
    for column in EVENT_LABEL_ENDS:
        data[column] = pd.to_datetime(data[column], format="mixed", errors="coerce")
    return data.sort_values(["date", "symbol"]).reset_index(drop=True)


def raw_event_years(events: pd.DataFrame, cutoff: pd.Timestamp | None = None) -> list[int]:
    data = events[pd.to_numeric(events["event_count"], errors="coerce").fillna(0) > 0]
    if cutoff is not None:
        data = data[pd.to_datetime(data["date"]) < cutoff]
    return sorted(pd.to_datetime(data["date"], errors="coerce").dropna().dt.year.unique().astype(int).tolist())
