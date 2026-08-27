from __future__ import annotations

import numpy as np
import pandas as pd

from research_v10.features import V10_FEATURES
from research_v12.features import build_v12_dataset


EVENT_GROUPS = {
    "positive": ("回购", "增持", "中标", "预增", "扭亏", "分红", "股权激励", "重大合同"),
    "negative": ("减持", "立案", "处罚", "诉讼", "风险提示", "预亏", "预减", "退市", "违规", "终止"),
    "results": ("年度报告", "半年度报告", "季度报告", "业绩预告", "业绩快报"),
    "financing": ("定增", "增发", "可转债", "融资", "担保", "质押"),
    "governance": ("董事", "监事", "高管", "股东大会"),
}
WINDOWS = (5, 20, 60)
ANNOUNCEMENT_FEATURES = [
    *[f"announcement_count_{window}_rank" for window in WINDOWS],
    *[
        f"announcement_{group}_{window}_rank"
        for group in EVENT_GROUPS
        for window in (20, 60)
    ],
    "announcement_net_sentiment_20_rank",
    "announcement_net_sentiment_60_rank",
    "announcement_recency_rank",
]
V14_FEATURES = [*V10_FEATURES, *ANNOUNCEMENT_FEATURES]


def load_announcements(path="data/announcements_pit_v14.csv") -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"symbol": str, "announcement_id": str})
    data["symbol"] = data["symbol"].astype(str).str.zfill(6)
    data["announcement_date"] = pd.to_datetime(
        data["announcement_date"], format="mixed", errors="coerce"
    )
    return data.dropna(subset=["announcement_date"]).drop_duplicates(["symbol", "announcement_id"])


def _effective_dates(events: pd.DataFrame, trading_dates: pd.DatetimeIndex) -> pd.Series:
    # Publication time is not available, so same-day close cannot use the title.
    safe_date = events["announcement_date"].dt.normalize() + pd.Timedelta(days=1)
    indexes = trading_dates.searchsorted(safe_date, side="left")
    valid = indexes < len(trading_dates)
    result = pd.Series(pd.NaT, index=events.index, dtype="datetime64[ns]")
    result.loc[valid] = trading_dates[indexes[valid]].to_numpy()
    return result


def attach_announcement_features(
    panel: pd.DataFrame,
    announcements: pd.DataFrame,
    keep_intermediate: bool = True,
) -> pd.DataFrame:
    data = panel.sort_values(["symbol", "date"]).copy()
    data["date"] = pd.to_datetime(data["date"])
    events = announcements.copy()
    events["effective_date"] = _effective_dates(
        events, pd.DatetimeIndex(data["date"].drop_duplicates().sort_values())
    )
    title = events["title"].fillna("").astype(str)
    events["announcement_count"] = 1.0
    for group, keywords in EVENT_GROUPS.items():
        events[f"announcement_{group}"] = title.map(
            lambda value: float(any(keyword in value for keyword in keywords))
        )
    value_columns = ["announcement_count", *[f"announcement_{name}" for name in EVENT_GROUPS]]
    daily = (
        events.dropna(subset=["effective_date"])
        .groupby(["symbol", "effective_date"], as_index=False)[value_columns]
        .sum()
        .rename(columns={"effective_date": "date"})
    )
    data = data.merge(daily, on=["symbol", "date"], how="left", validate="one_to_one")
    data[value_columns] = data[value_columns].fillna(0.0)
    symbol_groups = data["symbol"]

    def rolling_sum(source: str, window: int) -> pd.Series:
        cumulative = data.groupby("symbol", sort=False)[source].cumsum()
        lagged = cumulative.groupby(symbol_groups, sort=False).shift(window).fillna(0.0)
        return cumulative - lagged

    raw_features = []
    for window in WINDOWS:
        name = f"announcement_count_{window}"
        data[name] = rolling_sum("announcement_count", window)
        raw_features.append(name)
    for group in EVENT_GROUPS:
        for window in (20, 60):
            name = f"announcement_{group}_{window}"
            source = f"announcement_{group}"
            data[name] = rolling_sum(source, window)
            raw_features.append(name)
    for window in (20, 60):
        name = f"announcement_net_sentiment_{window}"
        data[name] = data[f"announcement_positive_{window}"] - data[f"announcement_negative_{window}"]
        raw_features.append(name)
    event_date = data["date"].where(data["announcement_count"] > 0).groupby(data["symbol"]).ffill()
    data["announcement_recency"] = (data["date"] - event_date).dt.days.fillna(9999).clip(upper=9999)
    for name in raw_features:
        data[f"{name}_rank"] = data.groupby("date")[name].rank(pct=True, method="average").sub(0.5).fillna(0)
    data["announcement_recency_rank"] = (
        (-data["announcement_recency"]).groupby(data["date"]).rank(pct=True, method="average").sub(0.5).fillna(0)
    )
    data[ANNOUNCEMENT_FEATURES] = (
        data[ANNOUNCEMENT_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
    )
    if not keep_intermediate:
        intermediate = list(dict.fromkeys([*value_columns, *raw_features, "announcement_recency"]))
        data = data.drop(columns=intermediate)
    return data


def build_v14_dataset(panel: pd.DataFrame, announcements: pd.DataFrame) -> pd.DataFrame:
    return attach_announcement_features(
        build_v12_dataset(panel), announcements, keep_intermediate=False
    )
