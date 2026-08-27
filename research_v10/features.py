from __future__ import annotations

import numpy as np
import pandas as pd

from research_v3.features import _cross_section_rank
from research_v9.features import V9_FEATURES, build_v9_dataset
from stockpilot.trading import add_execution_columns

from .fundamentals import EXTRA_COLUMNS


EXTRA_LEVEL_FEATURES = [f"{column}_rank" for column in EXTRA_COLUMNS]
EXTRA_CHANGE_FEATURES = [f"{column}_change_rank" for column in EXTRA_COLUMNS]
V10_FEATURES = [*V9_FEATURES, *EXTRA_LEVEL_FEATURES, *EXTRA_CHANGE_FEATURES]


def broad_sector_v10(industry: object) -> str:
    value = str(industry) if pd.notna(industry) else ""
    mappings = {
        "technology": ("电子", "半导体", "计算机", "软件", "通信", "传媒", "互联网"),
        "finance_real_estate": ("银行", "金融", "证券", "保险", "房地产"),
        "consumer": ("食品", "饮料", "家电", "零售", "旅游", "纺织", "服饰", "轻工", "美容"),
        "healthcare": ("医药", "医疗", "生物"),
        "defensive": ("公用事业", "农林", "牧渔", "环保"),
        "cyclical_manufacturing": (
            "有色", "煤炭", "石油", "钢铁", "化工", "建材", "机械", "汽车",
            "军工", "运输", "建筑", "电力设备",
        ),
    }
    for sector, keywords in mappings.items():
        if any(keyword in value for keyword in keywords):
            return sector
    return "other"


def technology_subsector(industry: object) -> str:
    value = str(industry) if pd.notna(industry) else ""
    if any(keyword in value for keyword in ("半导体", "集成电路", "元器件")):
        return "semiconductor_components"
    if any(keyword in value for keyword in ("软件", "计算机", "信息服务")):
        return "software_computing"
    if "通信" in value:
        return "communications"
    if any(keyword in value for keyword in ("传媒", "互联网", "文化")):
        return "media_internet"
    if "电子" in value:
        return "electronics"
    return "non_technology"


def _extended_changes(data: pd.DataFrame) -> pd.DataFrame:
    filings = (
        data.loc[data["available_date"].notna(), ["symbol", "available_date", *EXTRA_COLUMNS]]
        .drop_duplicates(["symbol", "available_date"], keep="last")
        .sort_values(["symbol", "available_date"])
    )
    for column in EXTRA_COLUMNS:
        current = pd.to_numeric(filings[column], errors="coerce")
        previous = current.groupby(filings["symbol"]).shift(1)
        filings[f"{column}_change"] = (current - previous) / (previous.abs() + 1.0)
    changes = [f"{column}_change" for column in EXTRA_COLUMNS]
    return data.merge(
        filings[["symbol", "available_date", *changes]],
        on=["symbol", "available_date"],
        how="left",
    )


def _residual_target(data: pd.DataFrame, target_column: str) -> pd.Series:
    result = pd.Series(np.nan, index=data.index, dtype=float)
    for indexes in data.groupby("date", sort=False).groups.values():
        group = data.loc[indexes]
        valid = group["eligible"].fillna(False) & group[target_column].notna()
        if valid.sum() < 20:
            continue
        idx = group.index[valid]
        styles = group.loc[idx, ["benchmark_weight_rank", "momentum", "low_volatility"]]
        industries = pd.get_dummies(
            group.loc[idx, "industry"].fillna("未知").astype(str), drop_first=True, dtype=float
        )
        design = np.column_stack(
            [np.ones(len(idx)), styles.to_numpy(dtype=float), industries.to_numpy(dtype=float)]
        )
        target = group.loc[idx, target_column].to_numpy(dtype=float)
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        result.loc[idx] = target - design @ coefficients
    return result


def build_v10_dataset(panel: pd.DataFrame) -> pd.DataFrame:
    data = _extended_changes(build_v9_dataset(panel))
    data["broad_sector"] = data["industry"].map(broad_sector_v10)
    data["technology_subsector"] = data["industry"].map(technology_subsector)
    negative_direction = {"interest_debt_ratio", "operating_cycle"}
    for column in EXTRA_COLUMNS:
        data[f"{column}_rank"] = _cross_section_rank(data, column)
        data[f"{column}_change_rank"] = _cross_section_rank(data, f"{column}_change")
        if column in negative_direction:
            data[f"{column}_rank"] *= -1
            data[f"{column}_change_rank"] *= -1

    ordered = data.sort_values(["symbol", "date"]).copy()
    grouped = ordered.groupby("symbol", group_keys=False)
    ordered["entry_open_20"] = grouped["open"].shift(-1)
    ordered["exit_open_20"] = grouped["open"].shift(-21)
    ordered["label_end_date_20"] = grouped["date"].shift(-21)
    ordered["future_return_20"] = ordered["exit_open_20"] / ordered["entry_open_20"] - 1

    execution_input = ordered.copy()
    execution_input["entry_open"] = execution_input["entry_open_20"]
    execution_input["exit_open"] = execution_input["exit_open_20"]
    execution = add_execution_columns(execution_input, 20)[
        [
            "date",
            "symbol",
            "entry_tradable",
            "execution_return",
            "execution_exit_open",
            "execution_exit_date",
            "exit_deferred",
        ]
    ].rename(
        columns={
            "entry_tradable": "entry_tradable_20",
            "execution_return": "execution_return_20",
            "execution_exit_open": "execution_exit_open_20",
            "execution_exit_date": "execution_exit_date_20",
            "exit_deferred": "exit_deferred_20",
        }
    )
    ordered = ordered.merge(execution, on=["date", "symbol"], how="left", validate="one_to_one")
    ordered[V10_FEATURES] = (
        ordered[V10_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    ordered["v10_target_20"] = _residual_target(ordered, "future_return_20")
    return ordered.sort_values(["date", "symbol"]).reset_index(drop=True)

