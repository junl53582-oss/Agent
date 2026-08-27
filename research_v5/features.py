from __future__ import annotations

import numpy as np
import pandas as pd

from research_v4.features import build_v4_dataset

MODEL_FEATURES = [
    "quality",
    "growth",
    "momentum",
    "short_reversal",
    "volume_attention",
    "low_volatility",
    "liquidity",
]

TECH = {"电子", "计算机", "通信", "传媒"}
FINANCE = {"银行", "非银金融", "房地产"}
CONSUMER = {"食品饮料", "家用电器", "商贸零售", "社会服务", "美容护理", "纺织服饰", "轻工制造"}
HEALTHCARE = {"医药生物"}
DEFENSIVE = {"公用事业", "农林牧渔", "环保"}
CYCLICAL = {
    "有色金属",
    "煤炭",
    "石油石化",
    "钢铁",
    "基础化工",
    "建筑材料",
    "机械设备",
    "汽车",
    "国防军工",
    "交通运输",
    "建筑装饰",
    "电力设备",
}


def broad_sector(industry: object) -> str:
    value = str(industry) if pd.notna(industry) else ""
    if value in TECH:
        return "technology"
    if value in FINANCE:
        return "finance_real_estate"
    if value in CONSUMER:
        return "consumer"
    if value in HEALTHCARE:
        return "healthcare"
    if value in DEFENSIVE:
        return "defensive"
    if value in CYCLICAL:
        return "cyclical_manufacturing"
    return "other"


def _rank_by_date(data: pd.DataFrame, values: pd.Series) -> pd.Series:
    return values.groupby(data["date"]).rank(pct=True, method="average").sub(0.5).fillna(0)


def build_v5_dataset(panel: pd.DataFrame) -> pd.DataFrame:
    data = build_v4_dataset(panel)
    data["momentum"] = _rank_by_date(
        data, data[["ret_20_rank", "momentum_60_rank", "ma_gap_20_rank"]].mean(axis=1)
    )
    data["short_reversal"] = _rank_by_date(
        data, -data[["ret_1_rank", "ret_5_rank"]].mean(axis=1)
    )
    data["volume_attention"] = _rank_by_date(data, data["volume_ratio_20_rank"])
    data["liquidity"] = _rank_by_date(data, data["amount_rank"])
    data["broad_sector"] = data.get("industry", pd.Series(index=data.index)).map(broad_sector)
    eligible = data["eligible"].fillna(False)
    data["market_momentum_60"] = (
        data["momentum_60"].where(eligible).groupby(data["date"]).transform("mean")
    )
    data["positive_20d_breadth"] = (
        data["ret_20"].gt(0).where(eligible).groupby(data["date"]).transform("mean")
    )
    risk_on = (data["market_momentum_60"] > 0.02) & (data["positive_20d_breadth"] > 0.55)
    risk_off = (data["market_momentum_60"] < -0.02) & (data["positive_20d_breadth"] < 0.45)
    data["regime"] = np.select([risk_on, risk_off], ["risk_on", "risk_off"], "neutral")
    data[MODEL_FEATURES] = data[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    return data
