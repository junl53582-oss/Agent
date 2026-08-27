from __future__ import annotations

import numpy as np
import pandas as pd

from research_v4.stability import FactorSpec, score_with_specs
from research_v5.models import V5Models, score_v5

from .config import V6Settings


def _rank(values: pd.Series) -> pd.Series:
    return values.rank(pct=True, method="average").sub(0.5).fillna(0)


def score_v6(
    current: pd.DataFrame,
    v5_models: V5Models,
    v4_specs: list[FactorSpec],
    settings: V6Settings | None = None,
) -> pd.DataFrame:
    settings = settings or V6Settings()
    scored = score_v5(current, v5_models)
    scored["v5_rank"] = _rank(scored["score"])
    v4_raw = score_with_specs(scored, v4_specs)
    scored["v4_rank"] = _rank(v4_raw)
    scored["sector_rank"] = scored.groupby("broad_sector")["score"].transform(_rank)
    scored["score"] = (
        settings.v5_weight * scored["v5_rank"]
        + settings.v4_weight * scored["v4_rank"]
        + settings.sector_rank_weight * scored["sector_rank"]
    )
    return scored


def _sector_quotas(current: pd.DataFrame, top_n: int) -> dict[str, int]:
    counts = current.groupby("broad_sector").size().sort_index()
    target = min(top_n, len(current))
    raw = counts / counts.sum() * target
    quotas = np.floor(raw).astype(int).clip(lower=1)
    quotas = quotas.clip(upper=counts)
    while quotas.sum() < target:
        available = counts[counts > quotas]
        if available.empty:
            break
        priorities = (raw - quotas).loc[available.index]
        quotas.loc[priorities.idxmax()] += 1
    while quotas.sum() > target:
        removable = quotas[quotas > 1]
        if removable.empty:
            break
        priorities = (quotas - raw).loc[removable.index]
        quotas.loc[priorities.idxmax()] -= 1
    return {str(sector): int(value) for sector, value in quotas.items()}


def select_sector_balanced(
    current: pd.DataFrame, settings: V6Settings | None = None
) -> pd.DataFrame:
    settings = settings or V6Settings()
    quotas = _sector_quotas(current, settings.top_n)
    pieces = []
    sector_counts = current.groupby("broad_sector").size()
    total_count = sector_counts.sum()
    for sector, quota in quotas.items():
        selected = current[current["broad_sector"] == sector].nlargest(quota, "score").copy()
        inverse_vol = 1 / selected["volatility_20"].clip(lower=1e-6)
        within = inverse_vol / inverse_vol.sum()
        selected["weight"] = within * (sector_counts.loc[sector] / total_count)
        pieces.append(selected)
    result = pd.concat(pieces, ignore_index=False) if pieces else current.iloc[0:0].copy()
    if len(result) < settings.min_positions:
        return result.iloc[0:0].copy()
    result["weight"] /= result["weight"].sum()
    return result.sort_values("score", ascending=False)
