from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from research_v15.features import EVENT_TARGETS, raw_event_years
from research_v15.text_model import event_training_masks

from .config import V18Settings


@dataclass
class EmbeddingTextModel:
    events: pd.DataFrame
    embeddings: np.ndarray
    regressors: list[Ridge]
    target_means: np.ndarray
    target_scales: np.ndarray
    training_events: int
    event_years: list[int]
    available_event_years: list[int]

    @classmethod
    def fit(
        cls,
        events: pd.DataFrame,
        embeddings: np.ndarray,
        cutoff_year: int,
        earliest_year: int,
        settings: V18Settings,
    ) -> "EmbeddingTextModel":
        available, mature = event_training_masks(events, cutoff_year, earliest_year, settings)
        indexes = np.flatnonzero(mature.to_numpy())
        if len(indexes) < 500:
            raise RuntimeError(f"{cutoff_year}没有足够的成熟V18文本事件: {len(indexes)}")
        targets = events.loc[mature, EVENT_TARGETS].to_numpy(dtype=float)
        targets = np.clip(targets, -0.20, 0.20)
        means = targets.mean(axis=0)
        scales = targets.std(axis=0, ddof=1)
        scales = np.where(scales > 1e-8, scales, 1.0)
        standardized = (targets - means) / scales
        block = embeddings[indexes]
        regressors = [
            Ridge(alpha=settings.ridge_alpha, random_state=61 + horizon).fit(
                block, standardized[:, horizon]
            )
            for horizon in range(len(EVENT_TARGETS))
        ]
        years = raw_event_years(events.loc[mature])
        return cls(
            events, embeddings, regressors, means, scales,
            len(indexes), years, raw_event_years(events.loc[available]),
        )

    def _document_scores(self, indexes: np.ndarray, settings: V18Settings) -> np.ndarray:
        block = self.embeddings[indexes]
        weights = np.asarray(settings.target_weights, dtype=float)
        horizon_scores = np.column_stack([model.predict(block) for model in self.regressors])
        return horizon_scores @ weights

    def recent_scores(self, current: pd.DataFrame, settings: V18Settings) -> pd.DataFrame:
        columns = ["symbol", "text_score", "text_events"]
        if current.empty:
            return pd.DataFrame(columns=columns)
        date = pd.Timestamp(current["date"].iloc[0])
        start = date - pd.Timedelta(days=settings.recent_event_lookback_days)
        event_dates = pd.to_datetime(self.events["date"])
        scope = current[current["eligible"].eq(True)][["symbol", "broad_sector"]].drop_duplicates("symbol")
        mask = (
            (event_dates <= date) & (event_dates >= start)
            & self.events["symbol"].isin(scope["symbol"])
            & pd.to_numeric(self.events["event_count"], errors="coerce").gt(0)
        )
        indexes = np.flatnonzero(mask.to_numpy())
        if len(indexes) == 0:
            result = current[["symbol"]].drop_duplicates().copy()
            result["text_score"] = 0.0
            result["text_events"] = 0
            return result
        events = self.events.iloc[indexes][["symbol", "date", "event_count"]].copy()
        raw = self._document_scores(indexes, settings)
        age = (date - pd.to_datetime(events["date"])).dt.days.to_numpy(dtype=float)
        decay = np.exp(-np.log(2.0) * age / settings.recent_event_half_life_days)
        events["raw_text_score"] = raw * decay
        aggregated = events.groupby("symbol", as_index=False).agg(
            raw_text_score=("raw_text_score", "sum"),
            text_events=("event_count", "sum"),
        )
        aggregated = aggregated.merge(scope, on="symbol", how="inner", validate="one_to_one")
        aggregated["text_score"] = 0.0
        active = aggregated["raw_text_score"].abs() > 1e-12
        if active.any():
            ranks = aggregated.loc[active].groupby("broad_sector")["raw_text_score"].rank(
                pct=True, method="average"
            )
            centered = ranks.groupby(aggregated.loc[active, "broad_sector"]).transform(
                lambda values: values - values.mean()
            )
            aggregated.loc[active, "text_score"] = centered.to_numpy(dtype=float)
        result = current[["symbol"]].drop_duplicates().merge(
            aggregated[["symbol", "text_score", "text_events"]], on="symbol", how="left"
        )
        result["text_score"] = result["text_score"].fillna(0.0)
        result["text_events"] = result["text_events"].fillna(0).astype(int)
        return result
