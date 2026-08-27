from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDRegressor

from .config import V15Settings
from .features import EVENT_LABEL_ENDS, EVENT_TARGETS, raw_event_years


def event_training_masks(events, cutoff_year, earliest_year, settings):
    cutoff = pd.Timestamp(cutoff_year, 1, 1)
    embargo = cutoff - pd.Timedelta(days=settings.embargo_calendar_days)
    dates = pd.to_datetime(events["date"])
    available = (
        events["eligible"].eq(True)
        & pd.to_numeric(events["event_count"], errors="coerce").gt(0)
        & (dates.dt.year >= earliest_year) & (dates < cutoff)
    )
    ends = events[EVENT_LABEL_ENDS].apply(pd.to_datetime)
    mature = available & np.isfinite(events[EVENT_TARGETS]).all(axis=1)
    mature &= ends.lt(embargo).all(axis=1) & ends.gt(dates, axis=0).all(axis=1)
    return available, mature


@dataclass
class EventTextCorpus:
    events: pd.DataFrame
    matrix: sparse.csr_matrix
    vectorizer: HashingVectorizer

    @classmethod
    def build(cls, events: pd.DataFrame, settings: V15Settings) -> "EventTextCorpus":
        vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            n_features=settings.text_n_features,
            alternate_sign=False,
            norm="l2",
            lowercase=False,
            dtype=np.float32,
        )
        ordered = events.sort_values(["date", "symbol"]).reset_index(drop=True)
        matrix = vectorizer.transform(ordered["document"].fillna("").astype(str)).tocsr()
        return cls(ordered, matrix, vectorizer)


@dataclass
class MultiHorizonTextModel:
    corpus: EventTextCorpus
    regressors: list[SGDRegressor]
    target_means: np.ndarray
    target_scales: np.ndarray
    training_events: int
    event_years: list[int]
    available_event_years: list[int]

    @classmethod
    def fit(
        cls,
        corpus: EventTextCorpus,
        cutoff_year: int,
        earliest_year: int,
        settings: V15Settings,
    ) -> "MultiHorizonTextModel":
        events = corpus.events
        available, mature = event_training_masks(events, cutoff_year, earliest_year, settings)
        indexes = np.flatnonzero(mature.to_numpy())
        if len(indexes) < 500:
            raise RuntimeError(f"{cutoff_year}没有足够的成熟V15文本事件: {len(indexes)}")
        targets = events.loc[mature, EVENT_TARGETS].to_numpy(dtype=float)
        targets = np.clip(targets, -0.20, 0.20)
        means = targets.mean(axis=0)
        scales = targets.std(axis=0, ddof=1)
        scales = np.where(scales > 1e-8, scales, 1.0)
        standardized = (targets - means) / scales
        matrix = corpus.matrix[indexes]
        regressors = []
        for horizon_index in range(len(EVENT_TARGETS)):
            model = SGDRegressor(
                loss="huber",
                epsilon=1.35,
                penalty="elasticnet",
                alpha=settings.text_alpha,
                l1_ratio=settings.text_l1_ratio,
                max_iter=settings.text_max_iter,
                tol=None,
                shuffle=False,
                random_state=51 + horizon_index,
                average=True,
            )
            model.fit(matrix, standardized[:, horizon_index])
            regressors.append(model)
        years = raw_event_years(events.loc[mature])
        return cls(corpus, regressors, means, scales, len(indexes), years, raw_event_years(events.loc[available]))

    def _document_scores(self, indexes: np.ndarray, settings: V15Settings) -> np.ndarray:
        if len(indexes) == 0:
            return np.empty((0, len(self.regressors)), dtype=float)
        matrix = self.corpus.matrix[indexes]
        return np.column_stack([model.predict(matrix) for model in self.regressors])

    def recent_scores(self, current: pd.DataFrame, settings: V15Settings) -> pd.DataFrame:
        if current.empty:
            return pd.DataFrame(columns=["symbol", "text_score", "text_events"])
        date = pd.Timestamp(current["date"].iloc[0])
        start = date - pd.Timedelta(days=settings.recent_event_lookback_days)
        event_dates = pd.to_datetime(self.corpus.events["date"])
        scope = current[current["eligible"].eq(True)][["symbol", "broad_sector"]].drop_duplicates("symbol")
        mask = (
            (event_dates <= date) & (event_dates >= start)
            & self.corpus.events["symbol"].isin(scope["symbol"])
            & pd.to_numeric(self.corpus.events["event_count"], errors="coerce").gt(0)
        )
        indexes = np.flatnonzero(mask.to_numpy())
        if len(indexes) == 0:
            result = current[["symbol"]].drop_duplicates().copy()
            result["text_score"] = 0.0
            result["text_events"] = 0
            return result
        events = self.corpus.events.iloc[indexes][["symbol", "date", "event_count"]].copy()
        horizon_scores = self._document_scores(indexes, settings)
        blend = horizon_scores @ np.asarray(settings.target_weights, dtype=float)
        age = (date - pd.to_datetime(events["date"])).dt.days.to_numpy(dtype=float)
        decay = np.exp(-np.log(2.0) * age / settings.recent_event_half_life_days)
        events["raw_text_score"] = blend * decay
        aggregated = events.groupby("symbol", as_index=False).agg(
            raw_text_score=("raw_text_score", "sum"),
            text_events=("event_count", "sum"),
        )
        aggregated = aggregated.merge(scope, on="symbol", how="inner", validate="one_to_one")
        active = aggregated["raw_text_score"].abs() > 1e-12
        aggregated["text_score"] = 0.0
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
