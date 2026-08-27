from __future__ import annotations

from dataclasses import dataclass

import jieba
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDRegressor

from research_v15.features import EVENT_TARGETS, raw_event_years
from research_v15.text_model import event_training_masks

from .config import V16Settings


jieba.setLogLevel(60)

CHAR_HEAD_SEED = 51
WORD_HEAD_SEED = 151


def _word_tokens(text: str) -> list[str]:
    return [token for token in jieba.lcut(text) if token.strip()]


@dataclass
class EnsembleTextCorpus:
    events: pd.DataFrame
    char_matrix: sparse.csr_matrix
    word_matrix: sparse.csr_matrix

    @classmethod
    def build(cls, events: pd.DataFrame, settings: V16Settings) -> "EnsembleTextCorpus":
        ordered = events.sort_values(["date", "symbol"]).reset_index(drop=True)
        documents = ordered["document"].fillna("").astype(str)
        char_vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            n_features=settings.text_n_features,
            alternate_sign=False,
            norm="l2",
            lowercase=False,
            dtype=np.float32,
        )
        word_vectorizer = HashingVectorizer(
            analyzer=_word_tokens,
            ngram_range=(1, 2),
            n_features=settings.word_n_features,
            alternate_sign=False,
            norm="l2",
            lowercase=False,
            dtype=np.float32,
        )
        return cls(
            ordered,
            char_vectorizer.transform(documents).tocsr(),
            word_vectorizer.transform(documents).tocsr(),
        )


def _fit_regressor(
    matrix: sparse.csr_matrix, target: np.ndarray, seed: int, settings: V16Settings
) -> SGDRegressor:
    model = SGDRegressor(
        loss="huber",
        epsilon=1.35,
        penalty="elasticnet",
        alpha=settings.text_alpha,
        l1_ratio=settings.text_l1_ratio,
        max_iter=settings.text_max_iter,
        tol=None,
        shuffle=False,
        random_state=seed,
        average=True,
    )
    model.fit(matrix, target)
    return model


@dataclass
class EnsembleTextModel:
    corpus: EnsembleTextCorpus
    char_regressors: list[SGDRegressor]
    word_regressors: list[SGDRegressor]
    target_means: np.ndarray
    target_scales: np.ndarray
    training_events: int
    event_years: list[int]
    available_event_years: list[int]

    @classmethod
    def fit(
        cls,
        corpus: EnsembleTextCorpus,
        cutoff_year: int,
        earliest_year: int,
        settings: V16Settings,
    ) -> "EnsembleTextModel":
        events = corpus.events
        available, mature = event_training_masks(events, cutoff_year, earliest_year, settings)
        indexes = np.flatnonzero(mature.to_numpy())
        if len(indexes) < 500:
            raise RuntimeError(f"{cutoff_year}没有足够的成熟V16文本事件: {len(indexes)}")
        targets = events.loc[mature, EVENT_TARGETS].to_numpy(dtype=float)
        targets = np.clip(targets, -0.20, 0.20)
        means = targets.mean(axis=0)
        scales = targets.std(axis=0, ddof=1)
        scales = np.where(scales > 1e-8, scales, 1.0)
        standardized = (targets - means) / scales
        char_matrix = corpus.char_matrix[indexes]
        word_matrix = corpus.word_matrix[indexes]
        char_regressors = [
            _fit_regressor(char_matrix, standardized[:, horizon], CHAR_HEAD_SEED + horizon, settings)
            for horizon in range(len(EVENT_TARGETS))
        ]
        word_regressors = [
            _fit_regressor(word_matrix, standardized[:, horizon], WORD_HEAD_SEED + horizon, settings)
            for horizon in range(len(EVENT_TARGETS))
        ]
        years = raw_event_years(events.loc[mature])
        return cls(
            corpus,
            char_regressors,
            word_regressors,
            means,
            scales,
            len(indexes),
            years,
            raw_event_years(events.loc[available]),
        )

    def _head_scores(self, indexes: np.ndarray, regressors, matrix) -> np.ndarray:
        block = matrix[indexes]
        return np.column_stack([model.predict(block) for model in regressors])

    def _char_scores(self, indexes: np.ndarray) -> np.ndarray:
        return self._head_scores(indexes, self.char_regressors, self.corpus.char_matrix)

    def _word_scores(self, indexes: np.ndarray) -> np.ndarray:
        return self._head_scores(indexes, self.word_regressors, self.corpus.word_matrix)

    def _document_scores(self, indexes: np.ndarray, settings: V16Settings) -> tuple[np.ndarray, np.ndarray]:
        weights = np.asarray(settings.target_weights, dtype=float)
        char = self._char_scores(indexes) @ weights
        word = self._word_scores(indexes) @ weights
        ensemble = settings.char_word_blend * char + (1.0 - settings.char_word_blend) * word
        return char, ensemble

    def recent_scores(self, current: pd.DataFrame, settings: V16Settings) -> pd.DataFrame:
        columns = ["symbol", "text_score", "char_text_score", "text_events"]
        if current.empty:
            return pd.DataFrame(columns=columns)
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
            result["char_text_score"] = 0.0
            result["text_events"] = 0
            return result
        events = self.corpus.events.iloc[indexes][["symbol", "date", "event_count"]].copy()
        char_blend, ensemble_blend = self._document_scores(indexes, settings)
        age = (date - pd.to_datetime(events["date"])).dt.days.to_numpy(dtype=float)
        decay = np.exp(-np.log(2.0) * age / settings.recent_event_half_life_days)
        events["raw_char_score"] = char_blend * decay
        events["raw_text_score"] = ensemble_blend * decay
        aggregated = events.groupby("symbol", as_index=False).agg(
            raw_char_score=("raw_char_score", "sum"),
            raw_text_score=("raw_text_score", "sum"),
            text_events=("event_count", "sum"),
        )
        aggregated = aggregated.merge(scope, on="symbol", how="inner", validate="one_to_one")
        for raw_column, output_column in (
            ("raw_char_score", "char_text_score"),
            ("raw_text_score", "text_score"),
        ):
            aggregated[output_column] = 0.0
            active = aggregated[raw_column].abs() > 1e-12
            if active.any():
                ranks = aggregated.loc[active].groupby("broad_sector")[raw_column].rank(
                    pct=True, method="average"
                )
                centered = ranks.groupby(aggregated.loc[active, "broad_sector"]).transform(
                    lambda values: values - values.mean()
                )
                aggregated.loc[active, output_column] = centered.to_numpy(dtype=float)
        result = current[["symbol"]].drop_duplicates().merge(
            aggregated[["symbol", "text_score", "char_text_score", "text_events"]],
            on="symbol",
            how="left",
        )
        result["text_score"] = result["text_score"].fillna(0.0)
        result["char_text_score"] = result["char_text_score"].fillna(0.0)
        result["text_events"] = result["text_events"].fillna(0).astype(int)
        return result
