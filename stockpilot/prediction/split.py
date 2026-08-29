from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PurgedFold:
    validation_year: int
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    purge_gap_trading_days: int
    purge_cutoff: pd.Timestamp
    train_index: pd.Index
    validation_index: pd.Index


class PurgedWalkForwardSplit:
    def __init__(self, years: tuple[int, ...], gap_trading_days: int, training_window_years: int = 8, *, shuffle: bool = False):
        if shuffle:
            raise ValueError("random or shuffled time-series split is forbidden")
        if gap_trading_days < 1:
            raise ValueError("purge gap must be positive")
        self.years = tuple(int(year) for year in years)
        self.gap = int(gap_trading_days)
        self.training_window_years = int(training_window_years)

    def split(self, frame: pd.DataFrame, horizon: int) -> list[PurgedFold]:
        dates = pd.to_datetime(frame["date"])
        calendar = pd.DatetimeIndex(dates.drop_duplicates().sort_values())
        label_end = pd.to_datetime(frame[f"label_end_date_{horizon}d"])
        folds: list[PurgedFold] = []
        for year in self.years:
            validation_mask = dates.dt.year.eq(year) & frame["eligible"].fillna(False) & frame[f"tradable_up_{horizon}d"].notna()
            if not validation_mask.any():
                continue
            validation_start = dates[validation_mask].min()
            validation_end = dates[validation_mask].max()
            start_position = int(calendar.get_indexer([validation_start])[0])
            if start_position < self.gap:
                continue
            purge_cutoff = calendar[start_position - self.gap]
            earliest = pd.Timestamp(year - self.training_window_years, 1, 1)
            train_mask = (
                frame["eligible"].fillna(False)
                & frame[f"tradable_up_{horizon}d"].notna()
                & label_end.lt(validation_start)
                & dates.le(purge_cutoff)
                & dates.ge(earliest)
            )
            if train_mask.any() and label_end[train_mask].max() >= validation_start:
                raise AssertionError("training label crosses validation boundary")
            folds.append(PurgedFold(year, validation_start, validation_end, self.gap, purge_cutoff,
                                    frame.index[train_mask], frame.index[validation_mask]))
        return folds
