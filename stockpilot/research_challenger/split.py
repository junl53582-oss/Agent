from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ResearchFold:
    test_year: int
    horizon: int
    train_index: pd.Index
    validation_index: pd.Index
    refit_index: pd.Index
    test_index: pd.Index
    train_start: pd.Timestamp
    validation_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purge_gap_trading_days: int


def _decision_cutoff(
    dates: pd.Series, boundary: pd.Timestamp, gap: int
) -> pd.Timestamp:
    prior = pd.DatetimeIndex(dates[dates < boundary].drop_duplicates().sort_values())
    if len(prior) <= gap:
        raise ValueError("insufficient dates for purge gap")
    return pd.Timestamp(prior[-(gap + 1)])


def build_fold(
    frame: pd.DataFrame,
    test_year: int,
    horizon: int,
    *,
    training_window_years: int,
    validation_years: int,
    purge_gap_trading_days: int,
) -> ResearchFold:
    if validation_years != 1:
        raise ValueError("V31 protocol freezes exactly one validation year")
    date = pd.to_datetime(frame["date"])
    label_end = pd.to_datetime(frame[f"label_end_date_{horizon}d"])
    validation_start = pd.Timestamp(test_year - validation_years, 1, 1)
    test_start = pd.Timestamp(test_year, 1, 1)
    test_end = pd.Timestamp(test_year, 12, 31)
    train_start = pd.Timestamp(test_year - training_window_years - validation_years, 1, 1)
    inner_cutoff = _decision_cutoff(date, validation_start, purge_gap_trading_days)
    refit_cutoff = _decision_cutoff(date, test_start, purge_gap_trading_days)
    train_mask = (
        date.ge(train_start)
        & date.le(inner_cutoff)
        & label_end.lt(validation_start)
        & label_end.notna()
    )
    validation_mask = (
        date.ge(validation_start)
        & date.lt(test_start)
        & date.le(refit_cutoff)
        & label_end.lt(test_start)
        & label_end.notna()
    )
    refit_mask = (
        date.ge(train_start)
        & date.le(refit_cutoff)
        & label_end.lt(test_start)
        & label_end.notna()
    )
    test_mask = date.ge(test_start) & date.le(test_end) & label_end.notna()
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise RuntimeError(f"incomplete V31 fold: year={test_year}, horizon={horizon}")
    if label_end[train_mask].max() >= validation_start:
        raise AssertionError("training label crosses validation boundary")
    if label_end[refit_mask].max() >= test_start:
        raise AssertionError("refit label crosses OOS boundary")
    if date[train_mask].max() >= date[validation_mask].min():
        raise AssertionError("train/purge/validation ordering failed")
    if date[validation_mask].max() >= date[test_mask].min():
        raise AssertionError("validation/embargo/OOS ordering failed")
    return ResearchFold(
        test_year=test_year,
        horizon=horizon,
        train_index=frame.index[train_mask],
        validation_index=frame.index[validation_mask],
        refit_index=frame.index[refit_mask],
        test_index=frame.index[test_mask],
        train_start=train_start,
        validation_start=validation_start,
        test_start=test_start,
        test_end=test_end,
        purge_gap_trading_days=purge_gap_trading_days,
    )


def fold_receipt(frame: pd.DataFrame, fold: ResearchFold) -> dict:
    label_end = f"label_end_date_{fold.horizon}d"
    train = frame.loc[fold.train_index]
    validation = frame.loc[fold.validation_index]
    refit = frame.loc[fold.refit_index]
    test = frame.loc[fold.test_index]
    return {
        "test_year": fold.test_year,
        "horizon": fold.horizon,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "refit_rows": int(len(refit)),
        "oos_rows": int(len(test)),
        "train_start": str(train["date"].min().date()),
        "train_decision_end": str(train["date"].max().date()),
        "train_label_end": str(pd.to_datetime(train[label_end]).max().date()),
        "validation_start": str(validation["date"].min().date()),
        "validation_decision_end": str(validation["date"].max().date()),
        "validation_label_end": str(pd.to_datetime(validation[label_end]).max().date()),
        "oos_start": str(test["date"].min().date()),
        "oos_end": str(test["date"].max().date()),
        "purge_gap_trading_days": fold.purge_gap_trading_days,
        "train_label_maturity_verified": bool(
            pd.to_datetime(train[label_end]).max() < fold.validation_start
        ),
        "refit_label_maturity_verified": bool(
            pd.to_datetime(refit[label_end]).max() < fold.test_start
        ),
        "random_split_used": False,
    }
