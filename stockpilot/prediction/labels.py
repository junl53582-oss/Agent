from __future__ import annotations

import numpy as np
import pandas as pd


def add_prediction_labels(
    frame: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 20),
    thresholds: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Add market-calendar aligned next-open to future-open labels.

    A decision at T close enters at T+1 open and exits at T+H+1 open. Missing
    symbol opens (for example suspension) remain missing rather than shifting to
    the next available symbol observation.
    """
    thresholds = thresholds or {horizon: 0.0 for horizon in horizons}
    required = {"date", "symbol", "open"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction label input missing columns: {sorted(missing)}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise").dt.normalize()
    data["symbol"] = data["symbol"].astype(str).str.zfill(6)
    if data.duplicated(["date", "symbol"]).any():
        raise ValueError("prediction labels require unique date/symbol keys")
    calendar = pd.DatetimeIndex(data["date"].drop_duplicates().sort_values())
    symbols = pd.Index(data["symbol"].drop_duplicates().sort_values())
    date_positions = pd.Series(np.arange(len(calendar)), index=calendar)
    symbol_positions = pd.Series(np.arange(len(symbols)), index=symbols)
    row_dates = data["date"].map(date_positions).to_numpy(dtype=int)
    row_symbols = data["symbol"].map(symbol_positions).to_numpy(dtype=int)
    opens = data.pivot(index="date", columns="symbol", values="open").reindex(index=calendar, columns=symbols)
    open_values = opens.to_numpy(dtype=float)

    entry_positions = row_dates + 1
    valid_entry = entry_positions < len(calendar)
    entry_open = np.full(len(data), np.nan)
    entry_open[valid_entry] = open_values[entry_positions[valid_entry], row_symbols[valid_entry]]
    entry_dates = np.full(len(data), np.datetime64("NaT"), dtype="datetime64[ns]")
    entry_dates[valid_entry] = calendar.to_numpy()[entry_positions[valid_entry]]
    data["entry_date"] = pd.to_datetime(entry_dates)
    data["prediction_entry_open"] = entry_open

    for horizon in horizons:
        if horizon < 1:
            raise ValueError("prediction horizon must be positive")
        exit_positions = row_dates + horizon + 1
        valid_exit = exit_positions < len(calendar)
        exit_open = np.full(len(data), np.nan)
        exit_open[valid_exit] = open_values[exit_positions[valid_exit], row_symbols[valid_exit]]
        label_dates = np.full(len(data), np.datetime64("NaT"), dtype="datetime64[ns]")
        label_dates[valid_exit] = calendar.to_numpy()[exit_positions[valid_exit]]
        returns = exit_open / entry_open - 1.0
        returns[~np.isfinite(returns)] = np.nan
        data[f"label_end_date_{horizon}d"] = pd.to_datetime(label_dates)
        data[f"future_return_{horizon}d"] = returns
        data[f"raw_up_{horizon}d"] = pd.Series(returns, index=data.index).gt(0).where(np.isfinite(returns))
        threshold = float(thresholds.get(horizon, 0.0))
        data[f"tradable_up_{horizon}d"] = pd.Series(returns, index=data.index).gt(threshold).where(np.isfinite(returns))
    return data


def mature_training_view(
    frame: pd.DataFrame,
    horizon: int,
    validation_start: pd.Timestamp,
    purge_cutoff: pd.Timestamp,
    earliest_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    label_end = f"label_end_date_{horizon}d"
    target = f"tradable_up_{horizon}d"
    mask = (
        frame["eligible"].fillna(False)
        & frame[target].notna()
        & pd.to_datetime(frame[label_end]).lt(pd.Timestamp(validation_start))
        & pd.to_datetime(frame["date"]).le(pd.Timestamp(purge_cutoff))
    )
    if earliest_date is not None:
        mask &= pd.to_datetime(frame["date"]).ge(pd.Timestamp(earliest_date))
    return frame.loc[mask].sort_values(["date", "symbol"])
