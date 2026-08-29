from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


LEDGER_COLUMNS = [
    "prediction_date", "symbol", "horizon", "predicted_probability", "expected_return",
    "entry_date", "exit_date", "entry_open", "exit_open", "actual_return",
    "actual_direction", "correct_direction", "settled",
]


def settle_predictions(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    *,
    direction_thresholds: dict[int, float] | None = None,
) -> pd.DataFrame:
    thresholds = direction_thresholds or {1: 0.0, 5: 0.0, 20: 0.0}
    prices = market[["date", "symbol", "open"]].copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices["symbol"] = prices["symbol"].astype(str).str.zfill(6)
    calendar = pd.DatetimeIndex(prices["date"].drop_duplicates().sort_values())
    open_lookup = prices.set_index(["date", "symbol"])["open"]
    rows: list[dict] = []
    for prediction in predictions.itertuples(index=False):
        prediction_date = pd.Timestamp(prediction.date).normalize()
        position = calendar.get_indexer([prediction_date])[0]
        for horizon in (1, 5, 20):
            entry_position, exit_position = position + 1, position + horizon + 1
            entry_date = calendar[entry_position] if 0 <= entry_position < len(calendar) else pd.NaT
            exit_date = calendar[exit_position] if 0 <= exit_position < len(calendar) else pd.NaT
            entry_open = open_lookup.get((entry_date, str(prediction.symbol).zfill(6)), np.nan) if pd.notna(entry_date) else np.nan
            exit_open = open_lookup.get((exit_date, str(prediction.symbol).zfill(6)), np.nan) if pd.notna(exit_date) else np.nan
            actual_return = exit_open / entry_open - 1 if np.isfinite(entry_open) and np.isfinite(exit_open) and entry_open != 0 else np.nan
            settled = bool(np.isfinite(actual_return))
            actual_direction = bool(actual_return > thresholds.get(horizon, 0.0)) if settled else pd.NA
            predicted_probability = float(getattr(prediction, f"p_up_{horizon}d"))
            expected_return = float(getattr(prediction, f"expected_return_{horizon}d", np.nan))
            rows.append({
                "prediction_date": prediction_date,
                "symbol": str(prediction.symbol).zfill(6),
                "horizon": horizon,
                "predicted_probability": predicted_probability,
                "expected_return": expected_return,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_open": entry_open,
                "exit_open": exit_open,
                "actual_return": actual_return,
                "actual_direction": actual_direction,
                "correct_direction": bool((predicted_probability >= 0.5) == actual_direction) if settled else pd.NA,
                "settled": settled,
            })
    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def update_prediction_ledger(
    snapshot_dir: Path,
    market: pd.DataFrame,
    ledger_path: Path,
    *,
    direction_thresholds: dict[int, float] | None = None,
) -> pd.DataFrame:
    snapshots = [pd.read_csv(path, dtype={"symbol": str}) for path in sorted(snapshot_dir.glob("????-??-??.csv"))]
    if not snapshots:
        ledger = pd.DataFrame(columns=LEDGER_COLUMNS)
    else:
        ledger = settle_predictions(pd.concat(snapshots, ignore_index=True), market, direction_thresholds=direction_thresholds)
        ledger = ledger.sort_values(["prediction_date", "symbol", "horizon"]).drop_duplicates(
            ["prediction_date", "symbol", "horizon"], keep="last"
        )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(ledger_path, index=False, encoding="utf-8-sig")
    return ledger

