"""Build the small deterministic replay bundle used by sandbox E2E tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research_v10.features import V10_FEATURES
from stockpilot.daily_pit.pipeline import DAILY_FEATURE_COLUMNS
from stockpilot.prospective_r2.calendar import load_verified_calendar
from stockpilot.prospective_r2.integrity import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    write_immutable_json,
)
from stockpilot.research_challenger.prospective_gen2 import CALENDAR_PATH

TARGET = "2026-09-02"
SYMBOLS = [f"{number:06d}" for number in range(1, 11)]


def _historical_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2017-01-02", "2025-12-31")
    date_values = np.repeat(dates.to_numpy(), len(SYMBOLS))
    symbol_values = np.tile(np.asarray(SYMBOLS), len(dates))
    symbol_rank = np.tile(np.arange(1, len(SYMBOLS) + 1, dtype=float), len(dates))
    day_signal = np.repeat(np.arange(len(dates), dtype=float) % 17 / 1000, len(SYMBOLS))
    base = symbol_rank / len(SYMBOLS) + day_signal
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(date_values),
            "symbol": symbol_values,
            "broad_sector": np.tile([f"S{i % 3}" for i in range(len(SYMBOLS))], len(dates)),
            "industry": np.tile([f"I{i % 4}" for i in range(len(SYMBOLS))], len(dates)),
            "benchmark_weight": 1.0 / len(SYMBOLS),
            "benchmark_weight_rank": symbol_rank / len(SYMBOLS),
            "future_return_5d": base / 100,
            "future_return_20d": base / 50,
            "label_end_date_5d": pd.to_datetime(date_values) + pd.offsets.BDay(6),
            "label_end_date_20d": pd.to_datetime(date_values) + pd.offsets.BDay(21),
        }
    )
    for index, feature in enumerate(V10_FEATURES):
        if feature not in frame:
            frame[feature] = base * (1.0 + index / 1000)
    return frame


def _daily_panel() -> pd.DataFrame:
    target = pd.Timestamp(TARGET)
    values: dict[str, object] = {
        "date": [target] * len(SYMBOLS),
        "symbol": SYMBOLS,
        "eligible": [True] * len(SYMBOLS),
        "in_universe": [True] * len(SYMBOLS),
        "membership_snapshot_date": [pd.Timestamp("2026-08-31")] * len(SYMBOLS),
        "available_date": [pd.Timestamp("2026-08-25")] * len(SYMBOLS),
        "industry_effective_date": [pd.Timestamp("2026-07-01")] * len(SYMBOLS),
        "industry": [f"I{i % 4}" for i in range(len(SYMBOLS))],
        "broad_sector": [f"S{i % 3}" for i in range(len(SYMBOLS))],
        "benchmark_weight": [1.0 / len(SYMBOLS)] * len(SYMBOLS),
    }
    base = np.arange(1, len(SYMBOLS) + 1, dtype=float) / len(SYMBOLS)
    for index, feature in enumerate(V10_FEATURES):
        values[feature] = base * (1.0 + index / 1000)
    return pd.DataFrame(values)[DAILY_FEATURE_COLUMNS]


def _market() -> pd.DataFrame:
    rows = []
    for index, symbol in enumerate(SYMBOLS):
        price = 10.0 + index / 10
        rows.append(
            {
                "date": TARGET,
                "symbol": symbol,
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price * 1.002,
                "volume": 1_000_000 + index,
                "amount": (1_000_000 + index) * price,
            }
        )
    return pd.DataFrame(rows)


def _settlement_market() -> tuple[pd.DataFrame, str]:
    sessions = load_verified_calendar(CALENDAR_PATH).sessions()
    later = sessions[sessions > pd.Timestamp(TARGET)]
    maturity = later[20]
    selected = sessions[(sessions >= pd.Timestamp(TARGET)) & (sessions <= maturity)]
    rows = []
    for date_index, date in enumerate(selected):
        for symbol_index, symbol in enumerate(SYMBOLS):
            price = 10 + date_index * 0.02 + symbol_index * 0.001
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": price,
                    "close": price * 1.001,
                    "volume": 1_000_000.0,
                }
            )
    return pd.DataFrame(rows), str(maturity.date())


def build(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    market = _market()
    daily = _daily_panel()
    historical = _historical_panel()
    settlement, maturity = _settlement_market()
    market.to_csv(output / "market.csv", index=False, lineterminator="\n")
    daily.to_parquet(output / "panel.parquet", index=False, compression="zstd")
    historical.to_parquet(output / "historical_panel.parquet", index=False, compression="zstd")
    (output / "historical_manifest.json").write_bytes(
        canonical_json_bytes({"source_hashes": {}, "rows": len(historical)})
    )
    settlement.to_csv(output / "settlement_market.csv", index=False, lineterminator="\n")
    (output / "corporate_actions.json").write_bytes(canonical_json_bytes({"events": []}))
    membership = pd.DataFrame(
        {
            "snapshot_date": ["2026-08-31"] * len(SYMBOLS),
            "index_code": ["000300"] * len(SYMBOLS),
            "symbol": SYMBOLS,
            "weight": [1.0 / len(SYMBOLS)] * len(SYMBOLS),
            "source": ["sandbox-replay-fixture"] * len(SYMBOLS),
        }
    )
    membership.to_csv(output / "membership.csv", index=False, lineterminator="\n")
    names = (
        "market.csv",
        "panel.parquet",
        "historical_panel.parquet",
        "historical_manifest.json",
        "settlement_market.csv",
        "corporate_actions.json",
        "membership.csv",
    )
    manifest = {
        "contract_version": "DAILY_PIT_SANDBOX_REPLAY_V1",
        "mode": "SANDBOX_REPLAY_ONLY",
        "target_date": TARGET,
        "source_date": TARGET,
        "effective_timestamp": f"{TARGET}T15:00:00+08:00",
        "as_of_timestamp": f"{TARGET}T19:00:00+08:00",
        "settlement_as_of_timestamp": f"{maturity}T19:00:00+08:00",
        "pit_eligibility": "ELIGIBLE_RECORDED_FIXTURE",
        "normalized_input_hash": sha256_bytes(
            market.sort_values(["date", "symbol"])
            .to_csv(index=False, lineterminator="\n", float_format="%.12g")
            .encode("utf-8")
        ),
        "files": {name: sha256_file(output / name) for name in names},
        "provider_requests": 0,
        "sandbox_only": True,
    }
    write_immutable_json(output / "replay_manifest.json", manifest)


def main() -> None:
    value = argparse.ArgumentParser()
    value.add_argument("output", type=Path)
    args = value.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
