"""Lightweight all-year preflight; no model fitting or performance selection."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_v20.freeze import digest, write_new
from stockpilot.data import load_panel
from stockpilot.membership import load_membership_history
from .config import V20R2Settings
from .ledger import PriceBook, evaluation_schedule, snapshot_weights


MARKET_PATH = Path("data/market_history_v10_hfq.csv")
MEMBERSHIP_PATH = Path("data/universes/000300/history_v10.csv")


def load_book(settings=None):
    settings = settings or V20R2Settings()
    panel = load_panel(MARKET_PATH)
    events = json.loads(settings.action_path.read_text(encoding="utf-8"))["events"]
    book = PriceBook(panel, events)
    membership = load_membership_history(MEMBERSHIP_PATH)
    return panel, book, membership


def audit_inputs(panel, book, history, settings=None):
    settings = settings or V20R2Settings()
    ordered = panel.sort_values(["symbol", "date"])
    grouped = ordered.groupby("symbol")
    returns = grouped.open.shift(-21) / grouped.open.shift(-1) - 1
    missing = ordered.loc[~np.isfinite(returns), ["date", "symbol"]]
    failure_examples = []
    stale_at_decision = []
    schedule = evaluation_schedule(panel, settings)
    # Validate all snapshot members, not only rows still present in the market file.
    for date, start, end in schedule:
        base = snapshot_weights(history, date)
        for symbol in base:
            canonical = book.canonical({symbol: 1.0}, date)
            current = next(iter(canonical))
            for index in range(start - 1, end + 1):
                current = next(iter(book.canonical({symbol: 1.0}, book.dates[index])))
                book.mark(current, index)
            if not book.tradable[start - 1, book.columns[next(iter(canonical))]]:
                stale_at_decision.append({"date": str(date.date()), "symbol": symbol})
        for row in missing[missing.date.eq(date) & missing.symbol.isin(base)].itertuples():
            if row.symbol not in book.by_old:
                raise ValueError(f"unexplained missing forward label: {row.symbol} {date}")
            failure_examples.append({"date": str(date.date()), "symbol": row.symbol,
                                     "reason": "documented_merger_terminal_series"})
    return {"passed": True, "preflight_only": True, "performance_test": False,
            "market_rows": len(panel), "market_symbols": int(panel.symbol.nunique()),
            "evaluation_dates": len(schedule), "test_years": list(settings.test_years),
            "legacy_missing_forward_labels": failure_examples,
            "unquoted_snapshot_members": stale_at_decision,
            "actions": [{k: e[k] for k in ("old_symbol", "new_symbol", "ratio", "listing_date", "unit_ratio", "raw_old_close", "raw_new_open")} for e in book.events],
            "input_sha256": {str(p): digest(p) for p in (MARKET_PATH, MEMBERSHIP_PATH, settings.action_path)},
            "replacement_approved": False, "execution_authorized": False}


def run_audit():
    settings = V20R2Settings()
    panel, book, history = load_book(settings)
    report = audit_inputs(panel, book, history, settings)
    write_new(settings.artifact_dir / "data_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run_audit()
