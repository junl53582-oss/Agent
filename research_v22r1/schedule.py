import pandas as pd

from research_v20r2.ledger import PriceBook


def schedule_from_parent(parent: pd.DataFrame, book: PriceBook):
    """Exact V22 schedule logic with bracket access for the colliding mode column."""
    control = parent[parent["mode"].eq("v16_control")].copy()
    if control.date.duplicated().any() or len(control) != 73:
        raise ValueError("unexpected parent control schedule")
    rows = []
    for row in control.sort_values("date").itertuples():
        signal = pd.Timestamp(row.date)
        start, end = book.index(row.entry_date), book.index(row.end_date)
        if start != book.index(signal) + 1 or end - start != 20:
            raise ValueError("parent schedule is not the frozen common-calendar schedule")
        rows.append((signal, start, end))
    return rows
