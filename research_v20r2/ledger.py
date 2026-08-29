"""Calendar-time NAV accounting, independent of frozen forward-return labels.

Prices are HFQ economic units, not exchange share counts. At a documented swap,
convert through contemporaneous raw/HFQ anchors before joining another security.
Stale marks are explicitly recorded, never treated as executable quotes.
"""
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from research_v20r2.config import V20R2Settings
from stockpilot.trading import price_limit_rate


class PriceBook:
    def __init__(self, panel, events=()):
        data = panel[["date", "symbol", "open", "close", "volume"]].copy()
        data["date"] = pd.to_datetime(data["date"])
        if data.duplicated(["date", "symbol"]).any():
            raise ValueError("duplicate price keys")
        if not np.isfinite(data[["open", "close"]]).all().all() or (data[["open", "close"]] <= 0).any().any():
            raise ValueError("invalid observed prices")
        self.dates = pd.DatetimeIndex(sorted(data.date.unique()))
        self.symbols = sorted(data.symbol.unique())
        self.columns = {symbol: i for i, symbol in enumerate(self.symbols)}
        self.open = data.pivot(index="date", columns="symbol", values="open").reindex(index=self.dates, columns=self.symbols)
        self.close = data.pivot(index="date", columns="symbol", values="close").reindex(index=self.dates, columns=self.symbols)
        volume = data.pivot(index="date", columns="symbol", values="volume").reindex(index=self.dates, columns=self.symbols)
        previous_close = self.close.ffill().shift(1)
        # Only last observed CLOSE may mark a missing opening quote.
        self.marks = self.open.fillna(previous_close).to_numpy()
        self.opens = self.open.to_numpy()
        self.previous_closes = previous_close.to_numpy()
        self.tradable = (self.open.notna() & volume.gt(0)).to_numpy()
        self.last_dates = data.groupby("symbol").date.max().to_dict()
        self.events = sorted([dict(e) for e in events], key=lambda e: e["listing_date"])
        self.by_old = {e["old_symbol"]: e for e in self.events}
        if len(self.by_old) != len(self.events):
            raise ValueError("duplicate conversion source")
        for e in self.events:
            if e["new_symbol"] in self.by_old or e["old_symbol"] == e["new_symbol"]:
                raise ValueError("chained/cyclic conversions are unsupported")
            if not (e["halt_announced"] <= e["last_trade"] < e["halt_start"] <= e["listing_announced"] < e["listing_date"]):
                raise ValueError("invalid action chronology")
            if not np.isfinite(e["ratio"]) or e["ratio"] <= 0:
                raise ValueError("invalid share swap ratio")
            old, new = e["old_symbol"], e["new_symbol"]
            if self.last_dates[old] != pd.Timestamp(e["last_trade"]):
                raise ValueError(f"action does not match terminal quote: {old}")
            raw_old = self.raw_quote(e["old_quote_file"], old, e["last_trade"], "close")
            raw_new = self.raw_quote(e["new_quote_file"], new, e["listing_date"], "open")
            hfq_old = float(self.close.loc[e["last_trade"], old])
            hfq_new = float(self.open.loc[e["listing_date"], new])
            if not np.isfinite([hfq_old, hfq_new]).all() or min(hfq_old, hfq_new) <= 0:
                raise ValueError("missing HFQ conversion anchors")
            e["unit_ratio"] = e["ratio"] * (hfq_old / raw_old) / (hfq_new / raw_new)
            e["raw_old_close"] = raw_old
            e["raw_new_open"] = raw_new

    @staticmethod
    def raw_quote(path, symbol, date, kind):
        with open(path, encoding="utf-8") as handle:
            rows = json.load(handle)["data"]["sh" + symbol]["day"]
        values = [float(row[1 if kind == "open" else 2]) for row in rows if row[0] == date]
        if len(values) != 1 or not np.isfinite(values[0]) or values[0] <= 0:
            raise ValueError(f"missing/duplicate raw anchor: {symbol} {date}")
        return values[0]

    def index(self, date):
        return int(self.dates.get_loc(pd.Timestamp(date)))

    def mark(self, symbol, index):
        if symbol not in self.columns:
            raise ValueError(f"no price history: {symbol}")
        date = self.dates[index]
        if date > self.last_dates[symbol]:
            e = self.by_old.get(symbol)
            if not e or not pd.Timestamp(e["halt_start"]) <= date < pd.Timestamp(e["listing_date"]):
                raise ValueError(f"unexplained terminal price gap: {symbol} {date.date()}")
        value = self.marks[index, self.columns[symbol]]
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"no observable valuation: {symbol} {date.date()}")
        return float(value)

    def can_trade(self, symbol, index, side):
        if symbol not in self.columns:
            return False
        col = self.columns[symbol]
        if not self.tradable[index, col]:
            return False
        previous = self.previous_closes[index, col]
        if not np.isfinite(previous) or previous <= 0:
            return False
        gap = self.opens[index, col] / previous - 1
        limit = price_limit_rate(symbol, date=self.dates[index]) * 0.995
        return bool(gap < limit if side == "buy" else gap > -limit)

    def canonical(self, weights, date):
        """Apply only already-listed actions to an observable target basket."""
        result = {}
        for symbol, weight in weights.items():
            e = self.by_old.get(symbol)
            if e and pd.Timestamp(e["listing_date"]) <= pd.Timestamp(date):
                symbol = e["new_symbol"]
            result[symbol] = result.get(symbol, 0.0) + weight
        return result


@dataclass
class Ledger:
    book: PriceBook
    settings: V20R2Settings = field(default_factory=V20R2Settings)
    charge_costs: bool = True
    cash: float = 1.0
    units: dict = field(default_factory=dict)
    action_log: list = field(default_factory=list)
    applied: set = field(default_factory=set)
    stale_observations: int = 0

    def settle(self, index):
        date = self.book.dates[index]
        for e in self.book.events:
            old = e["old_symbol"]
            if old in self.applied or pd.Timestamp(e["listing_date"]) > date:
                continue
            if date != pd.Timestamp(e["listing_date"]) and self.units.get(old, 0) > 0:
                raise ValueError("missed action settlement date")
            quantity = self.units.pop(old, 0.0)
            if quantity:
                new = e["new_symbol"]
                self.units[new] = self.units.get(new, 0.0) + quantity * e["unit_ratio"]
                self.action_log.append({"date": str(date.date()), "old_symbol": old, "new_symbol": new,
                                        "old_units": quantity, "new_units": quantity * e["unit_ratio"],
                                        "raw_share_ratio": e["ratio"], "fees": 0.0})
            self.applied.add(old)

    def nav(self, index):
        return self.cash + sum(q * self.book.mark(s, index) for s, q in self.units.items())

    def rebalance(self, weights, index):
        weights = self.book.canonical(weights, self.book.dates[index])
        if any(not np.isfinite(w) or w < 0 for w in weights.values()) or sum(weights.values()) > 1 + 1e-8:
            raise ValueError("invalid long-only target weights")
        self.settle(index)
        before = self.nav(index)
        buy_rate = self.settings.fee_rate + self.settings.slippage if self.charge_costs else 0.0
        sell_rate = buy_rate + self.settings.stamp_duty if self.charge_costs else 0.0
        buys = sells = fees = 0.0
        blocked = []
        # A missing/suspended sell cannot release cash or erase ownership.
        for symbol in sorted(list(self.units)):
            price = self.book.mark(symbol, index)
            excess = max(0.0, self.units[symbol] * price - weights.get(symbol, 0.0) * before)
            if excess <= 1e-12:
                continue
            if not self.book.can_trade(symbol, index, "sell"):
                blocked.append({"symbol": symbol, "side": "sell", "value": excess})
                continue
            self.units[symbol] -= excess / price
            self.cash += excess * (1 - sell_rate)
            sells += excess
            fees += excess * sell_rate
            if self.units[symbol] < 1e-12:
                del self.units[symbol]
        orders = {}
        for symbol, weight in sorted(weights.items()):
            # No observed quote means no purchase, not a synthetic tradable bar.
            if not self.book.can_trade(symbol, index, "buy"):
                blocked.append({"symbol": symbol, "side": "buy", "target_weight": weight})
                continue
            price = self.book.mark(symbol, index)
            value = max(0.0, weight * before - self.units.get(symbol, 0.0) * price)
            if value > 1e-12:
                orders[symbol] = value
        total = sum(orders.values())
        scale = min(1.0, max(0.0, self.cash) / (total * (1 + buy_rate))) if total else 0.0
        for symbol, value in orders.items():
            value *= scale
            self.units[symbol] = self.units.get(symbol, 0.0) + value / self.book.mark(symbol, index)
            self.cash -= value * (1 + buy_rate)
            buys += value
            fees += value * buy_rate
        if self.cash < -1e-10:
            raise AssertionError("negative cash after trading")
        self.cash = max(0.0, self.cash)
        if not np.isclose(self.nav(index), before - fees, atol=1e-10, rtol=1e-10):
            raise AssertionError("NAV/cost reconciliation failed")
        return {"nav_before": before, "buy_turnover": buys / before, "sell_turnover": sells / before,
                "transaction_cost": fees / before, "blocked_orders": len(blocked), "blocked": blocked}

    def advance(self, start, end):
        values = []
        for index in range(start + 1, end + 1):
            self.settle(index)
            self.stale_observations += sum(not self.book.tradable[index, self.book.columns[s]] for s in self.units)
            values.append((self.book.dates[index], self.nav(index)))
        return values


def snapshot_weights(history, date):
    available = history[pd.to_datetime(history["snapshot_date"]) <= pd.Timestamp(date)]
    if available.empty:
        raise ValueError(f"no PIT membership snapshot: {date}")
    snapshot = available["snapshot_date"].max()
    current = available[available["snapshot_date"].eq(snapshot)]
    if current.symbol.duplicated().any():
        raise ValueError("duplicate PIT membership")
    weights = pd.to_numeric(current.set_index("symbol")["weight"], errors="raise")
    if not np.isfinite(weights).all() or weights.le(0).any():
        raise ValueError("invalid PIT weights")
    return (weights / weights.sum()).to_dict()


def evaluation_schedule(panel, settings):
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel.date).unique()))
    test = dates[dates.year.isin(settings.test_years)][::settings.rebalance_every]
    rows = []
    for date in test:
        i = dates.get_loc(date)
        end = i + settings.rebalance_every + 1
        if end >= len(dates):
            raise ValueError(f"immature final evaluation horizon: {date}")
        rows.append((date, i + 1, end))
    return rows
