import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from research_v20.config import V20Settings
from research_v20r2.backtest import run_backtest
from research_v20r2.config import V20R2Settings
from research_v20r2.ledger import Ledger, PriceBook, evaluation_schedule, snapshot_weights


def simple_panel(start="2019-11-01", end="2020-03-31"):
    dates = pd.bdate_range(start, end)
    return pd.DataFrame([{"date": date, "symbol": symbol, "open": 100.0, "close": 100.0, "volume": 1.0,
                          "in_universe": True, "benchmark_weight": 0.5, "eligible": True,
                          "future_return_20": 0.02, "label_5": 0.01, "v10_target_20": 0.01,
                          "broad_sector": "technology"}
                         for date in dates for symbol in ("600001", "600002")])


class CalendarLedgerTests(unittest.TestCase):
    def test_model_parameters_unchanged(self):
        parent, revised = asdict(V20Settings()), asdict(V20R2Settings())
        parent.pop("artifact_dir")
        revised.pop("artifact_dir")
        revised.pop("action_path")
        self.assertEqual(parent, revised)

    def test_calendar_end_does_not_skip_suspended_days(self):
        panel = simple_panel(end="2021-03-31")
        date = pd.Timestamp("2020-01-02")
        panel = panel[~(panel.symbol.eq("600001") & panel.date.between("2020-01-06", "2020-01-15"))]
        book = PriceBook(panel)
        schedule = evaluation_schedule(panel, V20R2Settings(test_years=(2020,)))
        _, start, end = next(r for r in schedule if r[0] == pd.Timestamp("2020-01-01"))
        self.assertEqual(end - start, 20)
        self.assertEqual(book.dates[start], date)

    def test_initial_purchase_is_self_financing_with_costs(self):
        book = PriceBook(simple_panel())
        ledger = Ledger(book)
        i = book.index("2020-01-02")
        trade = ledger.rebalance({"600001": 1.0}, i)
        self.assertGreaterEqual(ledger.cash, 0)
        self.assertAlmostEqual(ledger.nav(i) + trade["transaction_cost"], 1.0)
        self.assertAlmostEqual(ledger.units["600001"] * 100, 1 / 1.0008)

    def test_cash_exit_and_reentry_both_charge_actual_cost(self):
        book = PriceBook(simple_panel())
        ledger = Ledger(book)
        i = book.index("2020-01-02")
        ledger.rebalance({"600001": 1.0}, i)
        exit_trade = ledger.rebalance({}, i + 1)
        self.assertGreater(exit_trade["transaction_cost"], 0)
        self.assertFalse(ledger.units)
        reentry = ledger.rebalance({"600001": 1.0}, i + 2)
        self.assertGreater(reentry["buy_turnover"], 0)
        self.assertGreater(reentry["transaction_cost"], 0)

    def test_missing_bar_sale_keeps_asset_and_never_finances_other_buy(self):
        data = simple_panel()
        data = data[~(data.symbol.eq("600001") & data.date.eq("2020-01-03"))]
        book = PriceBook(data)
        ledger = Ledger(book, charge_costs=False)
        ledger.rebalance({"600001": 1.0}, book.index("2020-01-02"))
        result = ledger.rebalance({"600002": 1.0}, book.index("2020-01-03"))
        self.assertAlmostEqual(ledger.nav(book.index("2020-01-03")), 1.0)
        self.assertIn("600001", ledger.units)
        self.assertAlmostEqual(ledger.units.get("600002", 0.0), 0.0)
        self.assertEqual(result["sell_turnover"], 0.0)
        self.assertEqual(result["transaction_cost"], 0.0)

    def test_limit_down_exit_stays_held(self):
        data = simple_panel()
        data.loc[data.symbol.eq("600001") & data.date.eq("2020-01-03"), ["open", "close"]] = 90.0
        book = PriceBook(data)
        ledger = Ledger(book, charge_costs=False)
        ledger.rebalance({"600001": 1.0}, book.index("2020-01-02"))
        trade = ledger.rebalance({}, book.index("2020-01-03"))
        self.assertIn("600001", ledger.units)
        self.assertEqual(trade["sell_turnover"], 0)
        self.assertAlmostEqual(ledger.nav(book.index("2020-01-03")), 0.9)

    def test_limit_up_cannot_buy(self):
        data = simple_panel()
        data.loc[data.symbol.eq("600001") & data.date.eq("2020-01-03"), ["open", "close"]] = 110.0
        book = PriceBook(data)
        ledger = Ledger(book)
        ledger.rebalance({"600001": 1.0}, book.index("2020-01-03"))
        self.assertFalse(ledger.units)
        self.assertEqual(ledger.cash, 1.0)

    def test_unexplained_terminal_gap_fails(self):
        data = simple_panel()
        data = data[~(data.symbol.eq("600001") & data.date.ge("2020-01-03"))]
        book = PriceBook(data)
        with self.assertRaisesRegex(ValueError, "unexplained terminal"):
            book.mark("600001", book.index("2020-01-03"))

    def test_snapshot_includes_missing_bar_member_without_renormalizing_it_away(self):
        history = pd.DataFrame({"snapshot_date": pd.to_datetime(["2019-12-31"] * 2),
                                "symbol": ["600001", "600002"], "weight": [60, 40]})
        self.assertEqual(snapshot_weights(history, "2020-01-02"), {"600001": 0.6, "600002": 0.4})
        with self.assertRaisesRegex(ValueError, "no PIT"):
            snapshot_weights(history, "2019-12-30")

    def test_no_quote_mark_uses_previous_close_not_previous_open(self):
        data = simple_panel()
        data.loc[data.symbol.eq("600001") & data.date.eq("2020-01-02"), "close"] = 105.0
        data = data[~(data.symbol.eq("600001") & data.date.eq("2020-01-03"))]
        book = PriceBook(data)
        self.assertEqual(book.mark("600001", book.index("2020-01-03")), 105.0)
        self.assertFalse(book.can_trade("600001", book.index("2020-01-03"), "buy"))

    def test_target_sum_cannot_create_leverage(self):
        book = PriceBook(simple_panel())
        with self.assertRaisesRegex(ValueError, "invalid long-only"):
            Ledger(book).rebalance({"600001": 1, "600002": 1}, book.index("2020-01-02"))

    def test_future_quote_edit_does_not_change_earlier_trade(self):
        data = simple_panel()
        changed = data.copy()
        changed.loc[changed.date.gt("2020-01-02"), ["open", "close"]] *= 5
        positions = []
        for panel in (data, changed):
            book = PriceBook(panel)
            ledger = Ledger(book)
            ledger.rebalance({"600001": 0.7, "600002": 0.3}, book.index("2020-01-02"))
            positions.append((dict(ledger.units), ledger.cash))
        self.assertEqual(positions[0], positions[1])

    def test_weight_drift_changes_realized_sell_turnover(self):
        data = simple_panel()
        data.loc[data.symbol.eq("600001") & data.date.ge("2020-01-03"), ["open", "close"]] = 105.0
        book = PriceBook(data)
        ledger = Ledger(book, charge_costs=False)
        ledger.rebalance({"600001": 0.5, "600002": 0.5}, book.index("2020-01-02"))
        result = ledger.rebalance({"600001": 0.5, "600002": 0.5}, book.index("2020-01-03"))
        self.assertGreater(result["sell_turnover"], 0)
        self.assertGreater(result["buy_turnover"], 0)
        self.assertAlmostEqual(ledger.nav(book.index("2020-01-03")), 1.025)

    def test_immature_calendar_horizon_fails(self):
        with self.assertRaisesRegex(ValueError, "immature"):
            evaluation_schedule(simple_panel(), V20R2Settings(test_years=(2020,)))


class CorporateActionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        dates = pd.bdate_range("2025-01-01", "2025-01-10")
        self.data = pd.DataFrame([{"date": d, "symbol": s, "open": price, "close": price, "volume": 1.0}
                                  for d in dates for s, price in (("600001", 100.0), ("600002", 400.0))
                                  if s != "600001" or d <= pd.Timestamp("2025-01-02")])
        old, new = self.folder / "old.json", self.folder / "new.json"
        old.write_text(json.dumps({"data": {"sh600001": {"day": [["2025-01-02", "10", "10"]]}}}))
        new.write_text(json.dumps({"data": {"sh600002": {"day": [["2025-01-08", "20", "20"]]}}}))
        self.event = {"old_symbol": "600001", "new_symbol": "600002", "ratio": 0.5,
                      "halt_announced": "2025-01-02", "last_trade": "2025-01-02", "halt_start": "2025-01-03",
                      "listing_announced": "2025-01-07", "listing_date": "2025-01-08",
                      "old_quote_file": str(old), "new_quote_file": str(new)}

    def test_swap_converts_price_units_not_just_symbols(self):
        book = PriceBook(self.data, [self.event])
        self.assertEqual(book.events[0]["unit_ratio"], 0.25)
        ledger = Ledger(book, charge_costs=False)
        ledger.rebalance({"600001": 1.0}, book.index("2025-01-02"))
        ledger.advance(book.index("2025-01-02"), book.index("2025-01-08"))
        self.assertNotIn("600001", ledger.units)
        self.assertAlmostEqual(ledger.units["600002"], 0.0025)
        self.assertAlmostEqual(ledger.nav(book.index("2025-01-08")), 1.0)
        self.assertEqual(len(ledger.action_log), 1)
        ledger.settle(book.index("2025-01-08"))
        self.assertEqual(len(ledger.action_log), 1)
        self.assertEqual(ledger.action_log[0]["fees"], 0)

    def test_future_swap_does_not_change_prelisting_mark_or_ownership(self):
        for ratio in (0.5, 8.0):
            book = PriceBook(self.data, [{**self.event, "ratio": ratio}])
            ledger = Ledger(book, charge_costs=False)
            ledger.rebalance({"600001": 1.0}, book.index("2025-01-02"))
            ledger.advance(book.index("2025-01-02"), book.index("2025-01-07"))
            self.assertAlmostEqual(ledger.nav(book.index("2025-01-07")), 1.0)
            self.assertIn("600001", ledger.units)
            self.assertFalse(book.can_trade("600001", book.index("2025-01-07"), "sell"))
            self.assertEqual(book.canonical({"600001": 1.0}, "2025-01-07"), {"600001": 1.0})

    def test_swap_adds_to_existing_successor_position(self):
        book = PriceBook(self.data, [self.event])
        ledger = Ledger(book, charge_costs=False, cash=0.0, units={"600001": 0.005, "600002": 0.00125})
        ledger.advance(book.index("2025-01-02"), book.index("2025-01-08"))
        self.assertAlmostEqual(ledger.units["600002"], 0.0025)
        self.assertAlmostEqual(ledger.nav(book.index("2025-01-08")), 1.0)

    def test_announcement_after_listing_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "chronology"):
            PriceBook(self.data, [{**self.event, "listing_announced": "2025-01-09"}])

    def test_missing_raw_anchor_fails_closed(self):
        (self.folder / "new.json").write_text(json.dumps({"data": {"sh600002": {"day": []}}}))
        with self.assertRaisesRegex(ValueError, "raw anchor"):
            PriceBook(self.data, [self.event])


class BacktestIntegrationTests(unittest.TestCase):
    def test_checkpoints_preserve_partial_results_and_refuse_overwrite(self):
        from research_v20r2.runner import checkpoint
        with tempfile.TemporaryDirectory() as folder:
            frame = pd.DataFrame({"date": ["2020-01-02"], "value": [1.0]})
            with patch("research_v20r2.runner.DIRECTORY", Path(folder)):
                checkpoint(2020, frame, frame, frame)
                marker = json.loads((Path(folder) / "checkpoints/through_2020.json").read_text())
                self.assertTrue(marker["partial_result_only"])
                self.assertEqual(len(marker["output_sha256"]), 3)
                with self.assertRaises(FileExistsError):
                    checkpoint(2020, frame, frame, frame)

    def test_real_ledger_backtest_ignores_forward_return_for_execution(self):
        data = simple_panel(end="2021-03-31")
        settings = V20R2Settings(test_years=(2020,))
        history = pd.DataFrame({"snapshot_date": pd.to_datetime(["2019-12-31"] * 2),
                                "symbol": ["600001", "600002"], "weight": [0.5, 0.5]})

        def fit(dataset, corpus, year, config, cache):
            cache[year] = (None, None, None)
            return None

        def score(frame, *args):
            return frame.assign(v16_score=0.1, v13_comparable_score=0.1, text_event_score=0.1)

        with patch("research_v20r2.backtest.fit_v16_models", side_effect=fit), patch("research_v20r2.backtest.score_v16", side_effect=score), patch("research_v20r2.backtest.optimize_v16", return_value=({"600001": 0.5, "600002": 0.5}, set(), {})):
            first = run_backtest(data, None, PriceBook(data), history, settings)
            changed = data.assign(future_return_20=np.nan)
            second = run_backtest(changed, None, PriceBook(changed), history, settings)
        pd.testing.assert_frame_equal(first[0], second[0])
        pd.testing.assert_frame_equal(first[1], second[1])
        self.assertTrue(np.isfinite(first[0].period_return).all())
        self.assertTrue(first[0].market_data_end.le(first[0].date).all())
        self.assertTrue(first[0].entry_date.gt(first[0].date).all())
        self.assertTrue(first[0].cash_weight.ge(0).all())
        self.assertEqual(len(first[0]), len(evaluation_schedule(data, settings)) * 3)


if __name__ == "__main__":
    unittest.main()
