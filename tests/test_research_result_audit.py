import unittest

import numpy as np
import pandas as pd

from research_result_audit import MODES, check_frames, gate_screen


def fixture():
    equity, holdings, daily = [], [], []
    for mode in MODES:
        for date, entry, end, inside in (("2020-01-02", "2020-01-03", "2020-01-07", "2020-01-06"),
                                         ("2020-01-06", "2020-01-07", "2020-01-09", "2020-01-08")):
            equity.append(dict(mode=mode, date=date, entry_date=entry, end_date=end, test_year=2020,
                               nav=1.0, period_return=0.0, benchmark_return=0.0, excess_period_return=0.0,
                               buy_turnover=0.0, sell_turnover=0.0, transaction_cost=0.0, cash_weight=0.0,
                               rank_ic_5=-0.02, rank_ic_20=-0.01, technology_rank_ic_5=-0.03,
                               in_market=True, blocked_orders=0, stale_position_observations=0))
            holdings.append(dict(mode=mode, date=date, symbol="600001", units=1.0, weight=1.0, target_weight=1.0))
            daily.extend([dict(mode=mode, date=entry, point="after_rebalance", nav=1.0),
                          dict(mode=mode, date=inside, point="before_rebalance", nav=1.0),
                          dict(mode=mode, date=end, point="before_rebalance", nav=1.0)])
    frames = [pd.DataFrame(rows) for rows in (equity, holdings, daily)]
    for frame in frames:
        for name in ("date", "entry_date", "end_date"):
            if name in frame:
                frame[name] = pd.to_datetime(frame[name])
    settings = dict(test_years=[2020], rebalance_every=2, fee_rate=.0003, slippage=.0005, stamp_duty=.0005)
    return (*frames, settings, 2)


class ResultAuditTests(unittest.TestCase):
    def test_recompute_and_unverified_evidence(self):
        result = check_frames(*fixture())
        self.assertEqual(result[MODES[0]]["metrics"]["periods"], 2)
        gates = result[MODES[0]]["gate_screen"]
        self.assertEqual(gates["technology_ic_nonnegative"]["status"], "fail")
        for name in ("rank_ic_at_least_v6", "turnover_and_cost_at_most_v8", "candidate_future_shadow_126_days"):
            self.assertEqual(gates[name]["status"], "unverified")

    def test_missing_ic_is_not_pass(self):
        result = check_frames(*fixture())[MODES[0]]["metrics"]
        result["technology_rank_ic_5"] = None
        self.assertEqual(gate_screen(result)["technology_ic_nonnegative"]["status"], "unverified")

    def test_tampered_outputs_rejected(self):
        cases = [(0, "nav", 1.1), (0, "transaction_cost", .01), (0, "cash_weight", -1),
                 (0, "benchmark_return", .2), (0, "period_return", np.nan),
                 (1, "weight", 2), (1, "units", -1), (2, "nav", .5)]
        for index, column, value in cases:
            with self.subTest(column=column, index=index):
                values = fixture()
                values[index].loc[0, column] = value
                with self.assertRaises(ValueError):
                    check_frames(*values)

    def test_missing_or_duplicate_dates_rejected(self):
        for index in (0, 2):
            for duplicate in (True, False):
                values = list(fixture())
                frame = values[index]
                values[index] = pd.concat([frame, frame.iloc[:1]], ignore_index=True) if duplicate else frame.iloc[1:]
                with self.subTest(index=index, duplicate=duplicate), self.assertRaises(ValueError):
                    check_frames(*values)

    def test_entry_must_be_after_signal(self):
        values = fixture()
        values[0].loc[0, "entry_date"] = values[0].loc[0, "date"]
        with self.assertRaisesRegex(ValueError, "chronology"):
            check_frames(*values)


if __name__ == "__main__":
    unittest.main()
