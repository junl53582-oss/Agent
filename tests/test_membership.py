import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stockpilot.membership import (
    MEMBERSHIP_COLUMNS,
    attach_point_in_time_membership,
    export_qlib_intervals,
    load_membership_history,
    save_membership_history,
)


class MembershipTests(unittest.TestCase):
    def setUp(self):
        self.history = pd.DataFrame(
            [
                ["2024-01-01", "000300", "000001", 50.0, "test"],
                ["2024-01-01", "000300", "000002", 50.0, "test"],
                ["2024-07-01", "000300", "000002", 50.0, "test"],
                ["2024-07-01", "000300", "000003", 50.0, "test"],
            ],
            columns=MEMBERSHIP_COLUMNS,
        )
        self.history["snapshot_date"] = pd.to_datetime(self.history["snapshot_date"])

    def test_point_in_time_membership_does_not_backfill_future_snapshot(self):
        panel = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-12-29", "2024-03-01", "2024-08-01"] * 3),
                "symbol": ["000001"] * 3 + ["000002"] * 3 + ["000003"] * 3,
            }
        )
        result = attach_point_in_time_membership(panel, self.history)
        before = result[result["date"] < "2024-01-01"]
        self.assertFalse(before["membership_known"].any())
        march = result[result["date"] == "2024-03-01"].set_index("symbol")
        self.assertTrue(bool(march.loc["000001", "in_universe"]))
        self.assertFalse(bool(march.loc["000003", "in_universe"]))
        august = result[result["date"] == "2024-08-01"].set_index("symbol")
        self.assertFalse(bool(august.loc["000001", "in_universe"]))
        self.assertTrue(bool(august.loc["000003", "in_universe"]))
        self.assertEqual(int(august["membership_universe_size"].max()), 2)
        self.assertEqual(int(august["membership_available_size"].max()), 2)

    def test_roundtrip_and_qlib_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = save_membership_history(self.history, Path(tmp) / "history.csv")
            loaded = load_membership_history(csv_path)
            self.assertEqual(len(loaded), 4)
            qlib_path = export_qlib_intervals(
                loaded, Path(tmp) / "csi300.txt", end_date="2024-12-31"
            )
            text = qlib_path.read_text(encoding="utf-8")
            self.assertIn("SZ000001\t2024-01-01\t2024-06-30", text)
            self.assertIn("SZ000003\t2024-07-01\t2024-12-31", text)


if __name__ == "__main__":
    unittest.main()
