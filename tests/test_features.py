import unittest

from stockpilot.data import make_demo_panel
from stockpilot.features import FEATURE_COLUMNS, build_dataset


class FeatureTests(unittest.TestCase):
    def test_features_are_cross_sectional_and_labels_use_future_open(self):
        panel = make_demo_panel(symbols=8, periods=340)
        dataset = build_dataset(panel, horizon=5)
        eligible = dataset[dataset["eligible"]]
        self.assertFalse(eligible.empty)
        self.assertTrue(eligible[FEATURE_COLUMNS].notna().all().all())
        row = dataset.dropna(subset=["future_return"]).iloc[150]
        expected = row["exit_open"] / row["entry_open"] - 1
        self.assertAlmostEqual(row["future_return"], expected)
        self.assertGreater(row["label_end_date"], row["date"])
        neutral = dataset.dropna(subset=["neutral_label"])
        daily_means = neutral.groupby("date")["neutral_label"].mean().abs()
        self.assertLess(float(daily_means.max()), 1e-10)


if __name__ == "__main__":
    unittest.main()
