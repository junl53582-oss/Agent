import unittest

import pandas as pd
import numpy as np

from research_v10.features import V10_FEATURES
from research_v14.config import V14Settings
from research_v14.features import ANNOUNCEMENT_FEATURES, attach_announcement_features
from research_v14.model import EventTwoStageModel, stable_announcement_features


class ResearchV14Tests(unittest.TestCase):
    def test_announcement_is_not_visible_on_publication_day(self):
        panel = pd.DataFrame({
            "symbol": ["000001"] * 4,
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]),
        })
        events = pd.DataFrame({
            "symbol": ["000001"],
            "announcement_date": pd.to_datetime(["2020-01-03"]),
            "title": ["关于股份回购的公告"],
            "announcement_id": ["a1"],
        })
        result = attach_announcement_features(panel, events)
        by_date = result.set_index("date")
        self.assertEqual(by_date.loc[pd.Timestamp("2020-01-03"), "announcement_count_5"], 0)
        self.assertEqual(by_date.loc[pd.Timestamp("2020-01-06"), "announcement_count_5"], 1)
        self.assertEqual(by_date.loc[pd.Timestamp("2020-01-06"), "announcement_positive_20"], 1)

    def test_negative_and_positive_titles_are_counted_separately(self):
        panel = pd.DataFrame({
            "symbol": ["000001"] * 3,
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
        })
        events = pd.DataFrame({
            "symbol": ["000001", "000001"],
            "announcement_date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
            "title": ["重大合同中标公告", "风险提示及诉讼公告"],
            "announcement_id": ["a1", "a2"],
        })
        result = attach_announcement_features(panel, events).set_index("date")
        row = result.loc[pd.Timestamp("2020-01-03")]
        self.assertEqual(row["announcement_positive_20"], 1)
        self.assertEqual(row["announcement_negative_20"], 1)
        self.assertEqual(row["announcement_net_sentiment_20"], 0)

    def test_compact_feature_frame_drops_intermediate_columns(self):
        panel = pd.DataFrame({
            "symbol": ["000001"] * 3,
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
        })
        events = pd.DataFrame({
            "symbol": ["000001"],
            "announcement_date": pd.to_datetime(["2020-01-02"]),
            "title": ["重大合同中标公告"],
            "announcement_id": ["a1"],
        })
        result = attach_announcement_features(panel, events, keep_intermediate=False)
        self.assertNotIn("announcement_count_20", result.columns)
        self.assertIn("announcement_count_20_rank", result.columns)
        self.assertEqual(str(result["announcement_count_20_rank"].dtype), "float32")

    def test_stability_selector_rejects_inconsistent_event_feature(self):
        rows = []
        for year in range(2015, 2020):
            for day in range(4):
                date = pd.Timestamp(year, 1, day + 1)
                for rank in range(30):
                    target = rank / 30
                    rows.append({
                        "date": date,
                        "v12_net_marginal_target": target,
                        "announcement_count_5_rank": target,
                        "announcement_count_20_rank": target if year % 2 else -target,
                    })
        frame = pd.DataFrame(rows)
        for feature in ANNOUNCEMENT_FEATURES:
            if feature not in frame:
                frame[feature] = 0.0
        selected, diagnostics = stable_announcement_features(frame, V14Settings())
        self.assertIn("announcement_count_5_rank", selected)
        self.assertNotIn("announcement_count_20_rank", selected)
        self.assertTrue(diagnostics["announcement_count_5_rank"]["selected"])

    def test_stability_selector_uses_all_early_event_years(self):
        rows = []
        for day in range(4):
            for rank in range(30):
                target = rank / 30
                rows.append({
                    "date": pd.Timestamp(2017, 1, day + 1),
                    "v12_net_marginal_target": target,
                    "announcement_count_5_rank": target,
                })
        frame = pd.DataFrame(rows)
        for feature in ANNOUNCEMENT_FEATURES:
            if feature not in frame:
                frame[feature] = 0.0
        selected, diagnostics = stable_announcement_features(frame, V14Settings())
        self.assertIn("announcement_count_5_rank", selected)
        self.assertEqual(diagnostics["announcement_count_5_rank"]["required_years"], 1)
        self.assertEqual(diagnostics["announcement_count_5_rank"]["event_years_available"], 1)

    def test_event_two_stage_model_smoke(self):
        rng = np.random.default_rng(46)
        features = [*V10_FEATURES, "announcement_count_20_rank"]
        rows = 1800
        frame = pd.DataFrame(rng.normal(size=(rows, len(features))), columns=features)
        frame["date"] = np.repeat(pd.date_range("2018-01-01", periods=30), 60)
        frame["broad_sector"] = np.tile(np.repeat(["technology", "consumer"], 30), 30)
        frame["symbol"] = [f"{index % 60:06d}" for index in range(rows)]
        frame["v12_net_marginal_target"] = rng.normal(0.002, 0.03, rows)
        model = EventTwoStageModel(features).fit(frame, V14Settings())
        probability, magnitude = model.predict_components(frame.iloc[:60])
        self.assertTrue(((probability >= 0) & (probability <= 1)).all())
        self.assertTrue(np.isfinite(magnitude).all())


if __name__ == "__main__":
    unittest.main()
