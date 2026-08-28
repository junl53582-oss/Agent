import unittest

import numpy as np
import pandas as pd

from research_v18.config import V18Settings
from research_v18.text_model import EmbeddingTextModel


def _events(count=600):
    rows = []
    dates = pd.date_range("2017-01-03", periods=30, freq="14D")
    for index in range(count):
        positive = index % 2 == 0
        date = dates[index % len(dates)]
        signal = 0.03 if positive else -0.03
        rows.append({
            "symbol": f"{index % 60:06d}",
            "date": date,
            "document": "股份回购业绩预增" if positive else "股东减持风险提示",
            "event_count": 1,
            "eligible": True,
            "broad_sector": "technology" if index % 3 else "consumer",
            "benchmark_weight": 1 / 60,
            "event_target_1": signal,
            "event_target_5": signal * 1.5,
            "event_target_20": signal * 2,
            "event_label_end_1": date + pd.Timedelta(days=2),
            "event_label_end_5": date + pd.Timedelta(days=8),
            "event_label_end_20": date + pd.Timedelta(days=30),
        })
    return pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)


class ResearchV18Tests(unittest.TestCase):
    def test_settings_defaults(self):
        settings = V18Settings()
        self.assertEqual(settings.embedding_model, "BAAI/bge-small-zh-v1.5")
        self.assertEqual(settings.text_share, 0.35)
        self.assertEqual(settings.baseline_share, 0.65)

    def test_embedding_model_fit_and_score(self):
        events = _events()
        embeddings = np.random.default_rng(0).normal(size=(len(events), 16)).astype(np.float32)
        settings = V18Settings()
        model = EmbeddingTextModel.fit(events, embeddings, 2019, 2017, settings)
        current_date = events["date"].max()
        current = pd.DataFrame({
            "symbol": ["000000", "000001", "999999"],
            "date": [current_date] * 3,
            "broad_sector": ["consumer", "technology", "technology"],
            "eligible": [True] * 3,
        })
        scores = model.recent_scores(current, settings).set_index("symbol")
        self.assertTrue(np.isfinite(scores["text_score"]).all())
        self.assertEqual(scores.loc["999999", "text_score"], 0.0)
        self.assertEqual(scores.loc["999999", "text_events"], 0)
        self.assertEqual(model.event_years, [2017, 2018])

    def test_recent_scores_uses_current_eligible_universe(self):
        events = _events(4)
        embeddings = np.random.default_rng(1).normal(size=(len(events), 8)).astype(np.float32)
        settings = V18Settings()
        model = EmbeddingTextModel(events, embeddings, [], np.zeros(3), np.ones(3), 0, [], [])
        current = pd.DataFrame({
            "symbol": ["000000", "000001", "000002", "000003", "999999"],
            "date": [pd.Timestamp("2017-01-05")] * 5,
            "broad_sector": ["technology"] * 5, "eligible": [True, True, True, False, True],
        })
        scores = model.recent_scores(current, settings).set_index("symbol")
        self.assertAlmostEqual(scores.loc["000000", "text_score"], -0.25)
        self.assertAlmostEqual(scores.loc["000001", "text_score"], 0.25)


if __name__ == "__main__":
    unittest.main()
