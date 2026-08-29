"""Valid fixtures supplement, but do not rewrite, the frozen V18 tests."""
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from research_v18.config import V18Settings
from research_v18.text_model import EmbeddingTextModel


class ArchiveContractTests(unittest.TestCase):
    def test_v18_eligible_universe_with_fitted_regressor_contract(self):
        events = pd.DataFrame({
            "symbol": ["000001", "000002", "000003", "000004"],
            "date": pd.to_datetime(["2017-01-04"] * 4),
            "event_count": [1] * 4,
        })
        embeddings = np.array([[-1.0], [1.0], [100.0], [1000.0]])
        regressors = [SimpleNamespace(predict=lambda block: block[:, 0]) for _ in range(3)]
        model = EmbeddingTextModel(events, embeddings, regressors, np.zeros(3), np.ones(3), 600, [2017], [2017])
        current = pd.DataFrame({
            "symbol": ["000001", "000002", "000003", "999999"],
            "date": pd.to_datetime(["2017-01-05"] * 4),
            "broad_sector": ["technology"] * 4, "eligible": [True, True, False, True],
        })
        scores = model.recent_scores(current, V18Settings()).set_index("symbol")
        self.assertAlmostEqual(scores.loc["000001", "text_score"], -0.25)
        self.assertAlmostEqual(scores.loc["000002", "text_score"], 0.25)
        self.assertEqual(scores.loc["000003", "text_score"], 0.0)
        self.assertEqual(scores.loc["999999", "text_events"], 0)


if __name__ == "__main__":
    unittest.main()
