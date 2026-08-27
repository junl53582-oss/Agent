import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from research_v4.lock import verify_plan_lock
from research_v8.config import PLAN_LOCK_SHA256, V8Settings
from research_v8.features import ENHANCED_FEATURES, build_v8_dataset
from research_v8.model import V8Models, mature_training, score_v8


class ColumnPredictor:
    def __init__(self, column: str, direction: float = 1.0):
        self.column = column
        self.direction = direction

    def predict(self, features):
        return features[self.column].to_numpy(dtype=float) * self.direction


class ResearchV8Tests(unittest.TestCase):
    def test_plan_lock_rejects_changes(self):
        source = Path("artifacts/research_v8/plan.lock.json")
        verify_plan_lock(source, PLAN_LOCK_SHA256)
        with TemporaryDirectory() as directory:
            target = Path(directory) / "plan.lock.json"
            shutil.copyfile(source, target)
            target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_plan_lock(target, PLAN_LOCK_SHA256)

    def test_valuation_features_use_current_price_and_visible_fundamentals(self):
        rows = 5
        base = pd.DataFrame(
            {
                "date": pd.Timestamp("2025-01-02"),
                "symbol": [f"{index:06d}" for index in range(rows)],
                "close": [10.0] * rows,
                "book_value_per_share": [1, 2, 3, 4, 5],
                "earnings_per_share": [0.1, 0.2, 0.3, 0.4, 0.5],
                "industry": "电子",
                "ret_20": np.linspace(-0.1, 0.1, rows),
            }
        )
        for feature in ENHANCED_FEATURES:
            if feature not in base:
                base[feature] = 0.0
        with patch("research_v8.features.build_v5_dataset", return_value=base):
            result = build_v8_dataset(pd.DataFrame())
        self.assertAlmostEqual(result.loc[0, "book_to_price"], 0.1)
        self.assertAlmostEqual(result.loc[4, "book_to_price_rank"], 0.5)
        self.assertAlmostEqual(result.loc[4, "earnings_yield_rank"], 0.5)
        self.assertAlmostEqual(result.loc[4, "industry_momentum"], 0.5)

    def test_training_cutoff_uses_label_maturity_not_row_date_only(self):
        dataset = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-12-20", "2023-12-29", "2024-01-02"]),
                "label_end_date_5": pd.to_datetime(["2023-12-28", "2024-01-08", "2024-01-09"]),
                "label_5": [0.1, 999.0, 999.0],
                "eligible": True,
                "symbol": ["000001", "000002", "000003"],
            }
        )
        train = mature_training(dataset, 2024, V8Settings())
        self.assertEqual(list(train["symbol"]), ["000001"])

    def test_technology_rows_use_specialist_prediction(self):
        current = pd.DataFrame(
            {
                "symbol": ["000001", "000002", "000003", "000004"],
                "broad_sector": ["technology", "technology", "consumer", "consumer"],
                "regime": "neutral",
                "score": [0.0, 0.0, 0.0, 0.0],
            }
        )
        for feature in ENHANCED_FEATURES:
            current[feature] = np.linspace(-1, 1, len(current))
        models = V8Models(
            global_model=ColumnPredictor("book_to_price_rank", 1),
            technology_model=ColumnPredictor("book_to_price_rank", -1),
            technology_fallback=False,
            training_rows=10000,
            training_end=pd.Timestamp("2023-12-31"),
        )
        with patch("research_v8.model.score_v6", return_value=current.copy()):
            scored = score_v8(current, models, object(), [], V8Settings())
        tech = scored[scored["broad_sector"] == "technology"]
        self.assertGreater(tech.iloc[0]["technology_specialist"], tech.iloc[1]["technology_specialist"])
        self.assertNotEqual(tech.iloc[0]["enhanced_score"], tech.iloc[0]["enhanced_global"])


if __name__ == "__main__":
    unittest.main()
