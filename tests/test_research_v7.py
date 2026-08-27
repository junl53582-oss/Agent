import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from research_v4.lock import verify_plan_lock
from research_v5.features import MODEL_FEATURES
from research_v6.config import V6Settings
from research_v6.model import select_sector_balanced
from research_v7.config import PLAN_LOCK_SHA256, V7Settings
from research_v7.model import fit_multihorizon_models, score_multihorizon


class LinearPredictor:
    def __init__(self, multiplier: float):
        self.multiplier = multiplier

    def predict(self, features):
        return features["momentum"].to_numpy(dtype=float) * self.multiplier


class ResearchV7Tests(unittest.TestCase):
    def test_plan_lock_rejects_changes(self):
        source = Path("artifacts/research_v7/plan.lock.json")
        verify_plan_lock(source, PLAN_LOCK_SHA256)
        with TemporaryDirectory() as directory:
            target = Path(directory) / "plan.lock.json"
            shutil.copyfile(source, target)
            target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_plan_lock(target, PLAN_LOCK_SHA256)

    def test_multihorizon_score_reports_agreement_and_uncertainty(self):
        current = pd.DataFrame(
            {
                feature: np.linspace(-1, 1, 5) if feature == "momentum" else 0.0
                for feature in MODEL_FEATURES
            }
        )
        current["broad_sector"] = "technology"
        models = {
            5: type("Models", (), {"global_model": LinearPredictor(1), "experts": {}})(),
            20: type("Models", (), {"global_model": LinearPredictor(1), "experts": {}})(),
            60: type("Models", (), {"global_model": LinearPredictor(-1), "experts": {}})(),
        }
        scored = score_multihorizon(current, models, V7Settings(global_share=1, expert_share=0))
        self.assertTrue(scored["horizon_uncertainty"].gt(0).any())
        self.assertTrue(scored["horizon_agreement"].between(0, 1).all())
        self.assertAlmostEqual(scored.loc[4, "multihorizon_score"], 0.34)
        self.assertAlmostEqual(scored.loc[4, "horizon_agreement"], 2 / 3)

    def test_future_labels_do_not_change_fitted_models(self):
        rows = 90
        dates = pd.date_range("2023-09-01", periods=rows, freq="D")
        dataset = pd.DataFrame(
            {
                feature: np.linspace(-1, 1, rows) + index * 0.01
                for index, feature in enumerate(MODEL_FEATURES)
            }
        )
        dataset["date"] = dates
        dataset["symbol"] = [f"{index:06d}" for index in range(rows)]
        dataset["eligible"] = True
        dataset["broad_sector"] = "technology"
        for horizon in (5, 20, 60):
            dataset[f"label_{horizon}"] = np.linspace(-0.2, 0.2, rows)
            dataset[f"label_end_date_{horizon}"] = dates + pd.Timedelta(days=horizon)
        settings = V7Settings(test_years=(2024,))
        baseline = fit_multihorizon_models(dataset, 2024, settings)
        changed = dataset.copy()
        for horizon in settings.horizons:
            future = changed[f"label_end_date_{horizon}"] >= pd.Timestamp("2024-01-01")
            changed.loc[future, f"label_{horizon}"] = 9999
        refitted = fit_multihorizon_models(changed, 2024, settings)
        for horizon in settings.horizons:
            np.testing.assert_allclose(
                baseline[horizon].global_model.coef_, refitted[horizon].global_model.coef_
            )

    def test_holding_bonus_can_retain_borderline_position(self):
        current = pd.DataFrame(
            {
                "symbol": [f"{index:06d}" for index in range(31)],
                "broad_sector": "technology",
                "score": np.linspace(1.0, 0.0, 31),
                "volatility_20": 0.02,
            }
        )
        without_bonus = select_sector_balanced(current, V6Settings(top_n=30, min_positions=20))
        self.assertNotIn("000030", set(without_bonus["symbol"]))
        current.loc[current["symbol"] == "000030", "score"] += 0.05
        with_bonus = select_sector_balanced(current, V6Settings(top_n=30, min_positions=20))
        self.assertIn("000030", set(with_bonus["symbol"]))


if __name__ == "__main__":
    unittest.main()
