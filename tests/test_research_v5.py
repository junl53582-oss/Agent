import shutil
import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from research_v4.lock import verify_plan_lock
from research_v5.config import PLAN_LOCK_SHA256, V5Settings
from research_v5.features import MODEL_FEATURES, broad_sector
from research_v5.models import fit_v5_models, mature_training, score_v5


class ResearchV5Tests(unittest.TestCase):
    @staticmethod
    def _dataset() -> pd.DataFrame:
        rng = np.random.default_rng(11)
        rows = []
        for year in [2022, 2023, 2024]:
            for date in pd.bdate_range(f"{year}-01-03", periods=30):
                for index in range(20):
                    sector = "technology" if index < 10 else "finance_real_estate"
                    features = rng.normal(size=len(MODEL_FEATURES))
                    label = 0.2 * features[0] + 0.1 * features[2] + rng.normal(0, 0.05)
                    row = {
                        "date": date,
                        "symbol": f"{index:06d}",
                        "eligible": True,
                        "label_5": label,
                        "label_end_date_5": date.to_pydatetime() + timedelta(days=7),
                        "broad_sector": sector,
                        "regime": "neutral",
                    }
                    row.update(dict(zip(MODEL_FEATURES, features, strict=True)))
                    rows.append(row)
        return pd.DataFrame(rows)

    def test_locked_plan_detects_changes(self):
        source = Path("artifacts/research_v5/plan.lock.json")
        verify_plan_lock(source, PLAN_LOCK_SHA256)
        with TemporaryDirectory() as directory:
            target = Path(directory) / "plan.lock.json"
            shutil.copyfile(source, target)
            target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_plan_lock(target, PLAN_LOCK_SHA256)

    def test_industries_route_to_distinct_experts(self):
        self.assertEqual(broad_sector("电子"), "technology")
        self.assertEqual(broad_sector("银行"), "finance_real_estate")
        self.assertEqual(broad_sector("医药生物"), "healthcare")
        self.assertEqual(broad_sector("建筑材料"), "cyclical_manufacturing")

    def test_training_never_reads_test_year_labels(self):
        data = self._dataset()
        settings = V5Settings(minimum_expert_rows=100, minimum_expert_dates=10)
        original = fit_v5_models(data, 2024, settings)
        changed = data.copy()
        changed.loc[changed["date"].dt.year == 2024, "label_5"] *= -1000
        repeated = fit_v5_models(changed, 2024, settings)
        np.testing.assert_allclose(original.global_model.coef_, repeated.global_model.coef_)
        for sector in original.experts:
            np.testing.assert_allclose(
                original.experts[sector].coef_, repeated.experts[sector].coef_
            )
        self.assertTrue((mature_training(data, 2024, settings)["date"].dt.year < 2024).all())

    def test_score_combines_fundamental_and_industry_models(self):
        data = self._dataset()
        settings = V5Settings(minimum_expert_rows=100, minimum_expert_dates=10)
        models = fit_v5_models(data, 2024, settings)
        current = data[data["date"] == data[data["date"].dt.year == 2024]["date"].min()].copy()
        scored = score_v5(current, models)
        self.assertTrue(
            {"fundamental", "behavior", "risk", "global_model", "industry_expert", "score"}
            .issubset(scored.columns)
        )
        self.assertGreater(scored["score"].std(), 0)


if __name__ == "__main__":
    unittest.main()
