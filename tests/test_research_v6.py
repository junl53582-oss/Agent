import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from research_v4.lock import verify_plan_lock
from research_v6.config import PLAN_LOCK_SHA256, V6Settings
from research_v6.model import _sector_quotas, select_sector_balanced


class ResearchV6Tests(unittest.TestCase):
    def test_plan_lock_rejects_changes(self):
        source = Path("artifacts/research_v6/plan.lock.json")
        verify_plan_lock(source, PLAN_LOCK_SHA256)
        with TemporaryDirectory() as directory:
            target = Path(directory) / "plan.lock.json"
            shutil.copyfile(source, target)
            target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_plan_lock(target, PLAN_LOCK_SHA256)

    def test_sector_quotas_follow_eligible_universe(self):
        current = pd.DataFrame(
            {"broad_sector": ["technology"] * 60 + ["finance_real_estate"] * 40}
        )
        quotas = _sector_quotas(current, 30)
        self.assertEqual(quotas, {"finance_real_estate": 12, "technology": 18})

    def test_selected_weights_preserve_sector_shares(self):
        current = pd.DataFrame(
            {
                "symbol": [f"{index:06d}" for index in range(100)],
                "broad_sector": ["technology"] * 60 + ["finance_real_estate"] * 40,
                "score": np.linspace(-0.5, 0.5, 100),
                "volatility_20": np.linspace(0.01, 0.04, 100),
            }
        )
        selected = select_sector_balanced(current, V6Settings(top_n=30, min_positions=20))
        self.assertEqual(len(selected), 30)
        weights = selected.groupby("broad_sector")["weight"].sum()
        self.assertAlmostEqual(weights["technology"], 0.60)
        self.assertAlmostEqual(weights["finance_real_estate"], 0.40)
        self.assertAlmostEqual(selected["weight"].sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
