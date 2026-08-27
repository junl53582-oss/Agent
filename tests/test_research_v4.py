import shutil
import unittest
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from research_v4.config import V4Settings
from research_v4.lock import verify_plan_lock
from research_v4.names import normalize_stock_names
from research_v4.predict import _write_immutable_snapshot, build_latest_snapshot
from research_v4.stability import FactorSpec, learn_factor_specs


class ResearchV4Tests(unittest.TestCase):
    @staticmethod
    def _stability_dataset() -> pd.DataFrame:
        rows = []
        rng = np.random.default_rng(7)
        for year in [2022, 2023, 2024]:
            for day in pd.bdate_range(f"{year}-01-03", periods=70):
                for symbol in range(30):
                    quality = symbol / 29 - 0.5
                    label = quality + rng.normal(0, 0.05)
                    rows.append(
                        {
                            "date": day,
                            "symbol": f"{symbol:06d}",
                            "eligible": True,
                            "label_5": label,
                            "label_end_date_5": day.to_pydatetime() + timedelta(days=7),
                            "quality": quality,
                            "growth": rng.normal(),
                            "low_volatility": rng.normal(),
                            "trend": rng.normal(),
                        }
                    )
        return pd.DataFrame(rows)

    def test_plan_lock_detects_modification(self):
        source = Path("artifacts/research_v4/plan.lock.json")
        verify_plan_lock(source)
        with TemporaryDirectory() as directory:
            target = Path(directory) / "plan.lock.json"
            shutil.copyfile(source, target)
            target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_plan_lock(target)

    def test_factor_selection_never_reads_test_year_labels(self):
        data = self._stability_dataset()
        settings = V4Settings(minimum_ic_days_per_year=60)
        original, _ = learn_factor_specs(data, 2024, settings)
        changed = data.copy()
        changed.loc[changed["date"].dt.year == 2024, "label_5"] *= -1000
        repeated, _ = learn_factor_specs(changed, 2024, settings)
        self.assertEqual([asdict(spec) for spec in original], [asdict(spec) for spec in repeated])
        quality = next(spec for spec in original if spec.factor == "quality")
        self.assertTrue(quality.selected)
        self.assertEqual(quality.direction, 1)

    def test_training_window_excludes_older_years(self):
        data = self._stability_dataset()
        settings = V4Settings(training_year_window=2, minimum_ic_days_per_year=60)
        specs, _ = learn_factor_specs(data, 2024, settings)
        quality = next(spec for spec in specs if spec.factor == "quality")
        self.assertEqual(quality.training_years, (2022, 2023))

    def test_latest_snapshot_ranks_full_cross_section_and_selects_top(self):
        data = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-08-24"] * 6),
                "symbol": [f"{index:06d}" for index in range(6)],
                "eligible": True,
                "close": np.arange(10, 16),
                "quality": np.arange(6, dtype=float),
                "growth": 0.0,
                "low_volatility": 0.0,
                "trend": 0.0,
                "volatility_20": 0.02,
                "board": "主板",
            }
        )
        specs = [
            FactorSpec("quality", True, 1, 1.0, 0.02, 1.0, (2023, 2024, 2025), 180)
        ]
        settings = V4Settings(top_n=3, min_positions=2, industry_cap=1.0)
        signals, predictions = build_latest_snapshot(data, specs, settings)
        self.assertEqual(len(predictions), 6)
        self.assertEqual(len(signals), 3)
        self.assertEqual(signals.iloc[0]["symbol"], "000005")
        self.assertAlmostEqual(signals["weight"].sum(), 1.0)
        self.assertFalse(bool(signals["execution_authorized"].any()))

    def test_live_snapshot_is_idempotent_but_cannot_be_overwritten(self):
        data = pd.DataFrame(
            {
                "symbol": ["000001"],
                "score": [0.1],
                "generated_at_utc": ["first"],
            }
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.csv"
            self.assertTrue(_write_immutable_snapshot(data, path))
            repeated = data.assign(generated_at_utc="second")
            self.assertFalse(_write_immutable_snapshot(repeated, path))
            changed = repeated.assign(score=0.2)
            with self.assertRaises(RuntimeError):
                _write_immutable_snapshot(changed, path)

    def test_stock_names_are_normalized_and_deduplicated(self):
        raw = pd.DataFrame(
            {"code": [1, "000001", "300059"], "name": ["旧名称", "平安银行", "东方财富"]}
        )
        result = normalize_stock_names(raw)
        self.assertEqual(result["symbol"].tolist(), ["000001", "300059"])
        self.assertEqual(result.iloc[0]["name"], "平安银行")


if __name__ == "__main__":
    unittest.main()
