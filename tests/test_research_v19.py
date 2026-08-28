import unittest

import pandas as pd

from research_v19.config import V19Settings
from research_v19.model import apply_v19_weights, regime_name, regime_weight


class ResearchV19Tests(unittest.TestCase):
    def test_regime_weight_thresholds(self):
        settings = V19Settings()
        self.assertAlmostEqual(regime_weight(0.05, settings), settings.weight_bull)
        self.assertAlmostEqual(regime_weight(0.10, settings), settings.weight_bull)
        self.assertAlmostEqual(regime_weight(0.0, settings), settings.weight_neutral)
        self.assertAlmostEqual(regime_weight(-0.05, settings), settings.weight_bear)
        self.assertAlmostEqual(regime_weight(0.02, settings), settings.weight_neutral)
        self.assertAlmostEqual(regime_weight(-0.02, settings), settings.weight_neutral)

    def test_regime_names(self):
        self.assertEqual(regime_name(0.05), "bull")
        self.assertEqual(regime_name(0.0), "neutral")
        self.assertEqual(regime_name(-0.05), "bear")

    def test_apply_v19_weights_blend(self):
        settings = V19Settings()
        frame = pd.DataFrame({
            "v13_comparable_score": [0.10, -0.20],
            "text_event_score": [0.30, -0.05],
        })
        bull = apply_v19_weights(frame, 0.05, settings)
        self.assertAlmostEqual(bull["v19_score"].iloc[0], 0.5 * 0.10 + 0.5 * 0.30)
        self.assertEqual(bull["market_regime"].iloc[0], "bull")

        bear = apply_v19_weights(frame, -0.05, settings)
        self.assertAlmostEqual(bear["v19_score"].iloc[0], 0.8 * 0.10 + 0.2 * 0.30)
        self.assertEqual(bear["market_regime"].iloc[0], "bear")

    def test_weights_sum_to_one(self):
        settings = V19Settings()
        for momentum in (0.05, 0.0, -0.05):
            weight = regime_weight(momentum, settings)
            self.assertAlmostEqual(weight + (1 - weight), 1.0)

    def test_settings_defaults(self):
        settings = V19Settings()
        self.assertAlmostEqual(settings.weight_bull, 0.50)
        self.assertAlmostEqual(settings.weight_neutral, 0.65)
        self.assertAlmostEqual(settings.weight_bear, 0.80)


if __name__ == "__main__":
    unittest.main()
