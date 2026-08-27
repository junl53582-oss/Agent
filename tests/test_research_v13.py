import unittest

import numpy as np
import pandas as pd

from research_v10.features import V10_FEATURES
from research_v13.config import V13Settings
from research_v13.model import TwoStageModel, confidence_lower_bound, group_percentile
from research_v13.risk import DrawdownState


class ResearchV13Tests(unittest.TestCase):
    def test_group_percentile_is_sector_local(self):
        frame = pd.DataFrame({"date": ["2020-01-01"] * 20, "broad_sector": ["technology"] * 10 + ["consumer"] * 10, "v12_net_marginal_target": list(range(10)) + list(range(10))})
        percentile = group_percentile(frame)
        for indexes in frame.groupby("broad_sector").groups.values():
            self.assertEqual(int((percentile.loc[indexes] > 0.8).sum()), 2)

    def test_confidence_lower_bound_penalizes_uncertainty(self):
        stable = confidence_lower_bound([0.01] * 20, V13Settings().confidence_z)
        noisy = confidence_lower_bound([0.03, -0.01] * 10, V13Settings().confidence_z)
        self.assertGreater(stable, noisy)
        self.assertGreater(stable, 0)

    def test_drawdown_state_has_hysteresis(self):
        state = DrawdownState()
        def frame(drawdown, momentum):
            return pd.DataFrame({"v13_market_drawdown_120": [drawdown], "v12_market_momentum_60": [momentum]})
        self.assertEqual(state.update(frame(-0.07, -0.01))[0], 0.70)
        self.assertEqual(state.update(frame(-0.03, -0.01))[0], 0.70)
        self.assertEqual(state.update(frame(-0.03, 0.01))[0], 1.0)
        self.assertEqual(state.update(frame(-0.13, -0.01))[0], 0.40)
        self.assertEqual(state.update(frame(-0.07, 0.01))[0], 0.70)

    def test_drawdown_state_never_reads_future_return(self):
        state = DrawdownState()
        frame = pd.DataFrame({"v13_market_drawdown_120": [-0.07], "v12_market_momentum_60": [-0.01], "future_return_20": [-999]})
        first = state.update(frame)[0]
        frame["future_return_20"] = 999
        second = DrawdownState().update(frame)[0]
        self.assertEqual(first, second)

    def test_two_stage_model_smoke(self):
        rng = np.random.default_rng(44)
        rows = 1800
        frame = pd.DataFrame(rng.normal(size=(rows, len(V10_FEATURES))), columns=V10_FEATURES)
        frame["date"] = np.repeat(pd.date_range("2018-01-01", periods=30), 60)
        frame["broad_sector"] = np.tile(np.repeat(["technology", "consumer"], 30), 30)
        frame["symbol"] = [f"{i % 60:06d}" for i in range(rows)]
        frame["v12_net_marginal_target"] = rng.normal(0.002, 0.03, rows)
        model = TwoStageModel().fit(frame, V13Settings())
        probability, magnitude = model.predict_components(frame.iloc[:60])
        self.assertTrue(((probability >= 0) & (probability <= 1)).all())
        self.assertTrue(np.isfinite(magnitude).all())
        self.assertTrue(np.isfinite(model.predict(frame.iloc[:60])).all())


if __name__ == "__main__":
    unittest.main()

