import unittest

import numpy as np
import pandas as pd

from stockpilot.model import FeatureRanker, LightGBMRanker, RidgeRanker, create_model


class ModelTests(unittest.TestCase):
    def test_ridge_learns_direction(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(300, 3))
        y = 0.5 * x[:, 0] - 0.2 * x[:, 1] + rng.normal(0, 0.02, 300)
        model = RidgeRanker(alpha=0.1).fit(x, y)
        pred = model.predict(x)
        self.assertGreater(np.corrcoef(pred, y)[0, 1], 0.95)
        self.assertGreater(model.coef_[1], 0)
        self.assertLess(model.coef_[2], 0)

    def test_feature_ranker_direction(self):
        frame = pd.DataFrame({"ret_5_rank": [0.1, 0.7, 0.4]})
        model = FeatureRanker("ret_5_rank", direction=-1).fit(frame, np.zeros(3))
        np.testing.assert_allclose(model.predict(frame), [-0.1, -0.7, -0.4])

    def test_model_factory_rejects_unknown_name(self):
        with self.assertRaisesRegex(ValueError, "未知模型"):
            create_model("oracle")

    def test_lightgbm_accepts_date_groups(self):
        rng = np.random.default_rng(7)
        x = pd.DataFrame(rng.normal(size=(1000, 3)), columns=["a", "b", "c"])
        y = pd.Series(x["a"] - 0.5 * x["b"])
        model = LightGBMRanker().fit(x, y, group_sizes=np.repeat(20, 50))
        self.assertGreater(np.corrcoef(model.predict(x), y)[0, 1], 0.5)


if __name__ == "__main__":
    unittest.main()
