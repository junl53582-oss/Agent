import unittest

import numpy as np
import pandas as pd

from research_v10.features import V10_FEATURES
from research_v26.config import V26Settings
from research_v26.model import DirectionalModels, direction_labels, score_directional_delta


class FixedModel:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, frame):
        return self.values[:len(frame)]


class V26Tests(unittest.TestCase):
    def test_direction_labels_are_strictly_positive(self):
        np.testing.assert_array_equal(direction_labels(pd.Series([-1.0, 0.0, 2.0])), [0, 0, 1])

    def test_direction_labels_require_two_classes_and_finite_values(self):
        with self.assertRaisesRegex(ValueError, "both classes"):
            direction_labels([1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            direction_labels([1.0, np.nan])

    def test_only_registered_lightgbm_contribution_changes(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-02"] * 3), "symbol": ["1", "2", "3"],
                              "global_model_score": [0.1, 0.2, 0.3]})
        for feature in V10_FEATURES:
            frame[feature] = 0.0
        models = DirectionalModels(
            regression={"5": FixedModel([1, 2, 3]), "20": FixedModel([1, 2, 3])},
            probability={"5": FixedModel([3, 2, 1]), "20": FixedModel([3, 2, 1])},
            training_rows={}, positive_rates={},
        )
        result = score_directional_delta(frame, models, V26Settings())
        expected = 0.24 * pd.Series([2 / 3, 0, -2 / 3], dtype=float)
        np.testing.assert_allclose(result.directional_delta, expected)
        np.testing.assert_allclose(result.directional_probability_score, frame.global_model_score + expected)

    def test_evaluation_targets_do_not_enter_score(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-02"] * 3), "symbol": ["1", "2", "3"],
                              "global_model_score": [0.1, 0.2, 0.3], "label_5": [9, -9, 0]})
        for feature in V10_FEATURES:
            frame[feature] = np.arange(3)
        models = DirectionalModels(
            regression={h: FixedModel([1, 2, 3]) for h in ("5", "20")},
            probability={h: FixedModel([2, 3, 1]) for h in ("5", "20")},
            training_rows={}, positive_rates={},
        )
        one = score_directional_delta(frame, models, V26Settings()).directional_probability_score
        two = score_directional_delta(frame.assign(label_5=[-1e9, 1e9, 7]), models, V26Settings()).directional_probability_score
        np.testing.assert_allclose(one, two)


if __name__ == "__main__":
    unittest.main()
