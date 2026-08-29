import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research_v10.features import V10_FEATURES
from research_v25r1.config import V25R1Settings
from research_v25r1.model import TemporalModels, mature_window, score_temporal_delta
from research_v25r1.replay import run_replay, schedule_from_parent
from research_v22 import replay as parent_replay
from research_v20r2.ledger import PriceBook


class ConstantModel:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, frame):
        return self.values[:len(frame)]


class V25R1Tests(unittest.TestCase):
    def test_mature_window_excludes_unmatured_and_old_labels(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2016-01-01", "2017-01-01", "2019-12-01", "2019-12-20"]),
            "symbol": ["1", "2", "3", "4"], "eligible": [True] * 4,
            "target": [1.0] * 4,
            "end": pd.to_datetime(["2016-01-10", "2017-01-10", "2019-12-31", "2020-01-02"]),
        })
        result = mature_window(frame, 2020, 3, "target", "end")
        self.assertEqual(result.symbol.tolist(), ["2", "3"])

    def test_temporal_delta_changes_only_registered_lightgbm_share(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-02"] * 3), "symbol": ["1", "2", "3"],
                              "global_model_score": [0.1, 0.2, 0.3]})
        for feature in V10_FEATURES:
            frame[feature] = 0.0
        full = ConstantModel([1, 2, 3])
        short_a = ConstantModel([3, 2, 1])
        short_b = ConstantModel([3, 2, 1])
        models = TemporalModels({"5": [(8, full), (5, short_a), (3, short_b)],
                                 "20": [(8, full), (5, short_a), (3, short_b)]}, {})
        result = score_temporal_delta(frame, models, V25R1Settings())
        expected_horizon_delta = pd.Series([4 / 9, 0, -4 / 9], dtype=float)
        np.testing.assert_allclose(result.temporal_delta, 0.24 * expected_horizon_delta, atol=1e-12)
        np.testing.assert_allclose(result.temporal_ensemble_score, frame.global_model_score + result.temporal_delta)

    def test_evaluation_labels_are_not_model_inputs(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-02"] * 3), "symbol": ["1", "2", "3"],
                              "global_model_score": [0.1, 0.2, 0.3], "label_5": [99, -99, 0]})
        for feature in V10_FEATURES:
            frame[feature] = np.arange(3)
        models = TemporalModels({h: [(8, ConstantModel([1, 2, 3])), (5, ConstantModel([2, 3, 1])),
                                      (3, ConstantModel([3, 1, 2]))] for h in ("5", "20")}, {})
        one = score_temporal_delta(frame, models, V25R1Settings()).temporal_ensemble_score
        two = score_temporal_delta(frame.assign(label_5=[-1e9, 1e9, 7]), models, V25R1Settings()).temporal_ensemble_score
        np.testing.assert_allclose(one, two)

    def test_replay_restores_parent_globals(self):
        old_modes, old_columns = parent_replay.MODES, parent_replay.SCORE_COLUMNS
        with patch("research_v25r1.replay.parent.run_replay", side_effect=RuntimeError("stop")):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                run_replay(None, None, None, None, None)
        self.assertEqual(parent_replay.MODES, old_modes)
        self.assertEqual(parent_replay.SCORE_COLUMNS, old_columns)

    def test_schedule_uses_repaired_mode_column_access(self):
        dates = pd.bdate_range("2019-12-02", periods=110)
        panel = pd.DataFrame({"date": dates, "symbol": "600001", "open": 100.0,
                              "close": 100.0, "volume": 1.0})
        book = PriceBook(panel)
        rows = [{"date": signal, "entry_date": book.dates[i + 1], "end_date": book.dates[i + 21],
                 "mode": "v16_control"} for i, signal in enumerate(dates[:73])]
        self.assertEqual(len(schedule_from_parent(pd.DataFrame(rows), book)), 73)


if __name__ == "__main__":
    unittest.main()
