import unittest
from dataclasses import asdict

import numpy as np
import pandas as pd

from research_v28.config import V28Settings
from research_v29.config import V29Settings
from research_v29.model import sector_tail_labels
from research_v29.replay import MODES, SCORE_COLUMNS


class V29Tests(unittest.TestCase):
    def test_tail_labels_are_sector_and_date_local(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01"] * 20 + ["2020-01-02"] * 20),
            "broad_sector": (["technology"] * 10 + ["financial"] * 10) * 2,
            "target": list(range(10)) + list(range(100, 110)) + list(range(10, 0, -1)) + list(range(110, 100, -1)),
        })
        labels = sector_tail_labels(frame, "target", 0.8)
        for _, group in frame.assign(label=labels).groupby(["date", "broad_sector"]):
            self.assertEqual(int(group.label.sum()), 2)

    def test_missing_sector_fails_closed(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"] * 10),
                              "broad_sector": ["technology"] * 9 + [None], "target": np.arange(10)})
        with self.assertRaisesRegex(ValueError, "point-in-time sectors"):
            sector_tail_labels(frame, "target", 0.8)

    def test_only_artifact_dir_differs_from_v28_settings(self):
        parent, current = asdict(V28Settings()), asdict(V29Settings())
        parent.pop("artifact_dir")
        current.pop("artifact_dir")
        self.assertEqual(parent, current)

    def test_replay_changes_only_candidate_name_and_score_column(self):
        self.assertEqual(MODES[0], "v16_replay")
        self.assertEqual(SCORE_COLUMNS["v16_replay"], "v16_score")
        self.assertEqual(MODES[1], "v29_sector_tail")
        self.assertEqual(SCORE_COLUMNS[MODES[1]], "v29_score")


if __name__ == "__main__":
    unittest.main()
