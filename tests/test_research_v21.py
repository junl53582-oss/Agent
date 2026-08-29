import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from research_v20r2.config import V20R2Settings
from research_v20r2.ledger import evaluation_schedule
from research_v21.diagnostics import COMPONENTS, GROUPS, TARGETS, check_legacy_reproduction, component_diagnostics, rank_ic, summary_table
from research_v21.runner import json_safe, run_diagnosis


def scored_frame(date="2020-01-02"):
    data = pd.DataFrame({"date": pd.Timestamp(date), "symbol": [f"60000{i}" for i in range(6)],
                         "eligible": True, "broad_sector": ["technology"] * 3 + ["financial"] * 3,
                         "recent_text_events": [0, 0, 0, 1, 1, 1], "benchmark_weight": 1 / 6,
                         "label_end_date_5": pd.Timestamp(date) + pd.Timedelta(days=7),
                         "label_end_date_20": pd.Timestamp(date) + pd.Timedelta(days=28)})
    for name in (*COMPONENTS, *TARGETS):
        data[name] = np.arange(6, dtype=float)
    return data


def reference_for(dates):
    return pd.DataFrame([dict(date=date, mode=mode, rank_ic_5=1., rank_ic_20=1., technology_rank_ic_5=1.)
                         for date in dates for mode in ("v16_control", "v20_adaptive", "v20_timing")])


class ComponentTests(unittest.TestCase):
    def test_no_evidence_is_missing_not_zero(self):
        self.assertEqual(rank_ic(pd.Series([1., 1., 1.]), pd.Series([1., 2., 3.])), (None, 3))
        self.assertEqual(rank_ic(pd.Series([1., 2., np.nan]), pd.Series([1., 2., 3.])), (None, 2))

    def test_target_specific_masks_keep_valid_20_day_rows(self):
        frame = scored_frame()
        frame.loc[0, "label_5"] = np.nan
        metrics = component_diagnostics(frame)
        eligible = metrics[metrics.group.eq("eligible") & metrics.component.eq("v16_score")].set_index("target")
        self.assertEqual(eligible.loc["label_5", "labelled_rows"], 5)
        self.assertEqual(eligible.loc["v10_target_20", "labelled_rows"], 6)
        reference = reference_for([frame.date.iloc[0]])
        reference["technology_rank_ic_5"] = np.nan  # only two technology pairs remain
        proof = check_legacy_reproduction(frame, reference)
        self.assertEqual(next(row for row in proof if row["metric"] == "rank_ic_20")["labelled_rows"], 5)

    def test_component_groups_and_positive_direction(self):
        frame = scored_frame()
        output = component_diagnostics(frame)
        self.assertEqual(len(output), len(COMPONENTS) * len(TARGETS) * len(GROUPS))
        self.assertTrue(output.rank_ic.eq(1).all())
        output = component_diagnostics(frame.assign(v16_score=-frame.v16_score))
        self.assertTrue(output[output.component.eq("v16_score")].rank_ic.eq(-1).all())

    def test_duplicate_or_multiple_dates_rejected(self):
        frame = scored_frame()
        with self.assertRaises(ValueError):
            component_diagnostics(pd.concat([frame, frame.iloc[:1]]))
        frame.loc[0, "date"] += pd.Timedelta(days=1)
        with self.assertRaises(ValueError):
            component_diagnostics(frame)

    def test_reproduction_fails_on_sign_flip(self):
        frame = scored_frame()
        reference = reference_for([frame.date.iloc[0]])
        self.assertEqual(len(check_legacy_reproduction(frame, reference)), 6)
        with self.assertRaisesRegex(ValueError, "reproduction mismatch"):
            check_legacy_reproduction(frame.assign(v16_score=-frame.v16_score), reference)

    def test_summary_is_equal_date_weight_and_missing_not_zero(self):
        metrics = component_diagnostics(scored_frame())
        other = component_diagnostics(scored_frame("2021-01-04"))
        other["rank_ic"] = -1.
        other["labelled_rows"] *= 10
        summary = summary_table(pd.concat([metrics, other], ignore_index=True))
        self.assertTrue(summary[summary.test_year.eq("all")].mean_rank_ic.eq(0).all())
        self.assertEqual(json_safe({"x": np.nan, "y": np.bool_(False)}), {"x": None, "y": False})

    def test_protocol_matches_code(self):
        root = Path(__file__).resolve().parents[1]
        protocol = json.loads((root / "artifacts/research_v21/protocol.json").read_text(encoding="utf-8"))
        for field, expected in (("components", COMPONENTS), ("targets", TARGETS), ("groups", GROUPS)):
            self.assertEqual(protocol["fixed_design"][field], list(expected))

    def test_real_diagnostic_loop_checkpoints_without_changing_predictors(self):
        dates = pd.bdate_range("2019-12-20", "2021-02-15")
        dataset = pd.concat([scored_frame(day) for day in dates], ignore_index=True)
        dataset["in_universe"] = True
        settings = V20R2Settings(test_years=(2020,))
        selected = [row[0] for row in evaluation_schedule(dataset, settings)]
        reference = reference_for(selected)
        model = SimpleNamespace(global_gate=False, technology_gate=False, validation_diagnostics={}, training_events=20,
                                raw_event_years=[2018, 2019], payoff_lower_bound=-1., incremental_lower_bound=-1.,
                                technology_lower_bound=-1., technology_incremental_lower_bound=-1.,
                                baseline_model=SimpleNamespace(predict_components=lambda current: (np.ones(len(current)) * .5, np.ones(len(current)) * .01)))
        def fitting(data, corpus, year, supplied_settings, cache):
            self.assertIs(supplied_settings, settings)
            cache[year] = (None, None, None)
            return model
        def scoring(current, supplied_model, v5, v4, supplied_settings):
            self.assertIs(supplied_model, model)
            return current
        market = pd.DataFrame({"market_momentum": 0.0}, index=dates)
        with tempfile.TemporaryDirectory() as directory, \
             patch("research_v21.runner.fit_v16_models", side_effect=fitting), \
             patch("research_v21.runner.score_v16", side_effect=scoring), \
             patch("research_v21.runner.historical_market_state", return_value=market):
            folder = Path(directory)
            metrics, proofs, files = run_diagnosis(dataset, None, reference, settings, lambda *a, **k: None, folder)
            self.assertEqual(len(proofs), len(selected))
            self.assertEqual(len(files), 2)
            self.assertEqual(metrics.date.nunique(), len(selected))
            saved = pd.read_csv(folder / "scores_2020.csv", dtype={"symbol": str})
            self.assertFalse(saved.global_validation_gate.any())
            self.assertFalse(saved.technology_validation_gate.any())
            np.testing.assert_allclose(saved.v16_score, saved.adaptive_score)
            before = (folder / "scores_2020.csv").read_bytes()
            with self.assertRaises(FileExistsError):
                run_diagnosis(dataset, None, reference, settings, lambda *a, **k: None, folder)
            self.assertEqual(before, (folder / "scores_2020.csv").read_bytes())

    def test_lock_is_exclusive_and_tampering_detected(self):
        from research_v21.freeze import freeze, verify
        from research_v20.freeze import digest
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory, patch("research_v21.freeze.verify_parent", return_value={"lock_sha256": "parent", "settings": {}}):
            try:
                os.chdir(directory)
                required = ["research_v21/runner.py", "tests/test_research_v21.py", "artifacts/research_v21/protocol.json",
                            "artifacts/research_v21/test_receipt.json", "artifacts/research_v20r2/runtime_status.json",
                            "artifacts/autopilot/v20r2_acceptance_20260829.json", "artifacts/research_v20r2/equity.csv"]
                for name in required:
                    path = Path(name)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({"passed": True}), encoding="utf-8")
                parent = Path("artifacts/research_v20r2")
                (parent / "report.json").write_text(json.dumps({"lock_sha256": "parent", "output_sha256": {"equity.csv": digest(parent / "equity.csv")}}), encoding="utf-8")
                self.assertTrue(freeze()["frozen_inputs_intact"])
                self.assertFalse(verify()["execution_authorized"])
                with self.assertRaises(RuntimeError):
                    freeze()
                Path("research_v21/runner.py").write_text("changed", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "frozen file changed"):
                    verify()
            finally:
                os.chdir(original)


if __name__ == "__main__":
    unittest.main()
