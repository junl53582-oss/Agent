import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from research_v15.config import V15Settings
from research_v15.features import build_event_documents, raw_event_years
from research_v15.model import _sector_rank, _validation_predictions, baseline_training_view, validation_view, raw_year_gate
from research_v15.text_model import EventTextCorpus, MultiHorizonTextModel, event_training_masks
from research_v15.quality import event_quality_report, audit_event_data
from research_v15.freeze import freeze_research
from research_v15.validation import run_research_v15
from research_v15.backtest import max_drawdown
from research_v15.features import _official_sector_return


class ResearchV15Tests(unittest.TestCase):
    def test_raw_event_years_ignore_rank_like_ghost_values(self):
        events = pd.DataFrame({
            "date": pd.to_datetime(["2016-01-04", "2017-01-04", "2018-01-04"]),
            "event_count": [0, 2, 1],
            "announcement_rank_like_value": [0.001, 0.2, 0.3],
        })
        self.assertEqual(raw_event_years(events), [2017, 2018])

    def test_constant_sector_rank_is_exactly_zero(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-02"] * 6),
            "broad_sector": ["technology"] * 3 + ["consumer"] * 3,
        })
        ranked = _sector_rank(frame, np.ones(6))
        self.assertTrue(np.allclose(ranked, 0.0))

    def test_event_document_uses_next_trading_day(self):
        dataset = pd.DataFrame({
            "symbol": ["000001"] * 3,
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
            "eligible": [True] * 3,
            "broad_sector": ["finance_real_estate"] * 3,
            "benchmark_weight": [1.0] * 3,
            "event_target_1": [0.1, 0.2, 0.3],
            "event_target_5": [0.1, 0.2, 0.3],
            "event_target_20": [0.1, 0.2, 0.3],
            "event_label_end_1": pd.to_datetime(["2020-01-06"] * 3),
            "event_label_end_5": pd.to_datetime(["2020-01-10"] * 3),
            "event_label_end_20": pd.to_datetime(["2020-02-10"] * 3),
        })
        announcements = pd.DataFrame({
            "symbol": ["000001"],
            "announcement_date": pd.to_datetime(["2020-01-03"]),
            "title": ["关于股份回购的公告"],
            "announcement_id": ["a1"],
        })
        documents = build_event_documents(dataset, announcements)
        self.assertEqual(documents.loc[0, "date"], pd.Timestamp("2020-01-06"))
        self.assertIn("回购", documents.loc[0, "document"])

    def test_hashing_corpus_is_deterministic(self):
        events = pd.DataFrame({
            "symbol": ["000001", "000002"],
            "date": pd.to_datetime(["2018-01-02", "2018-01-02"]),
            "document": ["关于股份回购的公告", "关于股东减持的公告"],
        })
        settings = V15Settings(text_n_features=1024)
        first = EventTextCorpus.build(events, settings).matrix
        second = EventTextCorpus.build(events, settings).matrix
        self.assertEqual((first != second).nnz, 0)

    def test_multihorizon_text_model_smoke_and_non_event_zero(self):
        rows = []
        dates = pd.date_range("2017-01-03", periods=30, freq="14D")
        for index in range(600):
            positive = index % 2 == 0
            date = dates[index % len(dates)]
            signal = 0.03 if positive else -0.03
            rows.append({
                "symbol": f"{index % 60:06d}",
                "date": date,
                "document": "股份回购业绩预增" if positive else "股东减持风险提示",
                "event_count": 1,
                "eligible": True,
                "broad_sector": "technology" if index % 3 else "consumer",
                "benchmark_weight": 1 / 60,
                "event_target_1": signal,
                "event_target_5": signal * 1.5,
                "event_target_20": signal * 2,
                "event_label_end_1": date + pd.Timedelta(days=2),
                "event_label_end_5": date + pd.Timedelta(days=8),
                "event_label_end_20": date + pd.Timedelta(days=30),
            })
        events = pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)
        settings = V15Settings(text_n_features=2048, text_max_iter=10)
        corpus = EventTextCorpus.build(events, settings)
        model = MultiHorizonTextModel.fit(corpus, 2019, 2017, settings)
        current_date = events["date"].max()
        current = pd.DataFrame({
            "symbol": ["000000", "000001", "999999"],
            "date": [current_date] * 3,
            "broad_sector": ["consumer", "technology", "technology"],
            "eligible": [True] * 3,
        })
        scores = model.recent_scores(current, settings).set_index("symbol")
        self.assertTrue(np.isfinite(scores["text_score"]).all())
        self.assertEqual(scores.loc["999999", "text_score"], 0.0)
        self.assertEqual(scores.loc["999999", "text_events"], 0)
        self.assertEqual(model.event_years, [2017, 2018])

    @staticmethod
    def _events(count=100):
        return pd.DataFrame({
            "symbol": [f"{index:06d}" for index in range(count)],
            "date": [pd.Timestamp("2017-01-04")] * count,
            "eligible": [True] * count, "event_count": [1] * count,
            "document": ["股份回购公告"] * count, "broad_sector": ["technology"] * count,
            "event_target_1": [0.01] * count, "event_target_5": [0.02] * count,
            "event_target_20": [0.03] * count,
            "event_label_end_1": [pd.Timestamp("2017-01-06")] * count,
            "event_label_end_5": [pd.Timestamp("2017-01-12")] * count,
            "event_label_end_20": [pd.Timestamp("2017-02-10")] * count,
        })

    def test_quality_denominator_includes_all_eligible_incomplete_events(self):
        events = self._events(1000)
        events.loc[100:, "eligible"] = False
        events.loc[94:, ["event_target_1", "event_target_5", "event_target_20"]] = np.nan
        report = event_quality_report(events)
        self.assertEqual(report["eligible_event_documents"], 100)
        self.assertEqual(report["eligible_incomplete_documents"], 6)
        self.assertAlmostEqual(report["complete_three_target_ratio"], 0.094)
        self.assertAlmostEqual(report["eligible_complete_three_target_ratio"], 0.94)
        self.assertFalse(report["gates"]["eligible_complete_three_target_ratio_at_least_95pct"])

    def test_quality_rejects_empty_denominator_infinity_and_bad_chronology(self):
        events = self._events()
        events["eligible"] = False
        report = event_quality_report(events)
        self.assertFalse(report["gates"]["eligible_denominator_nonempty"])
        events["eligible"] = True
        events.loc[0, "event_target_5"] = np.inf
        events.loc[1, "event_label_end_20"] = pd.Timestamp("2016-01-01")
        report = event_quality_report(events)
        self.assertEqual(report["eligible_complete_documents"], 98)
        self.assertFalse(report["gates"]["label_chronology_valid"])

    def test_quality_keeps_all_event_ratio_when_eligible_gate_passes(self):
        events = self._events(1000)
        events.loc[100:, "eligible"] = False
        events.loc[99:, "event_target_20"] = np.nan
        report = event_quality_report(events)
        self.assertAlmostEqual(report["complete_three_target_ratio"], 0.099)
        self.assertAlmostEqual(report["eligible_complete_three_target_ratio"], 0.99)
        self.assertTrue(report["gates"]["eligible_complete_three_target_ratio_at_least_95pct"])

    def test_raw_year_gate_is_not_self_referential(self):
        required, passed = raw_year_gate([2017, 2018, 2019, 2020], [2019], 4)
        self.assertEqual(required, 4)
        self.assertFalse(passed)
        self.assertTrue(raw_year_gate([2017], [2017], 4)[1])
        self.assertFalse(raw_year_gate([], [], 4)[1])

    def test_training_requires_all_labels_mature_before_embargo(self):
        events = self._events(5)
        boundary = pd.Timestamp("2018-01-01") - pd.Timedelta(days=28)
        events.loc[1, "event_label_end_20"] = boundary
        events.loc[2, "event_label_end_5"] = boundary + pd.Timedelta(days=1)
        events.loc[3, "event_target_1"] = np.inf
        events.loc[4, "event_count"] = 0
        available, mature = event_training_masks(events, 2018, 2012, V15Settings())
        self.assertEqual(available.tolist(), [True, True, True, True, False])
        self.assertEqual(mature.tolist(), [True, False, False, False, False])

    def test_outer_validation_does_not_read_test_year_labels(self):
        boundary = pd.Timestamp("2020-01-01") - pd.Timedelta(days=28)
        dataset = pd.DataFrame({
            "date": pd.to_datetime(["2019-10-01"] * 3), "eligible": [True] * 3,
            "in_universe": [True] * 3, "future_return_20": [0.01] * 3,
            "v12_net_marginal_target": [0.01] * 3,
            "label_end_date_5": pd.to_datetime(["2019-10-08"] * 3),
            "label_end_date_20": [boundary - pd.Timedelta(days=1), boundary, pd.Timestamp("2020-01-05")],
        })
        self.assertEqual(validation_view(dataset, 2019, 2020, V15Settings()).index.tolist(), [0])
        self.assertEqual(baseline_training_view(dataset, 2020, V15Settings()).index.tolist(), [0])
        dataset.loc[0, "eligible"] = False
        dataset.loc[0, "v12_net_marginal_target"] = np.nan
        self.assertEqual(validation_view(dataset, 2019, 2020, V15Settings()).index.tolist(), [0])

    def test_validation_alignment_and_identical_baseline_blend(self):
        frame = pd.DataFrame({
            "symbol": ["000002", "000001", "000004", "000003"],
            "date": pd.to_datetime(["2019-01-03", "2019-01-02", "2019-01-03", "2019-01-02"]),
            "broad_sector": ["technology"] * 4,
            "probe": [0.04, 0.01, 0.03, 0.02], "text": [0.1, -0.2, 0.3, -0.4],
        }, index=[8, 3, 11, 2])
        baseline = Mock()
        baseline.predict_components.side_effect = lambda current: (np.ones(len(current)), current["probe"].to_numpy())
        text_model = Mock()
        text_model.recent_scores.side_effect = lambda current, settings: current[["symbol"]].assign(text_score=current["text"], text_events=1)
        with patch("research_v15.model.score_v10", side_effect=lambda current, *args: current.assign(global_model_score=current["probe"])):
            combined, base = _validation_predictions(frame, baseline, text_model, V15Settings(), (None, None, None))
        expected_base = 0.8 * _sector_rank(frame, frame["probe"]) + 0.2 * frame["probe"]
        np.testing.assert_allclose(base, expected_base.to_numpy())
        np.testing.assert_allclose(combined, (0.75 * expected_base + 0.25 * frame["text"]).to_numpy())

    def test_recent_text_uses_current_eligible_universe_and_sector(self):
        events = self._events(4)
        events.loc[1, "broad_sector"] = "old_sector"
        events.loc[2, "date"] = pd.Timestamp("2018-01-01")
        settings = V15Settings(text_n_features=64)
        corpus = EventTextCorpus.build(events, settings)
        model = MultiHorizonTextModel(corpus, [], np.zeros(3), np.ones(3), 0, [], [])
        current = pd.DataFrame({
            "symbol": ["000000", "000001", "000002", "000003", "999999"],
            "date": [pd.Timestamp("2017-01-05")] * 5,
            "broad_sector": ["technology"] * 5, "eligible": [True, True, True, False, True],
        })
        with patch.object(model, "_document_scores", return_value=np.array([[1, 1, 1], [2, 2, 2]])):
            scores = model.recent_scores(current, settings).set_index("symbol")
        self.assertAlmostEqual(scores.loc["000000", "text_score"], -0.25)
        self.assertAlmostEqual(scores.loc["000001", "text_score"], 0.25)
        for symbol in ["000002", "000003", "999999"]:
            self.assertEqual(scores.loc[symbol, "text_score"], 0)
            self.assertEqual(scores.loc[symbol, "text_events"], 0)

    def test_freeze_and_audit_refuse_existing_lock(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "artifacts/research_v15"
            directory.mkdir(parents=True)
            (directory / "plan.lock.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                freeze_research(root)
            with self.assertRaises(RuntimeError):
                audit_event_data(root=root)

    def test_run_rejects_nonfrozen_parameters_and_corpus(self):
        with self.assertRaises(RuntimeError):
            run_research_v15(settings=V15Settings(text_share=0.9))
        with self.assertRaises(RuntimeError):
            run_research_v15(event_path="other.csv")

    def test_drawdown_includes_initial_capital(self):
        self.assertAlmostEqual(max_drawdown(pd.Series([-0.1, 0.05])), -0.1)
        self.assertAlmostEqual(max_drawdown(pd.Series(dtype=float)), 0)

    def test_sector_relative_targets_only_defined_for_eligible_rows(self):
        frame = pd.DataFrame({
            "date": [pd.Timestamp("2017-01-04")] * 3,
            "broad_sector": ["technology"] * 3, "eligible": [True, True, False],
            "benchmark_weight": [0.2, 0.8, 0], "future": [0.1, 0.2, 0.9],
        })
        result = _official_sector_return(frame, "future")
        np.testing.assert_allclose(result.iloc[:2], [0.18, 0.18])
        self.assertTrue(np.isnan(result.iloc[2]))


if __name__ == "__main__":
    unittest.main()
