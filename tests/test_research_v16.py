import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from research_v15.config import V15Settings
from research_v15.text_model import EventTextCorpus
from research_v16.config import V16Settings
from research_v16.model import (
    V15_REPLICA_BASELINE_SHARE,
    V15_REPLICA_TEXT_SHARE,
    _sector_rank,
    _validation_predictions,
    baseline_training_view,
    raw_year_gate,
    validation_view,
)
from research_v16.text_model import (
    EnsembleTextCorpus,
    EnsembleTextModel,
    _word_tokens,
)
from research_v16.freeze import freeze_research
from research_v16.validation import run_research_v16
from research_v16.backtest import MODES, max_drawdown


def _synthetic_events(count=600):
    rows = []
    dates = pd.date_range("2017-01-03", periods=30, freq="14D")
    for index in range(count):
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
    return pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)


class ResearchV16Tests(unittest.TestCase):
    def test_word_tokens_drop_blank_entries(self):
        tokens = _word_tokens("关于股份回购的公告  ")
        self.assertTrue(all(token.strip() for token in tokens))
        self.assertGreater(len(tokens), 2)

    def test_ensemble_corpus_is_deterministic(self):
        events = pd.DataFrame({
            "symbol": ["000001", "000002"],
            "date": pd.to_datetime(["2018-01-02", "2018-01-02"]),
            "document": ["关于股份回购的公告", "关于股东减持的公告"],
        })
        settings = V16Settings(text_n_features=1024, word_n_features=1024)
        first = EnsembleTextCorpus.build(events, settings)
        second = EnsembleTextCorpus.build(events, settings)
        self.assertEqual((first.char_matrix != second.char_matrix).nnz, 0)
        self.assertEqual((first.word_matrix != second.word_matrix).nnz, 0)

    def test_char_head_matrix_matches_frozen_v15_corpus(self):
        events = pd.DataFrame({
            "symbol": ["000001", "000002", "000003"],
            "date": pd.to_datetime(["2018-01-02"] * 3),
            "document": ["重大资产购买报告书摘要", "关于股份回购的公告", "股东减持计划"],
        })
        v15 = EventTextCorpus.build(events, V15Settings(text_n_features=2048))
        v16 = EnsembleTextCorpus.build(events, V16Settings(text_n_features=2048, word_n_features=2048))
        self.assertEqual((v15.matrix != v16.char_matrix).nnz, 0)
        self.assertEqual(list(v15.events.index), list(v16.events.index))

    def test_constant_sector_rank_is_exactly_zero(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-02"] * 6),
            "broad_sector": ["technology"] * 3 + ["consumer"] * 3,
        })
        ranked = _sector_rank(frame, np.ones(6))
        self.assertTrue(np.allclose(ranked, 0.0))

    def test_ensemble_model_smoke_and_non_event_zero(self):
        events = _synthetic_events()
        settings = V16Settings(text_n_features=2048, word_n_features=2048, text_max_iter=10)
        corpus = EnsembleTextCorpus.build(events, settings)
        model = EnsembleTextModel.fit(corpus, 2019, 2017, settings)
        current_date = events["date"].max()
        current = pd.DataFrame({
            "symbol": ["000000", "000001", "999999"],
            "date": [current_date] * 3,
            "broad_sector": ["consumer", "technology", "technology"],
            "eligible": [True] * 3,
        })
        scores = model.recent_scores(current, settings).set_index("symbol")
        self.assertTrue(np.isfinite(scores["text_score"]).all())
        self.assertTrue(np.isfinite(scores["char_text_score"]).all())
        self.assertEqual(scores.loc["999999", "text_score"], 0.0)
        self.assertEqual(scores.loc["999999", "char_text_score"], 0.0)
        self.assertEqual(scores.loc["999999", "text_events"], 0)
        self.assertEqual(model.event_years, [2017, 2018])

    def test_ensemble_blend_is_weighted_average_of_heads(self):
        events = _synthetic_events(40)
        settings = V16Settings(text_n_features=256, word_n_features=256)
        corpus = EnsembleTextCorpus.build(events, settings)
        model = EnsembleTextModel(corpus, [], [], np.zeros(3), np.ones(3), 0, [], [])
        with patch.object(model, "_char_scores", return_value=np.ones((3, 3))), \
                patch.object(model, "_word_scores", return_value=np.zeros((3, 3))):
            char, ensemble = model._document_scores(np.arange(3), settings)
        np.testing.assert_allclose(char, np.ones(3))
        np.testing.assert_allclose(ensemble, np.full(3, settings.char_word_blend))

    def test_recent_text_uses_current_eligible_universe_and_sector(self):
        events = pd.DataFrame({
            "symbol": [f"{index:06d}" for index in range(4)],
            "date": [pd.Timestamp("2017-01-04")] * 4,
            "eligible": [True] * 4, "event_count": [1] * 4,
            "document": ["股份回购公告"] * 4, "broad_sector": ["technology"] * 4,
            "event_target_1": [0.01] * 4, "event_target_5": [0.02] * 4,
            "event_target_20": [0.03] * 4,
            "event_label_end_1": [pd.Timestamp("2017-01-06")] * 4,
            "event_label_end_5": [pd.Timestamp("2017-01-12")] * 4,
            "event_label_end_20": [pd.Timestamp("2017-02-10")] * 4,
        })
        events.loc[1, "broad_sector"] = "old_sector"
        events.loc[2, "date"] = pd.Timestamp("2018-01-01")
        settings = V16Settings(text_n_features=64, word_n_features=64)
        corpus = EnsembleTextCorpus.build(events, settings)
        model = EnsembleTextModel(corpus, [], [], np.zeros(3), np.ones(3), 0, [], [])
        current = pd.DataFrame({
            "symbol": ["000000", "000001", "000002", "000003", "999999"],
            "date": [pd.Timestamp("2017-01-05")] * 5,
            "broad_sector": ["technology"] * 5, "eligible": [True, True, True, False, True],
        })
        with patch.object(model, "_document_scores", return_value=(np.array([1.0, 2.0]), np.array([1.0, 2.0]))):
            scores = model.recent_scores(current, settings).set_index("symbol")
        self.assertAlmostEqual(scores.loc["000000", "text_score"], -0.25)
        self.assertAlmostEqual(scores.loc["000001", "text_score"], 0.25)
        self.assertAlmostEqual(scores.loc["000000", "char_text_score"], -0.25)
        for symbol in ["000002", "000003", "999999"]:
            self.assertEqual(scores.loc[symbol, "text_score"], 0)
            self.assertEqual(scores.loc[symbol, "char_text_score"], 0)
            self.assertEqual(scores.loc[symbol, "text_events"], 0)

    def test_score_components_replica_and_v16_weights(self):
        frame = pd.DataFrame({
            "symbol": ["000002", "000001", "000004", "000003"],
            "date": pd.to_datetime(["2019-01-03", "2019-01-02", "2019-01-03", "2019-01-02"]),
            "broad_sector": ["technology"] * 4,
            "probe": [0.04, 0.01, 0.03, 0.02],
            "text": [0.1, -0.2, 0.3, -0.4],
            "char": [0.05, -0.1, 0.15, -0.2],
        }, index=[8, 3, 11, 2])
        baseline = Mock()
        baseline.predict_components.side_effect = lambda current: (np.ones(len(current)), current["probe"].to_numpy())
        text_model = Mock()
        text_model.recent_scores.side_effect = lambda current, settings: current[["symbol"]].assign(
            text_score=current["text"], char_text_score=current["char"], text_events=1
        )
        settings = V16Settings()
        from research_v16.model import _score_components
        with patch("research_v16.model.score_v10", side_effect=lambda current, *args: current.assign(global_model_score=current["probe"])):
            scored = _score_components(frame, baseline, text_model, (None, None, None), settings)
        expected_base = 0.8 * _sector_rank(frame, frame["probe"]) + 0.2 * frame["probe"]
        np.testing.assert_allclose(scored["v13_comparable_score"], expected_base.to_numpy())
        np.testing.assert_allclose(
            scored["v15_replica_score"],
            (V15_REPLICA_BASELINE_SHARE * expected_base + V15_REPLICA_TEXT_SHARE * frame["char"]).to_numpy(),
        )
        np.testing.assert_allclose(
            scored["v16_score"],
            (settings.baseline_share * expected_base + settings.text_share * frame["text"]).to_numpy(),
        )
        self.assertEqual(V15_REPLICA_BASELINE_SHARE, 0.75)
        self.assertEqual(V15_REPLICA_TEXT_SHARE, 0.25)
        self.assertEqual(settings.baseline_share, 0.65)
        self.assertEqual(settings.text_share, 0.35)

    def test_validation_alignment_returns_v16_blend(self):
        frame = pd.DataFrame({
            "symbol": ["000002", "000001", "000004", "000003"],
            "date": pd.to_datetime(["2019-01-03", "2019-01-02", "2019-01-03", "2019-01-02"]),
            "broad_sector": ["technology"] * 4,
            "probe": [0.04, 0.01, 0.03, 0.02],
            "text": [0.1, -0.2, 0.3, -0.4],
            "char": [0.05, -0.1, 0.15, -0.2],
        }, index=[8, 3, 11, 2])
        baseline = Mock()
        baseline.predict_components.side_effect = lambda current: (np.ones(len(current)), current["probe"].to_numpy())
        text_model = Mock()
        text_model.recent_scores.side_effect = lambda current, settings: current[["symbol"]].assign(
            text_score=current["text"], char_text_score=current["char"], text_events=1
        )
        settings = V16Settings()
        with patch("research_v16.model.score_v10", side_effect=lambda current, *args: current.assign(global_model_score=current["probe"])):
            combined, base = _validation_predictions(frame, baseline, text_model, settings, (None, None, None))
        expected_base = 0.8 * _sector_rank(frame, frame["probe"]) + 0.2 * frame["probe"]
        np.testing.assert_allclose(base, expected_base.to_numpy())
        np.testing.assert_allclose(combined, (0.65 * expected_base + 0.35 * frame["text"]).to_numpy())

    def test_raw_year_gate_is_not_self_referential(self):
        required, passed = raw_year_gate([2017, 2018, 2019, 2020], [2019], 4)
        self.assertEqual(required, 4)
        self.assertFalse(passed)
        self.assertTrue(raw_year_gate([2017], [2017], 4)[1])
        self.assertFalse(raw_year_gate([], [], 4)[1])

    def test_outer_validation_does_not_read_test_year_labels(self):
        boundary = pd.Timestamp("2020-01-01") - pd.Timedelta(days=28)
        dataset = pd.DataFrame({
            "date": pd.to_datetime(["2019-10-01"] * 3), "eligible": [True] * 3,
            "in_universe": [True] * 3, "future_return_20": [0.01] * 3,
            "v12_net_marginal_target": [0.01] * 3,
            "label_end_date_5": pd.to_datetime(["2019-10-08"] * 3),
            "label_end_date_20": [boundary - pd.Timedelta(days=1), boundary, pd.Timestamp("2020-01-05")],
        })
        self.assertEqual(validation_view(dataset, 2019, 2020, V16Settings()).index.tolist(), [0])
        self.assertEqual(baseline_training_view(dataset, 2020, V16Settings()).index.tolist(), [0])
        dataset.loc[0, "eligible"] = False
        dataset.loc[0, "v12_net_marginal_target"] = np.nan
        self.assertEqual(validation_view(dataset, 2019, 2020, V16Settings()).index.tolist(), [0])

    def test_modes_include_char_replica_anchor(self):
        self.assertEqual(
            MODES,
            ("core", "v13_comparable", "v15_char_replica", "v16_text_ungated", "v16_text_gated"),
        )

    def test_freeze_refuses_existing_lock(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "artifacts/research_v16"
            directory.mkdir(parents=True)
            (directory / "plan.lock.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                freeze_research(root)

    def test_run_rejects_nonfrozen_parameters_and_corpus(self):
        with self.assertRaises(RuntimeError):
            run_research_v16(settings=V16Settings(text_share=0.9))
        with self.assertRaises(RuntimeError):
            run_research_v16(event_path="other.csv")

    def test_drawdown_includes_initial_capital(self):
        self.assertAlmostEqual(max_drawdown(pd.Series([-0.1, 0.05])), -0.1)
        self.assertAlmostEqual(max_drawdown(pd.Series(dtype=float)), 0)


if __name__ == "__main__":
    unittest.main()
