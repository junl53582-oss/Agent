from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research_v15.features import load_event_documents, raw_event_years
from research_v15.quality import QUALITY_PATH
from research_v15.text_model import event_training_masks

from .config import V16Settings
from .text_model import EnsembleTextCorpus, EnsembleTextModel


def run_preflight():
    if Path("artifacts/research_v16/plan.lock.json").exists():
        raise RuntimeError("V16已冻结，禁止改写预检")
    quality = json.loads(Path(QUALITY_PATH).read_text(encoding="utf-8"))
    if not quality.get("passed"):
        raise RuntimeError("V15数据审计未通过，V16拒绝预检")
    settings = V16Settings()
    events = load_event_documents()
    print("V16 preflight: hashing char and jieba word corpus", flush=True)
    corpus = EnsembleTextCorpus.build(events, settings)
    folds = []
    for test_year in settings.test_years:
        for cutoff in range(test_year - settings.validation_years, test_year + 1):
            available, mature = event_training_masks(
                corpus.events, cutoff, test_year - settings.training_window_years, settings
            )
            folds.append({
                "test_year": test_year,
                "cutoff_year": cutoff,
                "training_events": int(mature.sum()),
                "available_raw_event_years": raw_event_years(corpus.events.loc[available]),
                "mature_raw_event_years": raw_event_years(corpus.events.loc[mature]),
            })
    print("V16 preflight: first-fold dual-head fitting, no return/IC evaluation", flush=True)
    model = EnsembleTextModel.fit(corpus, 2018, 2012, settings)
    passed = all(fold["training_events"] >= 500 for fold in folds)
    passed = passed and all(
        np.isfinite(regressor.coef_).all()
        for regressor in [*model.char_regressors, *model.word_regressors]
    )
    report = {
        "passed": bool(passed),
        "performance_evaluated": False,
        "char_corpus_shape": list(corpus.char_matrix.shape),
        "word_corpus_shape": list(corpus.word_matrix.shape),
        "char_nonzero_features": int(corpus.char_matrix.nnz),
        "word_nonzero_features": int(corpus.word_matrix.nnz),
        "char_sparse_matrix_bytes": int(
            corpus.char_matrix.data.nbytes + corpus.char_matrix.indices.nbytes + corpus.char_matrix.indptr.nbytes
        ),
        "word_sparse_matrix_bytes": int(
            corpus.word_matrix.data.nbytes + corpus.word_matrix.indices.nbytes + corpus.word_matrix.indptr.nbytes
        ),
        "first_fold_fitted_events": model.training_events,
        "folds": folds,
    }
    Path("artifacts/research_v16/preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
