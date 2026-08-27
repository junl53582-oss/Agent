from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import V15Settings
from .features import load_event_documents, raw_event_years
from .quality import QUALITY_PATH
from .text_model import EventTextCorpus, MultiHorizonTextModel, event_training_masks


def run_preflight():
    if Path("artifacts/research_v15/plan.lock.json").exists():
        raise RuntimeError("V15已冻结，禁止改写预检")
    quality = json.loads(Path(QUALITY_PATH).read_text(encoding="utf-8"))
    if not quality.get("passed"):
        raise RuntimeError("V15数据审计未通过")
    settings = V15Settings()
    events = load_event_documents()
    print("V15 preflight: hashing title corpus", flush=True)
    corpus = EventTextCorpus.build(events, settings)
    folds = []
    for test_year in settings.test_years:
        for cutoff in range(test_year - settings.validation_years, test_year + 1):
            available, mature = event_training_masks(corpus.events, cutoff, test_year - settings.training_window_years, settings)
            folds.append({"test_year": test_year, "cutoff_year": cutoff, "training_events": int(mature.sum()),
                          "available_raw_event_years": raw_event_years(corpus.events.loc[available]),
                          "mature_raw_event_years": raw_event_years(corpus.events.loc[mature])})
    print("V15 preflight: first-fold text fitting, no return/IC evaluation", flush=True)
    model = MultiHorizonTextModel.fit(corpus, 2018, 2012, settings)
    passed = all(fold["training_events"] >= 500 for fold in folds)
    passed = passed and all(np.isfinite(reg.coef_).all() for reg in model.regressors)
    report = {
        "passed": bool(passed), "performance_evaluated": False,
        "corpus_shape": list(corpus.matrix.shape), "nonzero_features": int(corpus.matrix.nnz),
        "sparse_matrix_bytes": int(corpus.matrix.data.nbytes + corpus.matrix.indices.nbytes + corpus.matrix.indptr.nbytes),
        "first_fold_fitted_events": model.training_events,
        "folds": folds,
    }
    Path("artifacts/research_v15/preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
