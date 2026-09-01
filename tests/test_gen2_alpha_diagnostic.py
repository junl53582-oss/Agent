from __future__ import annotations

import numpy as np
import pandas as pd

from stockpilot.alpha_diagnostic.gen2 import (
    DiagnosticSettings,
    _block_bootstrap,
    _ic_summary,
    _select_sector_balanced,
)


def test_ic_summary_reports_rank_and_pearson_statistics() -> None:
    daily = pd.DataFrame(
        {"rank_ic": [0.1, -0.1, 0.2], "pearson_ic": [0.2, 0.0, 0.1]}
    )
    summary = _ic_summary(daily)
    assert summary["dates"] == 3
    assert np.isclose(summary["rank_ic_mean"], 0.2 / 3)
    assert np.isclose(summary["pearson_ic_mean"], 0.1)
    assert np.isclose(summary["positive_ic_ratio"], 2 / 3)


def test_block_bootstrap_is_deterministic() -> None:
    values = pd.Series(np.linspace(-0.1, 0.2, 100))
    settings = DiagnosticSettings(bootstrap_replications=50, bootstrap_block_length=5)
    assert _block_bootstrap(values, settings) == _block_bootstrap(values, settings)


def test_sector_balanced_selection_respects_k_and_is_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "symbol": [f"{index:06d}" for index in range(12)],
            "score_lightgbm": np.arange(12, dtype=float),
            "broad_sector": ["A"] * 8 + ["B"] * 4,
        }
    )
    first = _select_sector_balanced(frame, 5)
    second = _select_sector_balanced(frame.sample(frac=1, random_state=4), 5)
    assert len(first) == 5
    assert first["symbol"].tolist() == second["symbol"].tolist()
