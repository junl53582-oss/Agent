from pathlib import Path

import pandas as pd
import pytest

from stockpilot.prediction_forward import (
    ForwardPredictionSettings,
    compare_feature_panel,
    stitch_hfq_market,
)


def _market(scale: float = 1.0) -> pd.DataFrame:
    rows = []
    for symbol in ("000001", "000002"):
        for index, date in enumerate(pd.bdate_range("2026-08-03", "2026-08-28")):
            value = scale * (10 + index + (symbol == "000002"))
            rows.append({
                "date": date, "symbol": symbol, "open": value, "high": value * 1.01,
                "low": value * 0.99, "close": value, "volume": 1000 + index,
                "amount": (1000 + index) * value,
            })
    return pd.DataFrame(rows)


def _membership() -> pd.DataFrame:
    return pd.DataFrame({
        "snapshot_date": [pd.Timestamp("2026-06-30")] * 2,
        "index_code": ["000300"] * 2,
        "symbol": ["000001", "000002"],
        "weight": [0.5, 0.5],
        "source": ["test"] * 2,
    })


def test_hfq_stitch_preserves_frozen_scale_and_adds_future() -> None:
    frozen = _market().query("date <= '2026-08-21'")
    incremental = _market(scale=2.0)
    settings = ForwardPredictionSettings(minimum_current_coverage=1.0)
    combined, audit = stitch_hfq_market(
        frozen, incremental, _membership(), cutoff="2026-08-21", as_of="2026-08-28",
        settings=settings,
    )
    old = combined.loc[
        combined["date"].eq(pd.Timestamp("2026-08-21")) & combined["symbol"].eq("000001"),
        "close",
    ].iloc[0]
    new = combined.loc[
        combined["date"].eq(pd.Timestamp("2026-08-24")) & combined["symbol"].eq("000001"),
        "close",
    ].iloc[0]
    assert old == 24.0
    assert new == 25.0
    assert audit["passed"] is True
    assert audit["latest_current_coverage"] == 1.0


def test_hfq_stitch_rejects_unstable_overlap() -> None:
    frozen = _market().query("date <= '2026-08-21'")
    incremental = _market(scale=2.0)
    incremental.loc[
        (incremental["symbol"] == "000001") & (incremental["date"] == pd.Timestamp("2026-08-10")),
        "close",
    ] *= 1.2
    with pytest.raises(RuntimeError, match="failed overlap consistency"):
        stitch_hfq_market(
            frozen, incremental, _membership(), cutoff="2026-08-21", as_of="2026-08-28",
            settings=ForwardPredictionSettings(minimum_current_coverage=1.0),
        )


def test_hfq_stitch_isolates_unstable_noncurrent_symbol() -> None:
    frozen = _market().query("date <= '2026-08-21'")
    incremental = _market(scale=2.0)
    membership = _membership().query("symbol == '000002'")
    incremental.loc[
        (incremental["symbol"] == "000001") & (incremental["date"] == pd.Timestamp("2026-08-10")),
        "close",
    ] *= 1.2
    combined, audit = stitch_hfq_market(
        frozen, incremental, membership, cutoff="2026-08-21", as_of="2026-08-28",
        settings=ForwardPredictionSettings(minimum_current_coverage=1.0),
    )
    assert "000001" in audit["isolated_noncurrent_symbols"]
    assert combined.loc[combined["date"].gt(pd.Timestamp("2026-08-21")), "symbol"].eq("000001").sum() == 0


def test_feature_parity_detects_numeric_difference(tmp_path: Path) -> None:
    expected = pd.DataFrame({
        "symbol": ["000001"], "open": [10.0], "close": [10.0],
        "volatility_20": [0.1], "broad_sector": ["finance_real_estate"],
        "regime": ["neutral"], **{column: [0.0] for column in __import__(
            "research_v10.features", fromlist=["V10_FEATURES"]
        ).V10_FEATURES},
    })
    path = tmp_path / "expected.csv"
    expected.to_csv(path, index=False)
    actual = expected.copy()
    assert compare_feature_panel(actual, path)["passed"] is True
    actual.loc[0, "close"] = 11.0
    assert compare_feature_panel(actual, path)["passed"] is False


def test_forward_contract_never_authorizes_execution() -> None:
    source = Path("stockpilot/prediction_forward.py").read_text(encoding="utf-8")
    assert 'output["execution_authorized"] = False' in source
    assert '"production_prediction_ready_may_not_be_promoted": True' in source
    assert 'dataset.loc[dataset["symbol"].isin(full_history_bad), "eligible"] = False' in source
