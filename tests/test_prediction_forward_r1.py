import pandas as pd

from stockpilot.prediction_forward_r1 import attach_optional_ranking


def test_auxiliary_ranking_missing_rows_receive_neutral_without_dropping_probabilities(tmp_path) -> None:
    current = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-28", "2026-08-28"]),
        "symbol": ["000001", "000002"],
        "p_feature": [1.0, 2.0],
        "ranking_component": [0.1, 0.9],
    })
    ranking = pd.DataFrame({
        "date": ["2026-08-28"], "symbol": ["000001"], "score": [3.0],
    })
    path = tmp_path / "ranking.csv"
    ranking.to_csv(path, index=False)
    output, audit = attach_optional_ranking(current, path, "2026-08-28")
    assert len(output) == len(current)
    assert output.set_index("symbol").loc["000002", "ranking_component"] == 0.5
    assert audit["coverage"] == 0.5
    assert audit["used_by_probability_heads"] is False
    assert audit["missing_policy"] == "fixed neutral 0.5 for candidate_score only"


def test_auxiliary_ranking_requires_same_date(tmp_path) -> None:
    current = pd.DataFrame({"date": pd.to_datetime(["2026-08-28"]), "symbol": ["000001"]})
    path = tmp_path / "ranking.csv"
    pd.DataFrame({"date": ["2026-08-27"], "symbol": ["000001"], "score": [1.0]}).to_csv(
        path, index=False,
    )
    try:
        attach_optional_ranking(current, path, "2026-08-28")
    except RuntimeError as exc:
        assert "same-date" in str(exc)
    else:
        raise AssertionError("same-date auxiliary evidence must be required")
