"""Pairwise, target-specific diagnostics without tuning or selecting a winner."""
import numpy as np
import pandas as pd


COMPONENTS = ("global_model_score", "v13_baseline_score", "v13_comparable_score",
              "char_text_event_score", "text_event_score", "v16_score", "adaptive_score")
TARGETS = ("label_5", "v10_target_20", "v12_net_marginal_target", "future_return_20")
GROUPS = ("eligible", "technology", "event_covered", "no_recent_event")


def rank_ic(score, target):
    valid = np.isfinite(score) & np.isfinite(target)
    left, right = score[valid], target[valid]
    if len(left) < 3 or left.nunique() < 2 or right.nunique() < 2:
        return None, int(valid.sum())
    return float(left.corr(right, method="spearman")), int(valid.sum())


def component_diagnostics(frame):
    if frame.date.nunique() != 1 or frame.symbol.duplicated().any():
        raise ValueError("diagnostics require one date and unique symbols")
    eligible = frame[frame.eligible.eq(True)]
    subsets = {"eligible": eligible, "technology": eligible[eligible.broad_sector.eq("technology")],
               "event_covered": eligible[eligible.recent_text_events.gt(0)],
               "no_recent_event": eligible[eligible.recent_text_events.eq(0)]}
    rows = []
    for group, subset in subsets.items():
        for component in COMPONENTS:
            for target in TARGETS:
                value, count = rank_ic(subset[component], subset[target])
                rows.append({"date": frame.date.iloc[0], "test_year": int(frame.date.iloc[0].year),
                             "group": group, "component": component, "target": target,
                             "rank_ic": value, "labelled_rows": count, "eligible_rows": len(subset),
                             "distinct_scores": int(subset.loc[np.isfinite(subset[component]), component].nunique())})
    return pd.DataFrame(rows)


def check_legacy_reproduction(frame, reference):
    """Exactly reproduce the original *including* its label5 mask for IC20."""
    eligible = frame[frame.eligible.eq(True) & frame.label_5.notna()]
    tech = eligible[eligible.broad_sector.eq("technology")]
    evidence = []
    for mode, component in (("v16_control", "v16_score"), ("v20_adaptive", "adaptive_score")):
        original = reference[reference["mode"].eq(mode)]
        if len(original) != 1:
            raise ValueError("missing or duplicate parent reference")
        for name, subset, target in (("rank_ic_5", eligible, "label_5"),
                                     ("rank_ic_20", eligible, "v10_target_20"),
                                     ("technology_rank_ic_5", tech, "label_5")):
            value, count = rank_ic(subset[component], subset[target])
            expected = float(original.iloc[0][name])
            if not np.isclose(np.nan if value is None else value, expected, atol=1e-9, rtol=1e-7, equal_nan=True):
                raise ValueError(f"parent IC reproduction mismatch: {mode}/{name}: {value} != {expected}")
            evidence.append({"mode": mode, "metric": name, "value": value, "labelled_rows": count})
    return evidence


def summary_table(frame):
    keys = ["group", "component", "target"]
    # Equal weight per evaluation date, never pool all stock rows across years.
    aggregate = {"mean_rank_ic": ("rank_ic", "mean"), "valid_periods": ("rank_ic", "count"),
                 "periods": ("rank_ic", "size"), "labelled_rows": ("labelled_rows", "sum")}
    overall = frame.groupby(keys, sort=True).agg(**aggregate).reset_index()
    overall["test_year"] = "all"
    annual = frame.groupby(["test_year", *keys], sort=True).agg(**aggregate).reset_index()
    return pd.concat([overall, annual], ignore_index=True)
