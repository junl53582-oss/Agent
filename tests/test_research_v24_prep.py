import pytest

from research_v24_prep.contract import build_document_features, label_schedule, training_gate


def fact(metric, role, value, *, bound="point", currency="ratio", qualifier=None):
    return {"document_id": "000001_1", "page": 1, "metric": metric, "role": role, "bound": bound,
            "value_base": str(value), "currency": currency, "period": None, "qualifier": qualifier,
            "literal": "", "binding_approved": True}


def test_structured_features_preserve_role_and_compute_revision_without_labels():
    payload = {"gold_binding_passed": True, "facts": [
        fact("net_profit_parent", "current_guidance", 120, bound="lower", currency="CNY"),
        fact("net_profit_parent", "current_guidance", 140, bound="upper", currency="CNY"),
        fact("net_profit_parent", "previous_guidance", 100, bound="lower", currency="CNY"),
        fact("net_profit_parent", "previous_guidance", 120, bound="upper", currency="CNY"),
        fact("net_profit_parent", "prior_year_comparator", 100, currency="CNY"),
    ]}
    row = build_document_features(payload)[0]
    assert row["guidance_revision_delta"] == pytest.approx((130 - 110) / 110)
    assert row["current_vs_prior_delta"] == pytest.approx(0.3)


def test_unapproved_facts_and_failed_gold_are_rejected():
    with pytest.raises(ValueError, match="gold binding"):
        build_document_features({"gold_binding_passed": False, "facts": []})
    item = fact("x", "current_guidance", 1)
    item["binding_approved"] = False
    with pytest.raises(ValueError, match="unapproved"):
        build_document_features({"gold_binding_passed": True, "facts": [item]})


def test_labels_always_start_next_session_and_require_full_horizon():
    sessions = [f"2026-09-{day:02d}" for day in range(1, 26)]
    result = label_schedule("2026-09-01T09:00:00+08:00", sessions)
    assert result["entry_date"] == "2026-09-02"
    assert result["label_end_dates"] == {"5": "2026-09-07", "20": "2026-09-22"}
    late = label_schedule("2026-09-24T20:00:00+08:00", sessions)
    assert late["entry_date"] == "2026-09-25" and late["label_end_dates"]["5"] is None


def test_training_gate_requires_real_prospective_bodies_and_both_mature_targets():
    assert training_gate(prospective_events=0, approved_bodies=0, mature_5d=0, mature_20d=0)["model_training_ready"] is False
    assert training_gate(prospective_events=10, approved_bodies=10, mature_5d=10, mature_20d=9)["model_training_ready"] is False
    assert training_gate(prospective_events=10, approved_bodies=10, mature_5d=10, mature_20d=10)["model_training_ready"] is True

