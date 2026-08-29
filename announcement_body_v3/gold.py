from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _fact(document_id, page, metric, role, value, currency, *, bound="point", qualifier=None, period=None, provenance=None):
    return {"document_id": document_id, "page": page, "metric": metric, "role": role, "bound": bound,
            "value_base": str(value), "currency": currency, "qualifier": qualifier, "period": period,
            "provenance": provenance or {"page": page}, "annotation_source": "manual_full_page_visual_review"}


def expanded_gold():
    parent = json.loads((ROOT / "artifacts/announcement_body_v2/gold.json").read_text(encoding="utf-8"))
    facts = list(parent["facts"])
    add = facts.append

    q = "proposal_shareholder_approval_required"
    add(_fact("000008_1205688090", 1, "repurchase_price_cap", "plan_limit", "5.99", "CNY_per_share", qualifier=q))
    add(_fact("000008_1205688090", 1, "repurchase_amount", "plan_budget", "300000000", "CNY", bound="lower", qualifier=q))
    add(_fact("000008_1205688090", 1, "repurchase_amount", "plan_budget", "600000000", "CNY", bound="upper", qualifier=q))
    add(_fact("000008_1205688090", 1, "repurchase_duration_months", "plan_duration", "12", "count", qualifier=q))
    add(_fact("000008_1205688090", 3, "estimated_repurchase_shares", "plan_estimate", "50083472", "shares", bound="lower", qualifier=q))
    add(_fact("000008_1205688090", 3, "estimated_repurchase_shares", "plan_estimate", "100166944", "shares", bound="upper", qualifier=q))
    add(_fact("000008_1205688090", 3, "estimated_repurchase_ratio", "plan_estimate", "0.018", "ratio", bound="lower", qualifier=q))
    add(_fact("000008_1205688090", 3, "estimated_repurchase_ratio", "plan_estimate", "0.036", "ratio", bound="upper", qualifier=q))

    q = "final_contract_amount_unconfirmed_not_signed"
    add(_fact("600481_1211571975", 1, "contract_amount", "current_award_notice", "178600000", "CNY", qualifier=q))
    add(_fact("600481_1211571975", 2, "contract_to_revenue", "reference_ratio", "0.0862", "ratio", qualifier="reference_revenue_year_2020"))

    flash = [
        ("600674_1211243407", 1, "operating_revenue", "893055276.43", "650469281.00", "0.3729", "CNY", "prior_year_comparator"),
        ("600674_1211243407", 1, "net_profit_parent", "2779626204.57", "2708883809.00", "0.0261", "CNY", "prior_year_comparator"),
        ("600674_1211243407", 1, "basic_eps", "0.6311", "0.6154", "0.0255", "CNY_per_share", "prior_year_comparator"),
        ("600674_1211243407", 2, "total_assets", "45117089373.28", "41347852404.15", "0.0912", "CNY", "period_begin"),
        ("600674_1211243407", 2, "equity_parent", "31313017785.12", "28795635611.60", "0.0874", "CNY", "period_begin"),
        ("000728_1220627969", 1, "operating_revenue", "3084447300.00", "3110841500.00", "-0.0085", "CNY", "prior_year_comparator"),
        ("000728_1220627969", 1, "net_profit_parent", "1000169300.00", "913917300.00", "0.0944", "CNY", "prior_year_comparator"),
        ("000728_1220627969", 1, "basic_eps", "0.23", "0.21", "0.0952", "CNY_per_share", "prior_year_comparator"),
        ("000728_1220627969", 1, "total_assets", "149899188800.00", "132855982500.00", "0.1283", "CNY", "period_begin"),
        ("000728_1220627969", 1, "equity_parent", "35646559000.00", "34578867300.00", "0.0309", "CNY", "period_begin"),
    ]
    for document_id, page, metric, current, previous, change, currency, previous_role in flash:
        provenance = {"page": page, "table_index": 0, "metric_row": metric}
        add(_fact(document_id, page, metric, "current_preliminary", current, currency, qualifier="preliminary_unaudited", provenance=provenance))
        add(_fact(document_id, page, metric, previous_role, previous, currency, qualifier="preliminary_unaudited", provenance=provenance))
        add(_fact(document_id, page, metric + "_change", "reported_change", change, "ratio", qualifier="preliminary_unaudited", provenance=provenance))

    q = "unaudited_forecast"
    add(_fact("002624_1210454093", 1, "net_profit_parent", "current_guidance", "230000000", "CNY", bound="lower", qualifier=q))
    add(_fact("002624_1210454093", 1, "net_profit_parent", "current_guidance", "270000000", "CNY", bound="upper", qualifier=q))
    add(_fact("002624_1210454093", 1, "net_profit_parent", "prior_year_comparator", "1270597800.00", "CNY", qualifier=q))
    add(_fact("002624_1210454093", 1, "net_profit_parent_yoy", "current_guidance", "-0.819", "ratio", bound="lower", qualifier=q))
    add(_fact("002624_1210454093", 1, "net_profit_parent_yoy", "current_guidance", "-0.7875", "ratio", bound="upper", qualifier=q))
    add(_fact("002624_1210454093", 1, "basic_eps", "current_guidance", "0.12", "CNY_per_share", bound="lower", qualifier=q))
    add(_fact("002624_1210454093", 1, "basic_eps", "current_guidance", "0.14", "CNY_per_share", bound="upper", qualifier=q))
    add(_fact("002624_1210454093", 1, "basic_eps", "prior_year_comparator", "0.66", "CNY_per_share", qualifier=q))

    for document_id, page, cap, max_shares, duration, as_of, shares, ratio, low, high, paid in [
        ("000069_1209041295", 1, "8", "246080000", "12", "2020-12-31", "162847162", "0.01985", "5.84", "7.04", "1033099334.60"),
        ("002050_1220789517", 1, "36.00", None, "12", "2024-07-31", "13961794", "0.003740", "19.81", "29.09", "319919680.07"),
    ]:
        add(_fact(document_id, page, "repurchase_price_cap", "plan_limit", cap, "CNY_per_share", qualifier="approved_plan"))
        if max_shares:
            add(_fact(document_id, page, "repurchase_shares_cap", "plan_limit", max_shares, "shares", qualifier="approved_plan"))
        if document_id.startswith("002050"):
            add(_fact(document_id, page, "repurchase_amount", "plan_budget", "200000000", "CNY", bound="lower", qualifier="approved_plan"))
            add(_fact(document_id, page, "repurchase_amount", "plan_budget", "400000000", "CNY", bound="upper", qualifier="approved_plan"))
        add(_fact(document_id, page, "repurchase_duration_months", "plan_duration", duration, "count", qualifier="approved_plan"))
        aq = f"as_of_{as_of}_excluding_fees"
        add(_fact(document_id, 2 if document_id.startswith("000069") else 1, "repurchased_shares", "actual_progress", shares, "shares", qualifier=aq))
        add(_fact(document_id, 2 if document_id.startswith("000069") else 1, "repurchased_ratio", "actual_progress", ratio, "ratio", qualifier=aq))
        add(_fact(document_id, 2 if document_id.startswith("000069") else 1, "repurchase_execution_price", "actual_progress", low, "CNY_per_share", bound="lower", qualifier=aq))
        add(_fact(document_id, 2 if document_id.startswith("000069") else 1, "repurchase_execution_price", "actual_progress", high, "CNY_per_share", bound="upper", qualifier=aq))
        add(_fact(document_id, 2 if document_id.startswith("000069") else 1, "repurchase_paid", "actual_progress", paid, "CNY", qualifier=aq))

    return {"documents": parent["documents"] + ["000008_1205688090", "600481_1211571975", "600674_1211243407",
            "002624_1210454093", "000069_1209041295", "000728_1220627969", "002050_1220789517"],
            "facts": facts, "annotation_method": "Manual visual review of all pages of eleven text PDFs",
            "historical_pit_verified": False, "model_training_ready": False, "execution_authorized": False}

