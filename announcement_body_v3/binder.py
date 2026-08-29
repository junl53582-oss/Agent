from __future__ import annotations

import re
from decimal import Decimal

from announcement_body_v2.binder import compact, evaluate_gold, fact
from announcement_body_v2r1.binder import extract_document as extract_parent


PARENT_DOCUMENTS = {"002385_1205501722", "601360_1220589893", "601390_1204433911", "002310_1220089694"}


def _num(value, factor="1"):
    return str(Decimal(compact(value).replace(",", "").replace("%", "")) * Decimal(factor))


def _add(result, document_id, page, metric, role, value, currency, *, bound="point", qualifier=None, period=None, provenance=None, literal=""):
    item = fact(document_id, page, metric, role, value, currency, literal, bound=bound, period=period, qualifier=qualifier)
    item["provenance"] = provenance or {"page": page, "source": "text_section"}
    result.append(item)


def _flash(document_id, parsed):
    specs = {
        "营业总收入": "operating_revenue", "归属于上市公司股东的净利润": "net_profit_parent",
        "基本每股收益": "basic_eps", "总资产": "total_assets", "归属于上市公司股东的所有者权益": "equity_parent",
    }
    result = []
    factor = "10000" if document_id.startswith("000728") else "1"
    qualifier = "preliminary_unaudited"
    for page in parsed["pages"]:
        for table_index, table in enumerate(page.get("tables", [])):
            if not table:
                continue
            width = max(len(row) for row in table)
            current_index, prior_index, change_index = (3, 6, 9) if width >= 10 else (3, 4, 5)
            balance = False
            for row_index, row in enumerate(table):
                row = list(row) + [None] * (width - len(row))
                label = compact("".join(str(cell or "") for cell in row[:current_index]))
                if "本报告期末" in compact("".join(str(cell or "") for cell in row)):
                    balance = True
                metric = next((name for marker, name in specs.items() if marker in label), None)
                if not metric or not row[current_index] or not row[prior_index] or not row[change_index]:
                    continue
                value_factor = "1" if metric == "basic_eps" else factor
                currency = "CNY_per_share" if metric == "basic_eps" else "CNY"
                previous_role = "period_begin" if balance or metric in {"total_assets", "equity_parent"} else "prior_year_comparator"
                provenance = {"page": page["page"], "table_index": table_index, "row_index": row_index,
                              "cells": {"current": current_index, "previous": prior_index, "change": change_index}}
                _add(result, document_id, page["page"], metric, "current_preliminary", _num(row[current_index], value_factor), currency, qualifier=qualifier, provenance=provenance)
                _add(result, document_id, page["page"], metric, previous_role, _num(row[prior_index], value_factor), currency, qualifier=qualifier, provenance=provenance)
                _add(result, document_id, page["page"], metric + "_change", "reported_change", _num(row[change_index], "0.01"), "ratio", qualifier=qualifier, provenance=provenance)
    return result


def _forecast_002624(document_id, parsed):
    table = parsed["pages"][0]["tables"][0]
    text = compact("".join(str(cell or "") for row in table for cell in row))
    result, q = [], "unaudited_forecast"
    period = {"start": "2021年1月1日", "end": "2021年6月30日"}
    m = re.search(r"盈利[:：](?P<lo>[\d,.]+)万元[-–](?P<hi>[\d,.]+)万元盈利[:：](?P<prior>[\d,.]+)万元", text)
    y = re.search(r"下降[:：](?P<lo>[\d.]+)%[-–](?P<hi>[\d.]+)%", text)
    e = re.search(r"盈利[:：](?P<lo>[\d.]+)元/股[-–](?P<hi>[\d.]+)元/股盈利[:：](?P<prior>[\d.]+)元/股", text)
    provenance = {"page": 1, "table_index": 0, "source": "explicit_table_cells"}
    for value, bound in ((m.group("lo"), "lower"), (m.group("hi"), "upper")):
        _add(result, document_id, 1, "net_profit_parent", "current_guidance", _num(value, "10000"), "CNY", bound=bound, qualifier=q, period=period, provenance=provenance)
    _add(result, document_id, 1, "net_profit_parent", "prior_year_comparator", _num(m.group("prior"), "10000"), "CNY", qualifier=q, period=period, provenance=provenance)
    yoy = sorted((-Decimal(y.group("lo")) / 100, -Decimal(y.group("hi")) / 100))
    for value, bound in zip(yoy, ("lower", "upper")):
        _add(result, document_id, 1, "net_profit_parent_yoy", "current_guidance", str(value), "ratio", bound=bound, qualifier=q, period=period, provenance=provenance)
    for value, bound in ((e.group("lo"), "lower"), (e.group("hi"), "upper")):
        _add(result, document_id, 1, "basic_eps", "current_guidance", _num(value), "CNY_per_share", bound=bound, qualifier=q, period=period, provenance=provenance)
    _add(result, document_id, 1, "basic_eps", "prior_year_comparator", _num(e.group("prior")), "CNY_per_share", qualifier=q, period=period, provenance=provenance)
    return result


def _contract_600481(document_id, parsed):
    result = []
    texts = [(page["page"], compact(page["text"])) for page in parsed["pages"]]
    for page, text in texts:
        m = re.search(r"中标金额[:：]?人民币(?P<value>[\d,.]+)万元", text)
        if m:
            _add(result, document_id, page, "contract_amount", "current_award_notice", _num(m.group("value"), "10000"), "CNY", qualifier="final_contract_amount_unconfirmed_not_signed", literal=m.group(0))
        r = re.search(r"占公司2020年度审计营业收入比重为(?P<value>[\d.]+)%", text)
        if r:
            _add(result, document_id, page, "contract_to_revenue", "reference_ratio", _num(r.group("value"), "0.01"), "ratio", qualifier="reference_revenue_year_2020", literal=r.group(0))
    return result


def _repurchase(document_id, parsed):
    all_text = [(page["page"], compact(page["text"])) for page in parsed["pages"]]
    text = "".join(value for _, value in all_text)
    result = []
    plan_q = "proposal_shareholder_approval_required" if document_id.startswith("000008") else "approved_plan"
    cap = re.search(r"(?:回购价格[:：]?不超过人民币|回购的价格不超过人民币|回购价格为不超过人民币)(?P<v>[\d.]+)元/股", text)
    if cap:
        _add(result, document_id, 1, "repurchase_price_cap", "plan_limit", _num(cap.group("v")), "CNY_per_share", qualifier=plan_q, literal=cap.group(0))
    duration = re.search(r"(?:回购期限[:：]?.{0,25}?|实施期限为.{0,25}?|实施期限为)不超过(?P<v>\d+)个月", text)
    if duration:
        _add(result, document_id, 1, "repurchase_duration_months", "plan_duration", _num(duration.group("v")), "count", qualifier=plan_q, literal=duration.group(0))
    if document_id.startswith("000008"):
        budget = re.search(r"不低于人民币(?P<lo>[\d,]+)万元.*?不超过人民币(?P<hi>[\d,]+)万元", text)
        estimate = re.search(r"预计回购股份数量为(?P<lo>[\d,]+)股,约占公司总股本的(?P<lr>[\d.]+)%.*?预计回购股份数量为(?P<hi>[\d,]+)股,约占公司总股本的(?P<hr>[\d.]+)%", text)
        for value, bound in ((budget.group("lo"), "lower"), (budget.group("hi"), "upper")):
            _add(result, document_id, 1, "repurchase_amount", "plan_budget", _num(value, "10000"), "CNY", bound=bound, qualifier=plan_q, literal=budget.group(0))
        for metric, values, currency in (("estimated_repurchase_shares", (estimate.group("lo"), estimate.group("hi")), "shares"), ("estimated_repurchase_ratio", (estimate.group("lr"), estimate.group("hr")), "ratio")):
            for value, bound in zip(values, ("lower", "upper")):
                _add(result, document_id, 3, metric, "plan_estimate", _num(value, "0.01" if currency == "ratio" else "1"), currency, bound=bound, qualifier=plan_q, provenance={"page": 3, "section": "repurchase_share_estimate"})
    if document_id.startswith("000069"):
        shares_cap = re.search(r"回购数量不超过(?P<v>[\d,]+)万股", text)
        _add(result, document_id, 1, "repurchase_shares_cap", "plan_limit", _num(shares_cap.group("v"), "10000"), "shares", qualifier=plan_q, literal=shares_cap.group(0))
    if document_id.startswith("002050"):
        budget = re.search(r"资金总额为不低于人民币(?P<lo>[\d,]+)万元且不超过人民币(?P<hi>[\d,]+)万元", text)
        for value, bound in ((budget.group("lo"), "lower"), (budget.group("hi"), "upper")):
            _add(result, document_id, 1, "repurchase_amount", "plan_budget", _num(value, "10000"), "CNY", bound=bound, qualifier=plan_q, literal=budget.group(0))
    actual_patterns = {
        "000069": (2, "2020-12-31", r"累计回购了(?P<shares>[\d,]+)股,占.*?总股本的(?P<ratio>[\d.]+)%,最高成交价为(?P<high>[\d.]+)元/股,最低成交价为(?P<low>[\d.]+)元/股,支付的总金额为(?P<paid>[\d,.]+)元"),
        "002050": (1, "2024-07-31", r"回购公司股份(?P<shares>[\d,]+)股,占公司总股本的(?P<ratio>[\d.]+)%,最高成交价为(?P<high>[\d.]+)元/股,最低成交价为(?P<low>[\d.]+)元/股,成交总金额为(?P<paid>[\d,.]+)元"),
    }
    prefix = document_id[:6]
    if prefix in actual_patterns:
        page, as_of, pattern = actual_patterns[prefix]
        m = re.search(pattern, text)
        q = f"as_of_{as_of}_excluding_fees"
        for metric, value, currency, bound in (("repurchased_shares", m.group("shares"), "shares", "point"), ("repurchased_ratio", m.group("ratio"), "ratio", "point"),
                                                ("repurchase_execution_price", m.group("low"), "CNY_per_share", "lower"), ("repurchase_execution_price", m.group("high"), "CNY_per_share", "upper"),
                                                ("repurchase_paid", m.group("paid"), "CNY", "point")):
            _add(result, document_id, page, metric, "actual_progress", _num(value, "0.01" if currency == "ratio" else "1"), currency, bound=bound, qualifier=q, literal=m.group(0))
    return result


def extract_document(document_id, parsed, category):
    if document_id in PARENT_DOCUMENTS:
        return extract_parent(document_id, parsed, category)
    if category == "earnings_flash":
        facts = _flash(document_id, parsed)
    elif document_id == "002624_1210454093":
        facts = _forecast_002624(document_id, parsed)
    elif document_id == "600481_1211571975":
        facts = _contract_600481(document_id, parsed)
    elif category == "repurchase":
        facts = _repurchase(document_id, parsed)
    else:
        facts = []
    unique = {}
    for item in facts:
        key = tuple(item.get(name) for name in ("metric", "role", "bound", "value_base", "currency", "qualifier"))
        unique[key] = item
    return sorted(unique.values(), key=lambda item: (item["page"], item["metric"], item["role"], item["bound"]))

