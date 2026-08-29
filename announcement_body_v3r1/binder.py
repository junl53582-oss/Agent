from __future__ import annotations

import re
from decimal import Decimal

from announcement_body_v2.binder import compact
from announcement_body_v3 import binder as parent


def _flash(document_id, parsed):
    result = []
    factor = "10000" if document_id.startswith("000728") else "1"
    for page in parsed["pages"]:
        for table_index, table in enumerate(page.get("tables", [])):
            if not table:
                continue
            width = max(len(row) for row in table)
            current_index, prior_index, change_index = (3, 6, 9) if width >= 10 else (3, 4, 5)
            balance = False
            for row_index, raw in enumerate(table):
                row = list(raw) + [None] * (width - len(raw))
                joined = compact("".join(str(cell or "") for cell in row))
                if "本报告期末" in joined:
                    balance = True
                if not row[current_index] or not row[prior_index] or not row[change_index]:
                    continue
                label = compact("".join(str(cell or "") for cell in row[:current_index]))
                lookahead = row_index + 1
                while lookahead < len(table):
                    following = list(table[lookahead]) + [None] * (width - len(table[lookahead]))
                    if following[current_index] or following[prior_index] or following[change_index]:
                        break
                    label += compact("".join(str(cell or "") for cell in following[:current_index]))
                    lookahead += 1
                if "扣除非经常性" in label:
                    metric = None
                elif "归属于上市公司股东的净利润" in label:
                    metric = "net_profit_parent"
                elif "营业总收入" in label:
                    metric = "operating_revenue"
                elif "基本每股收益" in label:
                    metric = "basic_eps"
                elif "总资产" in label.replace(" ", ""):
                    metric = "total_assets"
                elif "归属于上市公司股东的所有者权益" in label:
                    metric = "equity_parent"
                else:
                    metric = None
                if not metric:
                    continue
                value_factor = "1" if metric == "basic_eps" else factor
                currency = "CNY_per_share" if metric == "basic_eps" else "CNY"
                previous_role = "period_begin" if balance or metric in {"total_assets", "equity_parent"} else "prior_year_comparator"
                provenance = {"page": page["page"], "table_index": table_index, "row_index": row_index,
                              "continued_through_row": lookahead - 1,
                              "cells": {"current": current_index, "previous": prior_index, "change": change_index}}
                parent._add(result, document_id, page["page"], metric, "current_preliminary", parent._num(row[current_index], value_factor), currency, qualifier="preliminary_unaudited", provenance=provenance)
                parent._add(result, document_id, page["page"], metric, previous_role, parent._num(row[prior_index], value_factor), currency, qualifier="preliminary_unaudited", provenance=provenance)
                parent._add(result, document_id, page["page"], metric + "_change", "reported_change", parent._num(row[change_index], "0.01"), "ratio", qualifier="preliminary_unaudited", provenance=provenance)
    return result


def extract_document(document_id, parsed, category):
    if category == "earnings_flash":
        facts = _flash(document_id, parsed)
    else:
        facts = parent.extract_document(document_id, parsed, category)
    if document_id == "600481_1211571975":
        for page in parsed["pages"]:
            text = compact(page["text"])
            match = re.search(r"占公司2020年度(?:经)?审计营业收入比重为(?P<value>[\d.]+)%", text)
            if match and not any(item["metric"] == "contract_to_revenue" for item in facts):
                parent._add(facts, document_id, page["page"], "contract_to_revenue", "reference_ratio",
                            parent._num(match.group("value"), "0.01"), "ratio", qualifier="reference_revenue_year_2020", literal=match.group(0))
    if document_id == "000069_1209041295" and not any(item["metric"] == "repurchase_duration_months" for item in facts):
        text = "".join(compact(page["text"]) for page in parsed["pages"])
        match = re.search(r"实施期限为自.*?之日起(?P<value>\d+)个月内", text)
        parent._add(facts, document_id, 1, "repurchase_duration_months", "plan_duration", parent._num(match.group("value")), "count", qualifier="approved_plan", literal=match.group(0))
    if document_id == "000008_1205688090":
        for item in facts:
            if item["metric"] == "estimated_repurchase_ratio":
                item["value_base"] = str(Decimal(item["value_base"]).normalize())
    unique = {}
    for item in facts:
        key = tuple(item.get(name) for name in ("metric", "role", "bound", "value_base", "currency", "qualifier"))
        unique[key] = item
    return sorted(unique.values(), key=lambda item: (item["page"], item["metric"], item["role"], item["bound"]))

