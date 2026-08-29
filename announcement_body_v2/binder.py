from __future__ import annotations

import re
import unicodedata
from decimal import Decimal


MONEY_FACTORS = {"亿元": Decimal("100000000"), "万元": Decimal("10000"), "元": Decimal("1")}


def compact(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).replace("−", "-").replace("—", "-"))


def amount(number, unit):
    return str(Decimal(number.replace(",", "")) * MONEY_FACTORS[unit])


def fact(document_id, page, metric, role, value, currency, literal, *, bound="point", period=None, qualifier=None):
    return {"document_id": document_id, "page": page, "metric": metric, "role": role, "bound": bound,
            "value_base": str(value), "currency": currency, "period": period, "qualifier": qualifier,
            "literal": literal, "binding_approved": True}


def add_money_range(result, document_id, page, metric, role, match, period=None):
    left, unit_left, right, unit_right = match.group("left", "unit_left", "right", "unit_right")
    values = sorted((Decimal(amount(left, unit_left)), Decimal(amount(right, unit_right))))
    literal = match.group(0)
    result += [fact(document_id, page, metric, role, values[0], "CNY", literal, bound="lower", period=period),
               fact(document_id, page, metric, role, values[1], "CNY", literal, bound="upper", period=period)]


def add_ratio_range(result, document_id, page, metric, role, match, period=None, negative=False):
    values = [Decimal(match.group("left")) / 100, Decimal(match.group("right")) / 100]
    if negative:
        values = [-value for value in values]
    values.sort()
    result += [fact(document_id, page, metric, role, values[0], "ratio", match.group(0), bound="lower", period=period),
               fact(document_id, page, metric, role, values[1], "ratio", match.group(0), bound="upper", period=period)]


def earnings_facts(document_id, page, text):
    result = []
    period_match = re.search(r"业绩预告期间[:：]?(?P<start>\d{4}年\d{1,2}月\d{1,2}日)[至-](?P<end>\d{4}年\d{1,2}月\d{1,2}日)", text)
    period = {"start": period_match.group("start"), "end": period_match.group("end")} if period_match else None
    previous = re.search(r"前次业绩预告情况[:：]?(?P<body>.*?)(?:修正后的预计业绩|二、)", text)
    if previous:
        body = previous.group("body")
        money = re.search(r"变动区间为(?P<left>[+-]?[\d,.]+)(?P<unit_left>亿元|万元|元)至(?P<right>[+-]?[\d,.]+)(?P<unit_right>亿元|万元|元)", body)
        ratio = re.search(r"(?:下降|变动幅度为-)(?P<left>[\d.]+)%至-?(?P<right>[\d.]+)%", body)
        if money:
            add_money_range(result, document_id, page, "net_profit_parent", "previous_guidance", money, period)
        if ratio:
            add_ratio_range(result, document_id, page, "net_profit_parent_yoy", "previous_guidance", ratio, period, negative=True)
    corrected = re.search(r"修正后的预计业绩[:：]?(?P<body>.*?)(?:注[:：]|二、)", text)
    if corrected:
        body = corrected.group("body")
        ratio = re.search(r"比上年同期下降[:：]?(?P<left>[\d.]+)%至(?P<right>[\d.]+)%", body)
        money = re.search(r"盈利[:：]?(?P<left>[+-]?[\d,.]+)(?P<unit_left>亿元|万元|元)至(?P<right>[+-]?[\d,.]+)(?P<unit_right>亿元|万元|元)", body)
        prior = re.findall(r"盈利[:：]?(?P<value>[+-]?[\d,.]+)(?P<unit>亿元|万元|元)", body)
        if ratio:
            add_ratio_range(result, document_id, page, "net_profit_parent_yoy", "current_guidance", ratio, period, negative=True)
        if money:
            add_money_range(result, document_id, page, "net_profit_parent", "current_guidance", money, period)
        if len(prior) >= 2:
            value, unit = prior[-1]
            result.append(fact(document_id, page, "net_profit_parent", "prior_year_comparator", amount(value, unit), "CNY", value + unit, period=period))
    # Ordinary (non-correction) forecasts.
    current_section = text.split("二、上年同期业绩情况", 1)[0]
    patterns = (("net_profit_parent", r"归属于母公司所有者的净利润约为(?P<left>[+-]?[\d,.]+)(?P<unit_left>亿元|万元|元)至(?P<right>[+-]?[\d,.]+)(?P<unit_right>亿元|万元|元)"),
                ("net_profit_parent_ex_items", r"扣除非经常性损益的净利润约为(?P<left>[+-]?[\d,.]+)(?P<unit_left>亿元|万元|元)至(?P<right>[+-]?[\d,.]+)(?P<unit_right>亿元|万元|元)"))
    for metric, pattern in patterns:
        match = re.search(pattern, current_section)
        if match:
            add_money_range(result, document_id, page, metric, "current_guidance", match, period)
    if "二、上年同期业绩情况" in text:
        prior = text.split("二、上年同期业绩情况", 1)[1].split("三、", 1)[0]
        singles = (("profit_total", r"利润总额[:：](?P<value>[+-]?[\d,.]+)(?P<unit>亿元|万元|元)"),
                   ("net_profit_parent", r"归属于母公司所有者的净利润[:：](?P<value>[+-]?[\d,.]+)(?P<unit>亿元|万元|元)"),
                   ("net_profit_parent_ex_items", r"扣除非经常性损益的净利润[:：](?P<value>[+-]?[\d,.]+)(?P<unit>亿元|万元|元)"),
                   ("basic_eps", r"基本每股收益[:：](?P<value>[+-]?[\d,.]+)元/股"))
        for metric, pattern in singles:
            match = re.search(pattern, prior)
            if match:
                value = amount(match.group("value"), match.groupdict().get("unit", "元"))
                result.append(fact(document_id, page, metric, "prior_year_comparator", value, "CNY_per_share" if metric == "basic_eps" else "CNY", match.group(0), period=period))
    return result


def contract_facts(document_id, page, text):
    result = []
    foreign = re.search(r"中标金额约为(?P<value>[\d,.]+)亿(?P<currency>谢克尔|美元|欧元).*?约折合(?P<cny>[\d,.]+)亿元人民币", text)
    if foreign:
        codes = {"谢克尔": "ILS", "美元": "USD", "欧元": "EUR"}
        result += [fact(document_id, page, "contract_amount", "contract_original_currency", Decimal(foreign.group("value").replace(",", "")) * Decimal("100000000"), codes[foreign.group("currency")], foreign.group(0)),
                   fact(document_id, page, "contract_amount", "converted_rmb", Decimal(foreign.group("cny").replace(",", "")) * Decimal("100000000"), "CNY", foreign.group(0), qualifier="issuer_approximation")]
        share = re.search(r"约占.*?营业收入的(?P<value>[\d.]+)%", text)
        if share:
            result.append(fact(document_id, page, "contract_to_revenue", "reference_ratio", Decimal(share.group("value")) / 100, "ratio", share.group(0), qualifier="reference_revenue_year_2016"))
    tender = re.search(r"中标价格[:：]设计费(?P<design>[\d,.]+)万元[;；]施工费暂定金额(?P<construction>[\d,.]+)万元", text)
    if tender:
        result += [fact(document_id, page, "design_tender_amount", "current_tender_result", Decimal(tender.group("design").replace(",", "")) * 10000, "CNY", tender.group(0), qualifier="not_signed"),
                   fact(document_id, page, "construction_tender_amount", "current_tender_result", Decimal(tender.group("construction").replace(",", "")) * 10000, "CNY", tender.group(0), qualifier="provisional_not_signed")]
    return result


def extract_document(document_id, parsed, category):
    facts = []
    for page in parsed["pages"]:
        text = compact(page["text"])
        if category == "earnings_forecast":
            facts.extend(earnings_facts(document_id, page["page"], text))
        elif category == "contract":
            facts.extend(contract_facts(document_id, page["page"], text))
    unique = {}
    for item in facts:
        key = tuple(item.get(name) for name in ("metric", "role", "bound", "value_base", "currency"))
        unique[key] = item
    return sorted(unique.values(), key=lambda item: (item["page"], item["metric"], item["role"], item["bound"]))


def fact_key(item):
    return tuple(str(item.get(name)) for name in ("document_id", "metric", "role", "bound", "value_base", "currency", "qualifier"))


def evaluate_gold(predicted, gold):
    predicted_keys, gold_keys = {fact_key(item) for item in predicted}, {fact_key(item) for item in gold}
    true = predicted_keys & gold_keys
    return {"gold_facts": len(gold_keys), "predicted_facts": len(predicted_keys), "matched_facts": len(true),
            "precision": len(true) / len(predicted_keys) if predicted_keys else 0.0,
            "recall": len(true) / len(gold_keys) if gold_keys else 0.0,
            "missing": [list(value) for value in sorted(gold_keys - predicted_keys)],
            "unexpected": [list(value) for value in sorted(predicted_keys - gold_keys)]}
