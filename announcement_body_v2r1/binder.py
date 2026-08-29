import re

from announcement_body_v2.binder import amount, compact, earnings_facts, contract_facts


def correction_table_facts(document_id, page_number, tables, period):
    from announcement_body_v2.binder import add_money_range, fact
    result = []
    for table in tables:
        if not table or len(table) < 2:
            continue
        headers = [compact(cell or "") for cell in table[0]]
        if "本报告期" not in headers or "上年同期" not in headers:
            continue
        current_index, prior_index = headers.index("本报告期"), headers.index("上年同期")
        current = compact("".join((row[current_index] or "") for row in table[1:] if len(row) > current_index))
        prior = compact("".join((row[prior_index] or "") for row in table[1:] if len(row) > prior_index))
        current_match = re.search(r"盈利[:：]?(?P<left>[+-]?[\d,.]+)(?P<unit_left>亿元|万元|元)至(?P<right>[+-]?[\d,.]+)(?P<unit_right>亿元|万元|元)", current)
        prior_match = re.search(r"盈利[:：]?(?P<value>[+-]?[\d,.]+)(?P<unit>亿元|万元|元)", prior)
        if current_match:
            add_money_range(result, document_id, page_number, "net_profit_parent", "current_guidance", current_match, period)
        if prior_match:
            result.append(fact(document_id, page_number, "net_profit_parent", "prior_year_comparator",
                               amount(prior_match.group("value"), prior_match.group("unit")), "CNY", prior_match.group(0), period=period))
    return result


def extract_document(document_id, parsed, category):
    facts = []
    for page in parsed["pages"]:
        text = compact(page["text"])
        if category == "earnings_forecast":
            page_facts = earnings_facts(document_id, page["page"], text)
            period = next((item.get("period") for item in page_facts if item.get("period")), None)
            table_facts = correction_table_facts(document_id, page["page"], page.get("tables", []), period)
            if table_facts:
                page_facts = [item for item in page_facts if not (item["metric"] == "net_profit_parent" and item["role"] in {"current_guidance", "prior_year_comparator"})]
                page_facts.extend(table_facts)
            facts.extend(page_facts)
        elif category == "contract":
            facts.extend(contract_facts(document_id, page["page"], text))
    unique = {}
    for item in facts:
        key = tuple(item.get(name) for name in ("metric", "role", "bound", "value_base", "currency"))
        unique[key] = item
    return sorted(unique.values(), key=lambda item: (item["page"], item["metric"], item["role"], item["bound"]))
