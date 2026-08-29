import json
from pathlib import Path

from announcement_body_v2.binder import fact_key
from announcement_body_v3.binder import extract_document
from announcement_body_v3.gold import expanded_gold


ROOT = Path(__file__).resolve().parents[1]


def test_gold_is_unique_and_covers_all_text_documents():
    gold = expanded_gold()
    assert len(gold["documents"]) == 11
    assert len(set(gold["documents"])) == 11
    assert len({fact_key(item) for item in gold["facts"]}) == len(gold["facts"])
    assert {item["document_id"] for item in gold["facts"]} == set(gold["documents"])


def test_flash_synthetic_table_uses_cells_and_units():
    parsed = {"pages": [{"page": 1, "tables": [[
        ["项目", "本报告期", "上年同期", "增减"],
        ["营业总收入", None, None, "308,444.73", None, None, "311,084.15", None, None, "-0.85"],
    ]]}]}
    facts = extract_document("000728_1220627969", parsed, "earnings_flash")
    values = {(item["role"], item["value_base"]) for item in facts}
    assert values == {("current_preliminary", "3084447300.00"), ("prior_year_comparator", "3110841500.00"), ("reported_change", "-0.0085")}


def test_contract_excludes_background_project_numbers():
    parsed = {"pages": [{"page": 1, "text": "中标金额：人民币17,860万元（以最终合同签订金额为准）"},
                         {"page": 2, "text": "项目年产值约210亿元，新增就业约3000人。占公司2020年度审计营业收入比重为8.62%。"}]}
    facts = extract_document("600481_1211571975", parsed, "contract")
    values = {item["value_base"] for item in facts}
    assert values == {"178600000", "0.0862"}
    assert "21000000000" not in values and "3000" not in values


def test_scan_stays_outside_gold():
    selection = json.loads((ROOT / "artifacts/announcement_body_v1/selection.json").read_text(encoding="utf-8"))
    scans = []
    for record in selection["records"]:
        document_id = record["symbol"] + "_" + record["announcement_id"]
        parsed = json.loads((ROOT / "data/announcement_body_v1" / document_id / "parsed.json").read_text(encoding="utf-8"))
        if not parsed["body_extraction_passed"]:
            scans.append(document_id)
    assert scans == ["600009_1204429465"]
    assert scans[0] not in expanded_gold()["documents"]

