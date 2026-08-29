from announcement_body_v3r1.binder import extract_document


def test_continuation_row_and_ex_items_are_disambiguated():
    parsed = {"pages": [{"page": 1, "tables": [[
        ["项目", None, None, "本报告期", "上年同期", "变动"],
        ["", "归属于上市公司股", "", "100", "90", "11.11"],
        [None, "东的净利润", None, None, None, None],
        ["", "归属于上市公司股", "", "80", "70", "14.29"],
        [None, "东的扣除非经常性损益的净利润", None, None, None, None],
    ]]}]}
    facts = extract_document("600674_1211243407", parsed, "earnings_flash")
    assert len(facts) == 3
    assert {item["metric"] for item in facts} == {"net_profit_parent", "net_profit_parent_change"}


def test_contract_audited_wording_is_supported():
    parsed = {"pages": [{"page": 1, "text": "中标金额：人民币17,860万元"},
                         {"page": 2, "text": "占公司2020年度经审计营业收入比重为8.62%。"}]}
    facts = extract_document("600481_1211571975", parsed, "contract")
    assert {(item["metric"], item["value_base"]) for item in facts} == {("contract_amount", "178600000"), ("contract_to_revenue", "0.0862")}


def test_duration_wording_and_decimal_normalization():
    parsed = {"pages": [{"page": 1, "text": "回购价格为不超过人民币8元/股，回购数量不超过24,608万股，实施期限为自董事会审议通过回购股份方案之日起12个月内。累计回购了162,847,162股，占公司总股本的1.985%，最高成交价为7.04元/股，最低成交价为5.84元/股，支付的总金额为1,033,099,334.60元。"}]}
    facts = extract_document("000069_1209041295", parsed, "repurchase")
    assert any(item["metric"] == "repurchase_duration_months" and item["value_base"] == "12" for item in facts)
