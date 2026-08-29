import unittest

from announcement_body_v2.binder import compact, contract_facts, earnings_facts, evaluate_gold, extract_document


class BindingTests(unittest.TestCase):
    def test_correction_keeps_previous_current_and_prior_year_separate(self):
        text = compact("业绩预告期间：2018年1月1日-2018年9月30日。前次业绩预告情况：预计下降95%至65%，变动区间为4,197.45万元至29,382.13万元。修正后的预计业绩：比上年同期下降：50%至45% 盈利：41,974.47万元至46,171.91万元 盈利：83,948.93万元 注：结束")
        facts = earnings_facts("doc", 1, text)
        roles = {(item["metric"], item["role"], item["bound"]) for item in facts}
        self.assertIn(("net_profit_parent", "previous_guidance", "lower"), roles)
        self.assertIn(("net_profit_parent", "current_guidance", "upper"), roles)
        self.assertIn(("net_profit_parent", "prior_year_comparator", "point"), roles)

    def test_negative_forecast_range_is_numerically_ordered(self):
        text = compact("业绩预告期间2024年1月1日至2024年6月30日。预计2024年半年度实现归属于母公司所有者的净利润约为-3.5亿元至-2.4亿元。预计2024年半年度实现归属于母公司所有者的扣除非经常性损益的净利润约为-6.4亿元至-4.3亿元。")
        facts = earnings_facts("doc", 1, text)
        net = [item for item in facts if item["metric"] == "net_profit_parent"]
        self.assertEqual([item["value_base"] for item in net], ["-350000000.0", "-240000000.0"])

    def test_foreign_and_converted_currency_are_not_merged(self):
        facts = contract_facts("doc", 1, compact("中标金额约为24.9亿谢克尔，约折合45亿元人民币，约占本公司中国会计准则下2016年营业收入的0.70%。"))
        self.assertEqual([(item["role"], item["currency"]) for item in facts[:2]], [("contract_original_currency", "ILS"), ("converted_rmb", "CNY")])

    def test_provisional_unsigned_tender_is_qualified(self):
        facts = contract_facts("doc", 1, compact("中标价格：设计费631.87万元；施工费暂定金额65,133.61万元"))
        self.assertEqual([item["qualifier"] for item in facts], ["not_signed", "provisional_not_signed"])

    def test_gold_evaluation_rejects_extra_fact(self):
        fact = {"document_id":"d","metric":"m","role":"r","bound":"point","value_base":"1","currency":"CNY","qualifier":None}
        extra = {**fact, "metric":"other"}
        result = evaluate_gold([fact, extra], [fact])
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 1.0)

    def test_unreviewed_categories_never_auto_approve(self):
        parsed = {"pages": [{"page": 1, "text": "回购金额100万元"}]}
        self.assertEqual(extract_document("doc", parsed, "repurchase"), [])


if __name__ == "__main__":
    unittest.main()
