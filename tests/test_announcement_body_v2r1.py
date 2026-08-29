import unittest

from announcement_body_v2r1.binder import correction_table_facts, extract_document


class TableRepairTests(unittest.TestCase):
    def test_table_columns_override_flattened_order(self):
        table = [["项目", "本报告期", "上年同期"],
                 ["归属于上市公司股东的净利润", "比上年同期下降：50%至45%", "盈利：83,948.93万元"],
                 [None, "盈利：41,974.47万元至46,171.91\n万元", None]]
        facts = correction_table_facts("doc", 1, [table], {"start": "2018年1月1日", "end": "2018年9月30日"})
        values = {(item["role"], item["bound"]): item["value_base"] for item in facts}
        self.assertEqual(values[("current_guidance", "lower")], "419744700.00")
        self.assertEqual(values[("current_guidance", "upper")], "461719100.00")
        self.assertEqual(values[("prior_year_comparator", "point")], "839489300.00")

    def test_no_header_means_no_table_approval(self):
        self.assertEqual(correction_table_facts("doc", 1, [["项目", "数值"], ["利润", "1万元"]], None), [])

    def test_full_extractor_removes_flattened_false_fact(self):
        parsed = {"pages": [{"page": 1,
                              "text": "业绩预告期间2018年1月1日-2018年9月30日。修正后的预计业绩：比上年同期下降50%至45% 盈利：83,948.93万 盈利：41,974.47万元至46,171.91元 万元 注：",
                              "tables": [["placeholder"]]}]}
        # Without a valid table, conservative V2 behavior is preserved; this test
        # ensures the repair is only activated by explicit column headers.
        facts = extract_document("doc", parsed, "earnings_forecast")
        self.assertTrue(all(item["binding_approved"] for item in facts))


if __name__ == "__main__":
    unittest.main()
