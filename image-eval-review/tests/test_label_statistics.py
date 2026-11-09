"""E03 analytical checks use hand-counted synthetic observations, never private labels."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import label_statistics as stats


def fixture():
    catalog = {
        "subject": {"name": "合成主体", "dimension": "subject_features", "required": True},
        "aux": {"name": "合成辅助", "dimension": "subject_features", "required": False},
        "structure": {"name": "合成结构", "dimension": "structural_integrity", "required": True},
        "prompt": {"name": "合成提示词", "dimension": "prompt_adherence", "required": True},
    }
    context = {"rule_version": "synthetic_v1", "manifest": {
        "s1": {"split": "development"}, "s2": {"split": "development"}, "s3": {"split": "holdout"}},
        "expected": {sid: {cid: {k: v for k, v in rule.items() if k != "name"}
                            for cid, rule in catalog.items() if cid != "prompt" or sid == "s1"}
                     for sid in ("s1", "s2", "s3")}}
    statuses = {"s1": {"subject": "pass", "aux": "pass", "structure": "pass", "prompt": "pass"},
                "s2": {"subject": "fail", "aux": "unjudgeable", "structure": "pass"},
                "s3": {"subject": "unjudgeable", "aux": "fail", "structure": "pass"}}
    rows = [{"sample_id": sid, "split": context["manifest"][sid]["split"],
             "rule_version": "synthetic_v1", **context["expected"][sid][cid],
             "record": {"label": {"criterion_id": cid, "status": value},
                        "origin": {"kind": "author_status_codex_evidence",
                                   "decision_source": "explicit_user_chat_instruction"}
                        if (sid, cid) == ("s1", "subject") else {"kind": "author_page_edit"}}}
            for sid, values in statuses.items() for cid, value in values.items()]
    return context, rows, catalog


class LabelStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.context, self.rows, self.catalog = fixture()
        self.temp = tempfile.TemporaryDirectory()
        self.private = Path(self.temp.name) / "private"

    def tearDown(self):
        self.temp.cleanup()

    def compute(self):
        return stats.summarize(self.rows, self.context, self.catalog)

    def generate(self, output=None):
        with patch.object(stats, "PRIVATE", self.private), \
                patch.object(stats, "read_source", return_value=(self.context, self.rows, {"synthetic": "sha"})), \
                patch.object(stats, "criterion_catalog", return_value=self.catalog):
            return stats.generate_statistics(Path(self.temp.name) / "synthetic_e02", output)

    def verify(self, folder):
        with patch.object(stats, "read_source", return_value=(self.context, self.rows, {"synthetic": "sha"})):
            return stats.verify_statistics(folder)

    def test_known_totals_dimensions_requiredness_and_origins(self):
        result = self.compute()
        all_row = next(r for r in result["overview"] if r["scope"] == "all")
        self.assertEqual([all_row[k] for k in ("sample_count", "assigned_items", "required_items",
                                              "auxiliary_items", "pass", "fail", "unjudgeable")],
                         [3, 10, 7, 3, 6, 2, 2])
        self.assertEqual(len(result["dimensions"]), 9)
        self.assertEqual(len(result["criteria"]), 12)
        self.assertEqual(result["sources"][-1]["origin_counts"],
                         {"author_status_codex_evidence": 1, "author_page_edit": 9})
        self.assertEqual(result["sources"][-1]["explicit_chat_assisted_items"], 1)

    def test_unassigned_prompt_and_all_unjudgeable_have_distinct_meanings(self):
        rows = self.compute()["criteria"]
        prompt = next(r for r in rows if r["scope"] == "holdout" and r["criterion_id"] == "prompt")
        self.assertEqual([prompt[k] for k in ("assigned_items", "not_assigned_images", "not_applicable")], [0, 1, 0])
        self.assertIn("未发放", prompt["metric_limits"])
        subject = next(r for r in rows if r["scope"] == "holdout" and r["criterion_id"] == "subject")
        self.assertEqual(subject["decidable_items"], 0)
        self.assertIn("决策覆盖率不可计算", subject["metric_limits"])

    def test_no_positive_or_negative_explicitly_limits_metrics(self):
        rows = self.compute()["criteria"]
        structure = next(r for r in rows if r["scope"] == "all" and r["criterion_id"] == "structure")
        self.assertEqual(structure["human_fail_denominator"], 0)
        self.assertTrue(structure["no_fail"])
        self.assertIn("错误通过比例不可计算", structure["metric_limits"])
        subject = next(r for r in rows if r["scope"] == "development" and r["criterion_id"] == "subject")
        self.assertFalse(subject["no_pass"])
        self.assertFalse(subject["no_fail"])
        self.assertIn("待模型结果", subject["metric_limits"])

    def test_missing_duplicate_and_illegal_status_rejected(self):
        for rows in (self.rows[:-1], self.rows + [self.rows[0]]):
            with self.assertRaises(ValueError):
                stats.summarize(rows, self.context, self.catalog)
        for value in ("not_applicable", "", "PASS"):
            rows = copy.deepcopy(self.rows)
            rows[0]["record"]["label"]["status"] = value
            with self.subTest(status=value), self.assertRaises(ValueError):
                stats.summarize(rows, self.context, self.catalog)

    def test_split_dimension_and_requiredness_mismatch_rejected(self):
        for key, value in (("split", "holdout"), ("dimension", "prompt_adherence"), ("required", False)):
            rows = copy.deepcopy(self.rows)
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                stats.summarize(rows, self.context, self.catalog)

    def test_input_order_does_not_change_totals_and_input_unchanged(self):
        before = copy.deepcopy((self.rows, self.context))
        self.assertEqual(self.compute(), stats.summarize(list(reversed(self.rows)), self.context, self.catalog))
        self.assertEqual((self.rows, self.context), before)

    def test_private_nonoverwriting_reports_and_csv_readback(self):
        first = self.generate()
        second = self.generate()
        self.assertNotEqual(first["output"], second["output"])
        self.assertEqual(self.verify(first["output"])["status"], "verified_statistics")
        with self.assertRaises(FileExistsError):
            self.generate(first["output"])
        with self.assertRaisesRegex(ValueError, "private"):
            self.generate(Path(self.temp.name) / "public")
        raw = (Path(first["output"]) / "criteria.csv").read_bytes()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        self.assertIn(b"\r\n", raw)

    def test_modified_csv_or_rehashed_statistics_rejected(self):
        folder = Path(self.generate()["output"])
        path = folder / "criteria.csv"
        original = path.read_bytes()
        path.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "哈希不符"):
            self.verify(folder)
        path.write_bytes(original)
        target = folder / "statistics.json"
        report = json.loads(target.read_text(encoding="utf-8"))
        report["data"]["overview"][0]["pass"] += 1
        target.write_text(json.dumps(report), encoding="utf-8")
        hashes_path = folder / "checksums.json"
        hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
        hashes["statistics.json"] = stats.sha(target.read_bytes())
        hashes_path.write_text(json.dumps(hashes), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "初始快照不一致"):
            self.verify(folder)


if __name__ == "__main__":
    unittest.main()
