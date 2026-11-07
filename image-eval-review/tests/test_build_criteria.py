import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_manifest import DATA, read_jsonl
from build_criteria import build_rows


class CriteriaTests(unittest.TestCase):
    def setUp(self):
        self.manifest = read_jsonl(DATA / "manifest.jsonl")
        self.catalog = json.loads((DATA / "prompt_requirements.json").read_text(encoding="utf-8"))
        self.spec = json.loads((DATA / "evaluation_spec.draft.json").read_text(encoding="utf-8"))

    def build(self):
        return build_rows(self.manifest, self.catalog, self.spec)

    def test_only_wooden_table_excluded_original_fields_unchanged(self):
        before = copy.deepcopy((self.manifest, self.catalog, self.spec))
        result = self.build()
        self.assertEqual(len(result), 40)
        originals = {row["sample_id"]: row for row in self.manifest}
        for row in result:
            expected = [item["id"] for item in originals[row["sample_id"]]["requirements"]
                        if item["id"] != "wooden_table"]
            self.assertEqual([item["id"] for item in row["prompt_criteria"]], expected)
            self.assertEqual(len(row["structure_criterion_ids"]), 4)
            self.assertNotIn("wooden_table", expected)
        self.assertEqual(sum(len(row["prompt_criteria"]) for row in result), 240)
        self.assertEqual(sum(len(row["excluded_prompt_requirement_ids"]) for row in result), 32)
        self.assertEqual((self.manifest, self.catalog, self.spec), before)

    def test_smoke_counts_and_untouched_trigger_requirements(self):
        rows = {row["sample_id"]: row for row in self.build()}
        self.assertEqual([len(rows[key]["prompt_criteria"]) for key in
                          ("sample_020", "sample_029", "sample_042")], [2, 11, 13])
        trigger = [sample for sample in self.manifest if sample["prompt"] == "a photo of zqv figurine"]
        self.assertEqual(len(trigger), 8)
        for sample in trigger:
            active = rows[sample["sample_id"]]
            self.assertEqual(active["excluded_prompt_requirement_ids"], [])
            self.assertEqual([row["id"] for row in active["prompt_criteria"]], ["photo_style", "figurine"])

    def test_missing_unknown_duplicate_or_incomplete_boundaries_rejected(self):
        for failure in ("missing", "unknown_exclusion", "duplicate", "incomplete"):
            original = copy.deepcopy(self.spec)
            if failure == "missing":
                self.spec["prompt_criteria"].pop()
            elif failure == "unknown_exclusion":
                self.spec["excluded_prompt_requirements"].append({"id": "unknown"})
            elif failure == "duplicate":
                self.spec["structure_criteria"].append(self.spec["structure_criteria"][0])
            else:
                self.spec["prompt_criteria"][0]["fail_when"] = ""
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                self.build()
            self.spec = original

    def test_changed_original_prompt_or_requirement_rejected(self):
        for failure in ("prompt", "requirement"):
            original = copy.deepcopy(self.manifest)
            if failure == "prompt":
                self.manifest[0]["prompt"] = "unknown prompt"
            else:
                self.manifest[0]["requirements"].pop()
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                self.build()
            self.manifest = original

    def test_input_order_and_return_value_do_not_mutate_sources(self):
        result = self.build()
        self.manifest.reverse()
        self.assertEqual(self.build(), result)
        result[0]["prompt_criteria"][0]["description"] = "changed locally"
        self.assertNotEqual(result, self.build())


if __name__ == "__main__":
    unittest.main()
