import copy
import itertools
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_results import (
    DIMENSIONS, evaluate_result, expected_for_sample, load_context, validate_policy,
)


def make_result(sample_id, expected):
    # Synthetic evidence strings only; never persisted as human or model labels.
    return {
        "sample_id": sample_id,
        "labels": [
            {"criterion_id": key, "status": "pass", "candidate_region": "synthetic region",
             "candidate_observation": "synthetic observation", "reason": "synthetic reason"}
            for key in expected],
    }


class ResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy, cls.reference, cls.evaluation, cls.samples = load_context()

    def setUp(self):
        self.sample_id = "sample_020"
        self.expected = expected_for_sample(
            self.sample_id, self.reference, self.evaluation, self.samples)
        self.result = make_result(self.sample_id, self.expected)

    def label(self, key):
        return next(row for row in self.result["labels"] if row["criterion_id"] == key)

    def evaluate(self, **kwargs):
        return evaluate_result(
            self.sample_id, self.result, self.expected, self.policy, **kwargs)

    def test_all_40_samples_and_smoke_inventory(self):
        inventories = {
            row["sample_id"]: expected_for_sample(
                row["sample_id"], self.reference, self.evaluation, self.samples)
            for row in self.samples}
        self.assertEqual(len(inventories), 40)
        self.assertEqual(sum(map(len, inventories.values())), 720)
        self.assertEqual(sum(row["required"] for items in inventories.values()
                             for row in items.values()), 640)
        self.assertEqual([len(inventories[key]) for key in
                          ("sample_020", "sample_029", "sample_042")], [14, 23, 25])
        for sample_id, expected in inventories.items():
            with self.subTest(sample_id=sample_id):
                self.assertNotIn("wooden_table", expected)
                result = evaluate_result(
                    sample_id, make_result(sample_id, expected), expected, self.policy)
                self.assertEqual(result["candidate_status"], "candidate_pass")
                self.assertEqual(result["validation_errors"], [])

    def test_required_status_combinations_and_dimension_priority(self):
        subject_id = "subject_red_upper_garment"
        structure_id = next(key for key, row in self.expected.items()
                            if row["dimension"] == "structural_integrity")
        for subject, structure in itertools.product(
                ("pass", "fail", "unjudgeable"), repeat=2):
            with self.subTest(subject=subject, structure=structure):
                self.label(subject_id)["status"] = subject
                self.label(structure_id)["status"] = structure
                output = self.evaluate()
                expected = ("candidate_fail" if "fail" in (subject, structure)
                            else "needs_review" if "unjudgeable" in (subject, structure)
                            else "candidate_pass")
                self.assertEqual(output["candidate_status"], expected)
                mapping = {"pass": "candidate_pass", "fail": "candidate_fail",
                           "unjudgeable": "needs_review"}
                self.assertEqual(output["dimension_statuses"]["subject_features"],
                                 mapping[subject])
                self.assertEqual(output["dimension_statuses"]["structural_integrity"],
                                 mapping[structure])
                self.assertEqual(output["dimension_statuses"]["prompt_adherence"],
                                 "candidate_pass")
                self.assertEqual(output["engineering_status"], "valid_result")

    def test_auxiliary_fail_or_unjudgeable_does_not_veto(self):
        auxiliary = [key for key, row in self.expected.items() if not row["required"]]
        self.assertEqual(len(auxiliary), 2)
        for statuses in itertools.product(("pass", "fail", "unjudgeable"), repeat=2):
            for key, status in zip(auxiliary, statuses):
                self.label(key)["status"] = status
            output = self.evaluate()
            self.assertEqual(output["candidate_status"], "candidate_pass")
            self.assertEqual(set(output["dimension_statuses"].values()), {"candidate_pass"})
            self.assertEqual(output["required_fail_ids"], [])
            self.assertEqual(output["required_unjudgeable_ids"], [])

    def test_prompt_base_required_even_when_subject_base_auxiliary(self):
        expected = expected_for_sample(
            "sample_042", self.reference, self.evaluation, self.samples)
        result = make_result("sample_042", expected)
        rows = {row["criterion_id"]: row for row in result["labels"]}
        rows["subject_gold_coin_base"]["status"] = "unjudgeable"
        rows["round_gold_base"]["status"] = "fail"
        output = evaluate_result("sample_042", result, expected, self.policy)
        self.assertEqual(output["candidate_status"], "candidate_fail")
        self.assertEqual(output["dimension_statuses"]["subject_features"], "candidate_pass")
        self.assertEqual(output["required_fail_ids"], ["round_gold_base"])

    def test_engineering_error_precedes_visual_fail(self):
        self.label("subject_red_upper_garment")["status"] = "fail"
        self.label("subject_under_eye_scar")["status"] = "unjudgeable"
        self.assertEqual(self.evaluate()["candidate_status"], "candidate_fail")
        # Even an auxiliary missing row is incomplete; don't release partial judgments.
        self.result["labels"] = [
            row for row in self.result["labels"]
            if row["criterion_id"] != "subject_back_straw_hat"]
        output = self.evaluate()
        self.assertEqual(output["engineering_status"], "invalid_result")
        self.assertEqual(output["candidate_status"], "needs_review")
        self.assertEqual(set(output["dimension_statuses"].values()), {"needs_review"})
        self.assertIn({"code": "missing_criterion_id", "criterion_id": "subject_back_straw_hat"},
                      output["validation_errors"])

    def test_pending_running_failed_are_not_visual_failures(self):
        for task_status in ("pending", "running", "failed"):
            output = self.evaluate(task_status=task_status)
            self.assertEqual(output["candidate_status"], "needs_review")
            self.assertEqual(output["engineering_status"], "task_not_succeeded")
            self.assertEqual(output["required_fail_ids"], [])

    def test_unknown_duplicate_omitted_and_wrong_sample_rejected(self):
        original = copy.deepcopy(self.result)
        for case in ("unknown", "wooden_table", "duplicate", "missing", "sample"):
            with self.subTest(case=case):
                self.result = copy.deepcopy(original)
                if case in ("unknown", "wooden_table"):
                    extra = dict(self.result["labels"][0], criterion_id=case)
                    self.result["labels"].append(extra)
                elif case == "duplicate":
                    self.result["labels"].append(copy.deepcopy(self.result["labels"][0]))
                elif case == "missing":
                    self.result["labels"].pop()
                else:
                    self.result["sample_id"] = "sample_999"
                output = self.evaluate()
                self.assertEqual(output["engineering_status"], "invalid_result")
                self.assertEqual(output["candidate_status"], "needs_review")

    def test_invalid_envelopes_and_rows_do_not_crash(self):
        cases = [None, [], "not JSON", {}, {"sample_id": self.sample_id, "labels": None},
                 {"sample_id": self.sample_id, "labels": [None, [], {}, 42]},
                 {**self.result, "candidate_status": "candidate_pass"}]
        for result in cases:
            with self.subTest(result=result):
                output = evaluate_result(
                    self.sample_id, result, self.expected, self.policy)
                self.assertEqual(output["engineering_status"], "invalid_result")
                self.assertEqual(output["candidate_status"], "needs_review")

    def test_invalid_status_and_not_applicable_rejected(self):
        for status in ("not_applicable", "PASS", "", None, [], {}, 1):
            with self.subTest(status=status):
                self.result["labels"][0]["status"] = status
                output = self.evaluate()
                self.assertEqual(output["candidate_status"], "needs_review")
                code = ("not_applicable_forbidden" if status == "not_applicable"
                        else "invalid_status")
                self.assertIn(code, [row["code"] for row in output["validation_errors"]])

    def test_evidence_required_and_labels_cannot_override_requiredness(self):
        original = copy.deepcopy(self.result)
        for field, value in itertools.product(self.policy["evidence_fields"],
                                              ("", "  ", None, 42, [])):
            self.result = copy.deepcopy(original)
            self.result["labels"][0][field] = value
            self.assertEqual(self.evaluate()["candidate_status"], "needs_review")
        for extra in ("required", "dimension", "confidence"):
            self.result = copy.deepcopy(original)
            self.result["labels"][0][extra] = False
            self.assertEqual(self.evaluate()["engineering_status"], "invalid_result")
        for field in self.policy["label_fields"]:
            self.result = copy.deepcopy(original)
            del self.result["labels"][0][field]
            self.assertEqual(self.evaluate()["engineering_status"], "invalid_result")

    def test_inputs_unchanged_and_label_order_irrelevant(self):
        before = copy.deepcopy((self.expected, self.policy, self.result))
        output = self.evaluate()
        self.assertEqual((self.expected, self.policy, self.result), before)
        self.result["labels"].reverse()
        self.assertEqual(self.evaluate(), output)

    def test_invalid_rules_raise_instead_of_passing_empty_or_corrupt_criteria(self):
        invalid = [{}, {"criterion": {"dimension": "unknown", "required": True}},
                   {"criterion": {"dimension": DIMENSIONS[0], "required": "true"}}]
        for expected in invalid:
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                evaluate_result(self.sample_id, self.result, expected, self.policy)
        policy = copy.deepcopy(self.policy)
        policy["aggregation_order"].reverse()
        with self.assertRaises(ValueError):
            validate_policy(policy)
        with self.assertRaises(ValueError):
            expected_for_sample("sample_999", self.reference, self.evaluation, self.samples)
        samples = copy.deepcopy(self.samples)
        next(row for row in samples if row["sample_id"] == self.sample_id)[
            "prompt_criteria"][0]["required"] = False
        with self.assertRaises(ValueError):
            expected_for_sample(
                self.sample_id, self.reference, self.evaluation, samples)


if __name__ == "__main__":
    unittest.main()
