"""Hand-calculated metric fixtures, invalid joins and export integrity; no API/private labels."""
import copy
import csv
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import holdout_metrics as m
from annotation_store import now
from run_store import RunError


def row(human, ai, *, sid="synthetic_1", cid="subject", dimension="subject_features", required=True):
    return {"prompt_version": "judge_v1", "sample_id": sid, "criterion_id": cid,
            "dimension": dimension, "required": required, "human_status": human, "ai_status": ai}


class MetricTests(unittest.TestCase):
    def fixture(self):
        catalog = {"subject": {"dimension": "subject_features", "required": True, "criterion_name": "必需主体"},
                   "aux": {"dimension": "subject_features", "required": False, "criterion_name": "辅助"},
                   "structure": {"dimension": "structural_integrity", "required": True, "criterion_name": "结构"},
                   "prompt": {"dimension": "prompt_adherence", "required": True, "criterion_name": "提示词"}}
        expected = {sid: {cid: {"dimension": r["dimension"], "required": r["required"]}
                         for cid, r in catalog.items()} for sid in ("synthetic_1", "synthetic_2")}
        snapshots = {sid: "synthetic-"+sid for sid in expected}
        human = [{"sample_id": sid, "split": "holdout", "rule_version": "rules_v1",
                  "snapshot_sha256": snapshots[sid], **rule,
                  "record": {"label": {"criterion_id": cid, "status": "fail" if cid == "subject" else "pass"}}}
                 for sid, rules in expected.items() for cid, rule in rules.items()]
        batches = {v: v+"-batch" for v in m.VERSIONS}
        results = [{"prompt_version": v, "batch_id": batches[v], "sample_id": sid,
                    "engineering_status": "succeeded", "result_attempt_id": v+sid, "stop_code": None,
                    "labels": [{"criterion_id": cid, "status": "pass"} for cid in rules]}
                   for v in m.VERSIONS for sid, rules in expected.items()]
        return human, results, expected, batches, snapshots, catalog

    def joined(self):
        h, a, e, b, s, c = self.fixture()
        return m.join_items(h, a, e, b, s), c

    def test_hand_calculated_matrix_preserves_abstention_and_failure_denominators(self):
        rows = [row(h, a, sid=str(i)) for i, (h, a) in enumerate(
            (h, a) for h in ("pass", "fail", "unjudgeable") for a in ("pass", "fail", "unjudgeable", None))]
        got = m.counts(rows)
        for key, n, d in (("decision_coverage", 4, 8), ("conditional_agreement", 2, 4),
                          ("false_pass", 1, 4), ("false_rejection", 1, 4)):
            self.assertEqual(got[key], m.ratio(n, d))
        self.assertEqual(got["human_unjudgeable"], 4)
        self.assertEqual(got["ai_engineering_missing"], 3)
        self.assertEqual(got["human_binary_ai_unjudgeable"], 2)
        self.assertEqual(got["human_binary_engineering_missing"], 2)

    def test_zero_denominators_are_null_not_zero_percent(self):
        got = m.counts([row("unjudgeable", "pass")])
        for key in m.METRICS:
            self.assertIsNone(got[key]["value"])
            self.assertEqual(got[key]["status"], "zero_denominator")
            self.assertIn("不可计算", m.display(got[key]))
        self.assertEqual(m.counts([row("pass", "pass")])["false_pass"], m.ratio(0, 0))
        self.assertEqual(m.ratio(0, 3)["value"], 0)
        self.assertEqual(m.ratio(3, 3)["value"], 1)

    def test_all_engineering_missing_has_zero_coverage_undefined_agreement(self):
        got = m.counts([row("pass", None), row("fail", None)])
        self.assertEqual(got["decision_coverage"], m.ratio(0, 2))
        self.assertEqual(got["conditional_agreement"], m.ratio(0, 0))
        self.assertEqual(got["false_pass"], m.ratio(0, 1))
        self.assertEqual(got["false_rejection"], m.ratio(0, 1))

    def test_invalid_status_or_ratio_rejected(self):
        for value in ("not_applicable", "", "missing", "FAIL"):
            with self.assertRaises(ValueError): m.counts([row("pass", value)])
        for n, d in ((-1, 2), (3, 2), (1, 0), (True, 1)):
            with self.assertRaises(ValueError): m.ratio(n, d)

    def test_joins_by_identity_not_input_order_and_retains_auxiliary(self):
        h, a, e, b, s, _ = self.fixture()
        expected = m.join_items(h, a, e, b, s)
        h.reverse(); a.reverse()
        for task in a: task["labels"].reverse()
        self.assertEqual(expected, m.join_items(h, a, e, b, s))
        self.assertEqual(len(expected), 16)
        self.assertEqual(sum(not r["required"] for r in expected), 4)

    def test_duplicate_unknown_missing_or_cross_split_result_rejected(self):
        h, a, e, b, s, _ = self.fixture()
        variants = [a+a[:1], a[:-1]]
        bad = copy.deepcopy(a);bad[0]["sample_id"] = "development_sample";variants.append(bad)
        bad = copy.deepcopy(a);bad[0]["prompt_version"] = "judge_v3";variants.append(bad)
        bad = copy.deepcopy(a);bad[0]["batch_id"] = b["judge_v2"];variants.append(bad)
        for results in variants:
            with self.assertRaises(ValueError): m.join_items(h, results, e, b, s)

    def test_missing_duplicate_or_illegal_ai_labels_rejected(self):
        h, a, e, b, s, _ = self.fixture()
        for change in (lambda labels: labels.pop(), lambda labels: labels.append(copy.deepcopy(labels[0])),
                       lambda labels: labels[0].update(status="not_applicable"),
                       lambda labels: labels[0].update(criterion_id="unknown")):
            bad = copy.deepcopy(a);change(bad[0]["labels"])
            with self.assertRaises(ValueError): m.join_items(h, bad, e, b, s)

    def test_failed_task_expands_as_null_never_fail(self):
        h, a, e, b, s, _ = self.fixture()
        a[0].update(engineering_status="failed", result_attempt_id=None, labels=None, stop_code="http_524")
        joined = m.join_items(h, a, e, b, s)
        failed = [r for r in joined if r["engineering_status"] == "failed"]
        self.assertEqual(len(failed), 4)
        self.assertTrue(all(r["ai_status"] is None for r in failed))
        a[0]["labels"] = []
        with self.assertRaises(ValueError): m.join_items(h, a, e, b, s)

    def test_human_reference_duplicates_missing_dimensions_and_snapshot_rejected(self):
        h, a, e, b, s, _ = self.fixture()
        variants = [h+h[:1], h[:-1]]
        for field, value in (("split", "development"), ("dimension", "prompt_adherence"),
                             ("required", False), ("snapshot_sha256", "changed")):
            bad = copy.deepcopy(h);bad[0][field] = value;variants.append(bad)
        bad = copy.deepcopy(h);bad[0]["record"]["label"]["status"] = "not_applicable";variants.append(bad)
        for rows in variants:
            with self.assertRaises(ValueError): m.join_items(rows, a, e, b, s)

    def test_summaries_reconcile_and_percentages_are_weighted(self):
        items, catalog = self.joined()
        got = m.summarize(items, catalog)
        self.assertEqual(len(got["dimensions"]), 6)
        self.assertEqual(got["overview"][0]["conditional_agreement"], m.ratio(6, 8))
        self.assertEqual(got["dimensions"][0]["conditional_agreement"], m.ratio(2, 4))
        self.assertNotAlmostEqual(got["overview"][0]["conditional_agreement"]["value"],
                                 sum(r["conditional_agreement"]["value"] for r in got["dimensions"][:3])/3)
        self.assertEqual(got["requiredness"][0]["assigned_items"], 6)
        self.assertEqual(got["requiredness"][1]["assigned_items"], 2)

    def test_paired_subset_uses_identical_human_binary_items(self):
        items, catalog = self.joined()
        for r in items:
            if r["sample_id"] == "synthetic_1" and r["criterion_id"] == "subject" and r["prompt_version"] == "judge_v2":
                r["ai_status"] = "fail"
            if r["sample_id"] == "synthetic_2" and r["criterion_id"] == "prompt" and r["prompt_version"] == "judge_v1":
                r["ai_status"] = None
        paired = m.summarize(items, catalog)["paired"][0]
        self.assertEqual(paired["common_decidable_items"], 7)
        self.assertEqual(paired["v1_only_agrees"], 0)
        self.assertEqual(paired["v2_only_agrees"], 1)
        self.assertEqual(paired["v1_agreement"], m.ratio(5, 7))
        self.assertEqual(paired["v2_agreement"], m.ratio(6, 7))

    def test_pair_reference_mismatch_or_duplicate_rejected(self):
        items, catalog = self.joined()
        bad = copy.deepcopy(items);bad[-1]["human_status"] = "fail"
        with self.assertRaises(ValueError): m.summarize(bad, catalog)
        with self.assertRaises(ValueError): m.summarize(items+items[:1], catalog)

    def test_csv_json_markdown_export_roundtrip_and_tamper_detection(self):
        items, catalog = self.joined()
        report = {"task": "G02", "created_at": now(), "generator_sha256": m.text_digest(Path(m.__file__)),
                  "catalog": catalog, "data": m.summarize(items, catalog)}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            m.client.write_json(folder/"metrics.json", report)
            (folder/"items.jsonl").write_text("\n".join(json.dumps(r) for r in items)+"\n", encoding="utf-8")
            for name in ("dimensions", "overview", "criteria", "requiredness", "paired"):
                (folder/(name+".csv")).write_bytes(("\ufeff"+m.csv_text(report["data"][name])).encode())
            (folder/"保留集主要指标.md").write_text(m.markdown(report), encoding="utf-8")
            m.client.write_json(folder/"checksums.json", {name: m.client.digest((folder/name).read_bytes()) for name in m.FILES})
            self.assertEqual(m.verify_export(folder, refresh_source=False)["items"], 16)
            parsed = list(csv.DictReader(io.StringIO((folder/"dimensions.csv").read_text(encoding="utf-8-sig"))))
            self.assertEqual(parsed[-1]["false_pass_value"], "")
            self.assertEqual(parsed[-1]["false_pass_status"], "zero_denominator")
            with (folder/"dimensions.csv").open("a", encoding="utf-8") as stream: stream.write("corrupt")
            with self.assertRaises(ValueError): m.verify_export(folder, refresh_source=False)

    def test_readonly_database_handle_refuses_writes_and_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"runs.sqlite3"
            db = sqlite3.connect(path)
            db.execute("CREATE TABLE demo(id INTEGER)")
            db.commit()
            db.close()
            before = path.read_bytes()
            store = m.MetricRuns(path)
            with store.connect() as db:
                with self.assertRaises(sqlite3.OperationalError): db.execute("INSERT INTO demo VALUES(1)")
            with self.assertRaises(RunError):
                with store.connect(immediate=True): pass
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
