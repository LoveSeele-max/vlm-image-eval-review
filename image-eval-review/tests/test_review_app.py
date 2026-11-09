"""F02 pure comparison, read-only evidence gate and HTTP boundary checks with synthetic data."""
import copy
import json
import sqlite3
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from contextlib import closing
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import review_app as review


def label(cid, status="pass", observation="synthetic observation"):
    return {"criterion_id": cid, "status": status, "candidate_region": "synthetic region",
            "candidate_observation": observation, "reason": "synthetic reason"}


def fixture():
    criteria = [{"id": cid, "name": "合成条目" + cid, "dimension": dim, "required": required,
                 "pass_when": "synthetic pass", "fail_when": "synthetic fail",
                 "unjudgeable_when": "synthetic unknown"}
                for cid, dim, required in (("c1", "subject_features", True),
                                          ("c2", "structural_integrity", True),
                                          ("c3", "prompt_adherence", False))]
    humans = {cid: {"label": label(cid, "unjudgeable" if cid == "c3" else "pass"),
                    "origin": {"kind": "author_status_codex_evidence",
                               "decision_source": "explicit_user_chat_instruction",
                               "private_path": "PRIVATE_PATH_MUST_NOT_LEAK"},
                    "private_detail": "NOT_FOR_DISPLAY"} for cid in ("c1", "c2", "c3")}
    ai = {"c1": label("c1", "fail"), "c2": label("c2", "pass"), "c3": label("c3", "unjudgeable")}
    first = review.compare_sample("sample_008", criteria, humans, ai, task_status="succeeded", error_code=None,
                                  attempts=[], cached=True, candidate_status="candidate_fail")
    second = review.compare_sample("sample_016", criteria, humans, None, task_status="failed", error_code="http_524",
                                   attempts=[{"number": 1, "status": "failed", "http_status": 524,
                                              "error_code": "http_524", "elapsed_seconds": 125}], cached=False)
    for item in (first, second):
        item["prompt"] = "synthetic prompt <script>alert(1)</script>"
    dataset = {"meta": {"holdout_locked": True, "read_only": True, "reference_version": "synthetic_reference"},
               "summary": {"samples": 2, **review.count_rows(first["rows"] + second["rows"])},
               "samples": [first, second]}
    return dataset, criteria, humans, ai


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.dataset, self.criteria, self.humans, self.ai = fixture()

    def select(self, **kwargs):
        return review.select_rows(self.dataset, **kwargs)

    def test_known_counts_and_failed_items_keep_author_and_null_ai(self):
        result = self.select()
        self.assertEqual(result["filter_counts"], {"all": 6, "disagreement": 1, "unjudgeable": 2, "engineering_failure": 3})
        failed = self.select(filter_name="engineering_failure")["rows"]
        self.assertEqual(len(failed), 3)
        self.assertTrue(all(r["ai"] is None and r["human"] for r in failed))
        self.assertTrue(all(not r["disagreement"] for r in failed))

    def test_disagreement_includes_status_difference_only(self):
        result = self.select(filter_name="disagreement")
        self.assertEqual([(r["sample_id"], r["criterion_id"]) for r in result["rows"]], [("sample_008", "c1")])
        self.assertEqual(result["matched_samples"], 1)
        self.assertEqual(self.select(filter_name="unjudgeable")["matched_items"], 2)

    def test_status_mismatch_and_unknown_vs_decidable_counts_as_disagreement(self):
        ai = {**self.ai, "c3": label("c3", "pass")}
        sample = review.compare_sample("sample_008", self.criteria, self.humans, ai,
            task_status="succeeded", error_code=None, attempts=[], cached=False)
        self.assertEqual(sample["counts"]["disagreement"], 2)
        self.assertTrue(sample["rows"][2]["unjudgeable"])

    def test_sample_dimension_search_and_empty_intersection(self):
        self.assertEqual(self.select(sample="sample_008", dimension="subject_features")["matched_items"], 1)
        self.assertEqual(self.select(query="SAMPLE_016")["matched_items"], 3)
        self.assertEqual(self.select(query="合成条目c2")["matched_items"], 2)
        self.assertEqual(self.select(filter_name="engineering_failure", sample="sample_008")["rows"], [])

    def test_holdout_unknown_filter_or_dimension_rejected(self):
        with self.assertRaises(review.LockedSample):
            self.select(sample="sample_001")
        for kwargs in ({"filter_name": "holdout"}, {"dimension": "secret"}, {"query": "a" * 201}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.select(**kwargs)

    def test_display_projection_keeps_provenance_without_paths(self):
        result = self.select()
        text = json.dumps(result)
        for value in ("PRIVATE_PATH_MUST_NOT_LEAK", "NOT_FOR_DISPLAY", "private_path", "private_detail"):
            self.assertNotIn(value, text)
        self.assertIn("作者明确指定", result["rows"][0]["human_provenance"])
        self.assertNotIn("rows", result["samples"][0])

    def test_malformed_or_incomplete_success_refused(self):
        for labels in (None, {"c1": label("c1")}):
            with self.assertRaises(ValueError):
                review.compare_sample("sample_008", self.criteria, self.humans, labels,
                    task_status="succeeded", error_code=None, attempts=[], cached=False)

    def test_comparison_never_mutates_initial_reference(self):
        original = copy.deepcopy((self.dataset, self.humans, self.ai))
        self.select(filter_name="disagreement")
        self.select(filter_name="engineering_failure")
        self.assertEqual((self.dataset, self.humans, self.ai), original)

    def test_literal_markup_preserved_for_text_only_browser_rendering(self):
        self.humans["c1"]["label"]["candidate_observation"] = '<img src=x onerror="bad()">'
        item = review.compare_sample("sample_008", self.criteria, self.humans, self.ai,
            task_status="succeeded", error_code=None, attempts=[], cached=False)
        self.assertEqual(item["rows"][0]["human"]["candidate_observation"], '<img src=x onerror="bad()">')
        script = (review.WEB / "review.js").read_text(encoding="utf-8")
        self.assertNotIn("innerHTML", script)


class ReadOnlyAndGateTests(unittest.TestCase):
    def test_readonly_store_missing_database_not_created_and_writes_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "test.sqlite3"
            store = review.ReadOnlyRuns(p)
            self.assertFalse(p.exists())
            with self.assertRaises(sqlite3.OperationalError):
                store.snapshot("missing")
            self.assertFalse(p.exists())
            with closing(sqlite3.connect(p)) as db:
                db.execute("CREATE TABLE synthetic(value TEXT)")
            before = p.read_bytes()
            with store.connect() as db:
                with self.assertRaises(sqlite3.OperationalError):
                    db.execute("INSERT INTO synthetic VALUES('write')")
            with self.assertRaises(review.RunError):
                with store.connect(immediate=True):
                    pass
            self.assertEqual(p.read_bytes(), before)

    def test_holdout_batch_rejected_before_worker_inputs_or_evidence(self):
        frozen = {"reference": {}, "initial_rows": []}
        store = Mock()
        store.snapshot.return_value = {"batch": {"repeat": 0}, "tasks": [{"sample_id": "sample_001"}]}
        with patch.object(review, "load_registered_reference", return_value=frozen), \
                patch.object(review, "ReadOnlyRuns", return_value=store), patch.object(review, "Worker") as worker:
            with self.assertRaises(review.LockedSample):
                review.load_dataset()
            worker.assert_not_called()

    def test_image_route_guard_precedes_source_read(self):
        service = review.ReviewService.__new__(review.ReviewService)
        service.references = {};service.allowed = {"sample_008"}
        service.sources = Mock()
        with self.assertRaises(review.LockedSample):
            service.image("sample_001")

    def test_reload_failure_keeps_old_snapshot(self):
        service = review.ReviewService.__new__(review.ReviewService)
        service.database = "synthetic";service.batch_id = "synthetic";service.lock = threading.Lock()
        service.dataset = {"old": True}
        with patch.object(review, "load_dataset", side_effect=ValueError("corrupt evidence")):
            with self.assertRaises(ValueError):
                service.reload()
        self.assertEqual(service.dataset, {"old": True})


class ReviewHTTPTests(unittest.TestCase):
    def setUp(self):
        self.dataset = fixture()[0]
        service = review.ReviewService.__new__(review.ReviewService)
        service.dataset = self.dataset
        service.reload = Mock(return_value={"loaded_at": "synthetic"})
        def image(identifier):
            if identifier not in ("sample_008", "sample_016", "REF-MAIN", "REF-SUPPLEMENT"):
                raise review.LockedSample("保留集未解锁")
            return b"synthetic-image"
        service.image = Mock(side_effect=image)
        self.service = service
        self.server = review.make_server(service, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True);self.thread.start()
        self.address = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)

    def request(self, path, *, post=False, headers=None):
        return urllib.request.urlopen(urllib.request.Request(self.address + path,
            data=b"" if post else None, headers=headers or {}), timeout=5)

    def test_http_filters_and_headers(self):
        with self.request("/api/review?filter=disagreement&sample=all&dimension=all&q=") as response:
            self.assertEqual(json.load(response)["matched_items"], 1)
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            self.assertIn("script-src 'self'", response.headers["Content-Security-Policy"])
        with self.request("/api/review?filter=engineering_failure") as response:
            self.assertEqual(json.load(response)["matched_items"], 3)

    def test_holdout_api_images_raw_exports_and_unknown_query_blocked(self):
        for path, code in (("/api/review?sample=sample_001",403), ("/api/image/sample_001",403),
                           ("/api/review?split=holdout",400), ("/api/review?sample=all&sample=sample_001",400),
                           ("/api/export?sample_id=sample_001",404), ("/../data/sample_sources.jsonl",404),
                           ("/api/review?filter=bad",400)):
            with self.subTest(path=path), self.assertRaises(urllib.error.HTTPError) as caught:
                self.request(path)
            self.assertEqual(caught.exception.code, code)

    def test_write_send_and_cross_origin_endpoints_rejected(self):
        for path in ("/api/save", "/api/confirm", "/api/run", "/api/resume"):
            with self.subTest(path=path), self.assertRaises(urllib.error.HTTPError) as caught:
                self.request(path,post=True)
            self.assertEqual(caught.exception.code,405)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/api/review",headers={"Origin":"https://example.invalid"})
        self.assertEqual(caught.exception.code,403)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/api/reload",post=True,headers={"Host":"example.invalid"})
        self.assertEqual(caught.exception.code,403)
        self.service.reload.assert_not_called()

    def test_reload_and_safe_static_assets(self):
        with self.request("/api/reload",post=True) as response:
            self.assertEqual(json.load(response)["loaded_at"],"synthetic")
        self.service.reload.assert_called_once()
        for path in ("/", "/review.js", "/review.css", "/api/health", "/api/image/REF-MAIN"):
            with self.subTest(path=path), self.request(path) as response:
                self.assertEqual(response.status,200)


if __name__ == "__main__":
    unittest.main()
