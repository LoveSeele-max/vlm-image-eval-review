import copy
import hashlib
import json
import re
import sqlite3
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annotation_app import AnnotationService, make_server
from annotation_store import AnnotationError, AnnotationStore, ConflictError, parse_markdown


def label_for(store, sid="sample_020", key=None, status="pass"):
    key = key or next(iter(store.expected[sid]))
    return {"criterion_id": key, "status": status, "candidate_region": "synthetic region",
            "candidate_observation": "synthetic observation", "reason": "synthetic reason"}


def markdown_for(store, sid="sample_020"):
    labels = [label_for(store, sid, key) for key in store.expected[sid]]
    text = f"## {sid}\n\n" + "\n".join(
        f"| name<br>`{row['criterion_id']}` | 符合 | synthetic region | synthetic observation | synthetic reason | □ |"
        for row in labels)
    return text + "\n", labels


class AnnotationStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.path = self.folder / "annotations.sqlite3"
        self.store = AnnotationStore(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_blank_defaults_and_progress_never_assume_pass(self):
        page = self.store.sample("sample_020")
        self.assertEqual(page["records"], {})
        self.assertEqual(page["progress"], {
            "sample_id": "sample_020", "complete": 0, "saved": 0, "total": 14})
        self.assertIsNone(page["initial"])
        draft = label_for(self.store, status="")
        draft["reason"] = ""
        self.store.save("sample_020", draft, "tester", 0)
        reopened = AnnotationStore(self.path).sample("sample_020")
        self.assertEqual(reopened["progress"]["complete"], 0)
        self.assertEqual(reopened["progress"]["saved"], 1)
        self.assertEqual(reopened["records"][draft["criterion_id"]]["label"], draft)

    def test_save_restore_conflict_and_idempotent_save(self):
        label = label_for(self.store, status="fail")
        self.assertEqual(self.store.save("sample_020", label, "水水", 0), 1)
        restored = AnnotationStore(self.path)
        self.assertEqual(restored.sample("sample_020")["records"][label["criterion_id"]]["label"], label)
        self.assertEqual(restored.save("sample_020", label, "水水", 1), 1)
        with self.assertRaises(ConflictError):
            self.store.save("sample_020", label, "水水", 0)
        label["reason"] = "updated synthetic reason"
        self.assertEqual(self.store.save("sample_020", label, "水水", 1), 2)
        history = self.store.export("sample_020")["history"]
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["payload"]["reason"], "synthetic reason")

    def test_invalid_labels_and_bad_ids_rejected(self):
        for change in ({"status": "not_applicable"}, {"status": "PASS"},
                       {"status": None}, {"criterion_id": "wooden_table"},
                       {"reason": []}, {"required": False}):
            label = {**label_for(self.store), **change}
            with self.subTest(change=change), self.assertRaises(AnnotationError):
                self.store.save("sample_020", label, "水水", 0)
        with self.assertRaises(AnnotationError):
            self.store.save("sample_020", label_for(self.store), "", 0)
        with self.assertRaises(AnnotationError):
            self.store.sample("sample_999")

    def test_initial_snapshot_immutable_and_revision_reason_required(self):
        with self.assertRaises(AnnotationError):
            self.store.confirm_initial("sample_020", "水水")
        for key in self.store.expected["sample_020"]:
            self.store.save("sample_020", label_for(self.store, key=key), "水水", 0)
        digest = self.store.confirm_initial("sample_020", "水水")
        original = self.store.export("sample_020")["initial_snapshot"]
        self.assertEqual(self.store.confirm_initial("sample_020", "someone else"), digest)
        label = label_for(self.store, status="fail")
        with self.assertRaises(AnnotationError):
            self.store.save("sample_020", label, "水水", 1)
        self.store.save("sample_020", label, "水水", 1, "synthetic revision")
        current = self.store.export("sample_020")
        self.assertEqual(current["initial_snapshot"], original)
        self.assertEqual(current["history"][-1]["change_reason"], "synthetic revision")
        self.assertTrue(current["history"][-1]["origin"]["after_initial_confirmation"])

    def test_markdown_import_exact_and_repeat_does_not_overwrite_page_edits(self):
        text, labels = markdown_for(self.store)
        path = self.folder / "labels.md"
        path.write_text(text, encoding="utf-8")
        self.assertEqual(self.store.import_markdown(path)["imported"], 14)
        records = self.store.sample("sample_020")["records"]
        self.assertEqual([records[row["criterion_id"]]["label"] for row in labels], labels)
        changed = {**labels[0], "status": "fail"}
        self.store.save("sample_020", changed, "水水", 1)
        self.assertTrue(self.store.import_markdown(path)["already_imported"])
        self.assertEqual(self.store.sample("sample_020")["records"][
            changed["criterion_id"]]["label"], changed)
        path.write_text(text + "\n", encoding="utf-8")
        with self.assertRaises(ConflictError):
            self.store.import_markdown(path)

    def test_import_bad_rows_rollback_entire_document(self):
        text, labels = markdown_for(self.store)
        next_text, _ = markdown_for(self.store, "sample_029")
        valid_row = next(line for line in next_text.splitlines() if line.startswith("|"))
        path = self.folder / "bad.md"
        for malformed in (text + next_text + valid_row + "\n",
                          text + next_text.replace(" | 符合 | ", " | 待填写 | ", 1),
                          text + next_text.replace("`subject_top_ear_geometry`",
                                                   "`unknown_criterion`", 1)):
            path.write_text(malformed, encoding="utf-8")
            with self.assertRaises(AnnotationError):
                self.store.import_markdown(path)
            self.assertEqual(self.store.sample("sample_020")["records"], {})

    def test_assisted_import_preserves_original_status_and_both_evidence_versions(self):
        original, labels = markdown_for(self.store)
        amended = original.replace("synthetic observation", "assisted observation")
        snapshot = self.folder / "original.md"
        working = self.folder / "working.md"
        log_path = self.folder / "supplement.json"
        snapshot.write_text(original, encoding="utf-8")
        working.write_text(amended, encoding="utf-8")
        log = {"snapshot": snapshot.name, "source_sha256": hashlib.sha256(
            snapshot.read_bytes()).hexdigest(),
            "supplemented_sha256": hashlib.sha256(working.read_bytes()).hexdigest(),
            "supplement_date": "2026-11-07"}
        log_path.write_text(json.dumps(log), encoding="utf-8")
        self.store.import_markdown(working, supplement_path=log_path)
        exported = self.store.export("sample_020")
        self.assertEqual(len(exported["history"]), 28)
        for row in exported["sample"]["records"].values():
            self.assertEqual(row["label"]["status"], "pass")
            self.assertEqual(row["origin"]["kind"], "author_status_codex_evidence")
            self.assertEqual(row["revision"], 2)
        originals = [row for row in exported["history"] if row["revision"] == 1]
        self.assertTrue(all(row["payload"]["candidate_observation"] == "synthetic observation"
                            for row in originals))
        self.assertEqual(parse_markdown(amended.replace("<br>", "&lt;br&gt;")),
                         parse_markdown(amended))

    def test_assisted_import_rejects_changed_status_or_unmatched_hash(self):
        text, _ = markdown_for(self.store)
        snapshot, working, log_path = (self.folder / name for name in (
            "original.md", "working.md", "supplement.json"))
        snapshot.write_text(text, encoding="utf-8")
        changed = text.replace(" | 符合 | ", " | 不符 | ", 1)
        working.write_text(changed, encoding="utf-8")
        log = {"snapshot": snapshot.name, "source_sha256": hashlib.sha256(
            snapshot.read_bytes()).hexdigest(),
            "supplemented_sha256": hashlib.sha256(working.read_bytes()).hexdigest(),
            "supplement_date": "2026-11-07"}
        log_path.write_text(json.dumps(log), encoding="utf-8")
        with self.assertRaises(AnnotationError):
            self.store.import_markdown(working, supplement_path=log_path)
        log["supplemented_sha256"] = "wrong"
        log_path.write_text(json.dumps(log), encoding="utf-8")
        with self.assertRaises(AnnotationError):
            self.store.import_markdown(working, supplement_path=log_path)
        self.assertEqual(self.store.sample("sample_020")["records"], {})

    def test_rule_binding_mismatch_is_rejected(self):
        with self.store.connect() as db:
            db.execute("UPDATE metadata SET value='different version' WHERE key='binding'")
        with self.assertRaises(AnnotationError):
            AnnotationStore(self.path)

    def test_display_data_excludes_generation_metadata_and_source_paths(self):
        display = json.dumps({"overview": self.store.overview(),
                              "samples": [self.store.sample(sid)
                                          for sid in self.store.manifest]}, ensure_ascii=False)
        for forbidden in ("source_path", "file_name", "model_group", "lora_scale",
                          "generation_config", "base_P1_seed789.png", "E:/", "candidate_status"):
            self.assertNotIn(forbidden, display)
        self.assertEqual(sum(row["total"] for row in self.store.overview()["samples"]), 720)
        self.assertEqual(len(self.store.overview()["smoke_ids"]), 3)


class AnnotationHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = AnnotationService(Path(self.temp.name) / "test.sqlite3", auto_import=False)
        self.server = make_server(self.service, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.address = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, route, body=None, token=True, origin=None):
        headers = {}
        if token:
            headers["X-Annotation-Token"] = self.service.token
        if origin:
            headers["Origin"] = origin
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        return urllib.request.urlopen(urllib.request.Request(
            self.address + route, data=data, headers=headers), timeout=5)

    def test_http_save_restore_draft_and_confirmation_errors(self):
        payload = {"sample_id": "sample_020", "label": label_for(self.service.store),
                   "annotator": "水水", "expected_revision": 0, "change_reason": ""}
        with self.request("/api/save", payload) as response:
            self.assertEqual(json.load(response)["revision"], 1)
        with self.request("/api/sample?sample_id=sample_020") as response:
            result = json.load(response)
            self.assertEqual(result["progress"]["complete"], 1)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/api/save", payload)
        self.assertEqual(caught.exception.code, 409)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/api/confirm", {"sample_id": "sample_020", "annotator": "水水"})
        self.assertEqual(caught.exception.code, 400)

    def test_token_origin_and_static_path_boundaries(self):
        payload = {"sample_id": "sample_020", "label": label_for(self.service.store),
                   "annotator": "水水", "expected_revision": 0, "change_reason": ""}
        for kwargs in ({"token": False}, {"origin": "https://example.invalid"}):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.request("/api/save", payload, **kwargs)
            self.assertEqual(caught.exception.code, 403)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/../data/sample_sources.jsonl")
        self.assertEqual(caught.exception.code, 404)
        with self.request("/") as response:
            html = response.read().decode()
            self.assertIn(self.service.token, html)
            self.assertNotIn("source_path", html)

    def test_reference_and_anonymous_candidate_images(self):
        # Synthetic source bytes exercise HTTP routing without requiring author media.
        from unittest.mock import patch
        with patch.object(self.service, "image", return_value=b"synthetic-image") as image:
            with self.request("/api/image/sample_020") as response:
                self.assertEqual(response.headers["Content-Type"], "image/png")
                self.assertEqual(response.read(), b"synthetic-image")
            image.assert_called_once_with("sample_020")


if __name__ == "__main__":
    unittest.main()
