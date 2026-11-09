"""E02 fault cases use a synthetic SQLite fixture, never the author's database."""
import copy
import hashlib
import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit_annotations as audit
from annotation_store import AnnotationStore, encode


class AnnotationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_dir = tempfile.TemporaryDirectory()
        cls.template = Path(cls.fixture_dir.name) / "synthetic.sqlite3"
        cls.context = audit.load_context()
        store = AnnotationStore(cls.template)
        with store.connect() as db:
            for sid, criteria in store.expected.items():
                for cid in criteria:
                    store._write(db, sid, {
                        "criterion_id": cid, "status": "pass", "candidate_region": "synthetic region",
                        "candidate_observation": "synthetic observation", "reason": "synthetic reason",
                    }, "synthetic author", 1,
                        {"kind": "author_status_codex_evidence", "evidence_editor": "Codex"},
                        "synthetic explicit-author judgment; not real image labels")
        for sid in store.expected:
            store.confirm_initial(sid, "synthetic author")

    @classmethod
    def tearDownClass(cls):
        cls.fixture_dir.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.private = self.folder / "private"
        self.path = self.folder / "synthetic.sqlite3"
        shutil.copyfile(self.template, self.path)

    def tearDown(self):
        self.temp.cleanup()

    def state(self):
        return audit.read_database(self.path)

    def run_export(self, output=None):
        with patch.object(audit, "PRIVATE", self.private), \
                patch.object(audit, "load_context", return_value=self.context), \
                patch.object(audit, "validate_media", return_value=[]):
            return audit.export_annotations(self.path, output)

    def edit_snapshot(self, state, edit):
        row = state["initial_snapshots"][0]
        obj = json.loads(row["payload"])
        edit(obj)
        row["payload"] = encode(obj)
        row["sha256"] = audit.sha(row["payload"].encode("utf-8"))

    def test_roundtrip_read_only_preserves_all_sources_and_distinct_runs(self):
        before = self.path.read_bytes()
        first = self.run_export()
        second = self.run_export()
        self.assertNotEqual(first["output"], second["output"])
        self.assertEqual(self.path.read_bytes(), before)
        result = audit.verify_export(first["output"])
        self.assertEqual(result["summary"]["item_count"], 720)
        self.assertEqual(result["summary"]["initial_snapshot_count"], 40)
        exported = json.loads((Path(first["output"]) / "database_rows.json").read_text(encoding="utf-8"))
        self.assertEqual(exported, self.state())
        self.assertEqual(result["summary"]["initial_origin_counts"], {"author_status_codex_evidence": 720})

    def test_missing_database_not_created(self):
        absent = self.folder / "absent.sqlite3"
        with self.assertRaises(sqlite3.OperationalError):
            audit.read_database(absent)
        self.assertFalse(absent.exists())

    def test_missing_current_unknown_item_and_confirmation_rejected(self):
        for mutate in (
            lambda s: s["labels"].pop(),
            lambda s: s["labels"][0].update(criterion_id="unknown_criterion"),
            lambda s: s["initial_snapshots"].pop(),
        ):
            state = self.state()
            mutate(state)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                audit.audit_state(state, self.context)

    def test_illegal_status_and_blank_evidence_rejected(self):
        for change in ({"status": "not_applicable"}, {"status": ""}, {"reason": "  "}):
            state = self.state()
            row = state["labels"][0]
            label = json.loads(row["payload"])
            label.update(change)
            row["payload"] = encode(label)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, row["sample_id"]):
                audit.audit_state(state, self.context)

    def test_rule_binding_and_rehashed_snapshot_binding_mismatch(self):
        state = self.state()
        state["metadata"][0]["value"] = "{}"
        with self.assertRaisesRegex(ValueError, "规则绑定"):
            audit.audit_state(state, self.context)
        for key, value in (("rule_version", "rules_other"), ("candidate_sha256", "wrong"),
                           ("reference_sha256", ["wrong"])):
            state = self.state()
            self.edit_snapshot(state, lambda obj: obj.update({key: value}))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "错配"):
                audit.audit_state(state, self.context)

    def test_snapshot_hash_and_historical_source_tampering_rejected(self):
        state = self.state()
        state["initial_snapshots"][0]["payload"] += " "
        with self.assertRaisesRegex(ValueError, "快照哈希"):
            audit.audit_state(state, self.context)
        state = self.state()
        def change_record(obj):
            next(iter(obj["records"].values()))["origin"] = {"kind": "falsified provenance"}
        self.edit_snapshot(state, change_record)
        with self.assertRaisesRegex(ValueError, "历史来源"):
            audit.audit_state(state, self.context)
        state = self.state()
        state["history"].pop(0)
        with self.assertRaisesRegex(ValueError, "不连续"):
            audit.audit_state(state, self.context)

    def test_later_revision_keeps_initial_export_and_reports_change(self):
        store = AnnotationStore(self.path)
        cid = next(iter(store.expected["sample_020"]))
        label = store.sample("sample_020")["records"][cid]["label"]
        label = {**label, "status": "fail", "reason": "synthetic later correction"}
        store.save("sample_020", label, "synthetic author", 1, "synthetic correction reason")
        exported = Path(self.run_export()["output"])
        initial = [json.loads(line) for line in (exported / "initial_labels.jsonl").read_text(encoding="utf-8").splitlines()]
        current = [json.loads(line) for line in (exported / "current_labels.jsonl").read_text(encoding="utf-8").splitlines()]
        row = lambda rows: next(r for r in rows if r["sample_id"] == "sample_020" and r["record"]["label"]["criterion_id"] == cid)
        self.assertEqual(row(initial)["record"]["label"]["status"], "pass")
        self.assertEqual(row(current)["record"]["label"]["status"], "fail")
        self.assertEqual(len(audit.verify_export(exported)["summary"]["post_confirmation_changes"]), 1)

    def test_existing_or_nonprivate_output_is_rejected_without_overwrite(self):
        result = self.run_export()
        folder = Path(result["output"])
        before = {p.name: p.read_bytes() for p in folder.iterdir()}
        with self.assertRaises(FileExistsError):
            self.run_export(folder)
        self.assertEqual({p.name: p.read_bytes() for p in folder.iterdir()}, before)
        with self.assertRaisesRegex(ValueError, "private"):
            self.run_export(self.folder / "public")

    def test_concurrent_database_edit_prevents_completion_marker(self):
        original = audit.read_database
        calls = []
        def changing(path):
            state = original(path)
            calls.append(1)
            if len(calls) > 1:
                state["labels"][0]["updated_at"] = "synthetic concurrent edit"
            return state
        output = self.private / "incomplete"
        with patch.object(audit, "read_database", side_effect=changing):
            with self.assertRaisesRegex(ValueError, "人工库已变化"):
                self.run_export(output)
        self.assertFalse((output / "checksums.json").exists())

    def test_export_damage_and_rehashed_semantic_damage_detected(self):
        output = Path(self.run_export()["output"])
        target = output / "initial_labels.jsonl"
        original = target.read_bytes()
        target.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "文件哈希"):
            audit.verify_export(output)
        rows = original.decode("utf-8").splitlines()
        label = json.loads(rows[0]);label["record"]["label"]["status"] = "fail"
        rows[0] = encode(label)
        target.write_text("\n".join(rows) + "\n", encoding="utf-8")
        checks = output / "checksums.json"
        hashes = json.loads(checks.read_text(encoding="utf-8"))
        hashes[target.name] = audit.sha(target.read_bytes())
        checks.write_text(json.dumps(hashes), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "数据库快照不一致"):
            audit.verify_export(output)

    def test_source_media_hash_and_decode_verified_without_changes(self):
        source = self.folder / "synthetic.png"
        Image.new("RGB", (7, 9), "red").save(source)
        raw = source.read_bytes()
        context = {"references": [], "candidates": [{"sample_id": "synthetic", "source_path": str(source),
                    "sha256": hashlib.sha256(raw).hexdigest(), "width": 7, "height": 9}]}
        self.assertEqual(len(audit.validate_media(context)), 1)
        self.assertEqual(source.read_bytes(), raw)
        source.write_bytes(raw + b"changed")
        with self.assertRaisesRegex(ValueError, "图片哈希"):
            audit.validate_media(context)
        source.write_bytes(b"not an image")
        context["candidates"][0]["sha256"] = audit.sha(source.read_bytes())
        with self.assertRaises(OSError):
            audit.validate_media(context)

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, "重复键"):
            audit.strict_json('{"status":"pass","status":"fail"}')


if __name__ == "__main__":
    unittest.main()
