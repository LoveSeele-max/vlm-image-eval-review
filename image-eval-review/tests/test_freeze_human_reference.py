"""E04 tests freeze synthetic labels, using isolated databases and private roots."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit_annotations as audit
import label_statistics as stats
import freeze_human_reference as freeze
from annotation_store import AnnotationStore


class HumanReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.base = Path(cls.fixture.name)
        cls.private = cls.base / "private"
        cls.database = cls.base / "synthetic.sqlite3"
        store = AnnotationStore(cls.database)
        with store.connect() as db:
            for sid, rules in store.expected.items():
                for cid in rules:
                    store._write(db, sid, {"criterion_id": cid, "status": "pass",
                        "candidate_region": "synthetic region", "candidate_observation": "synthetic observation",
                        "reason": "synthetic reason"}, "synthetic author", 1,
                        {"kind": "author_status_codex_evidence", "evidence_editor": "Codex"},
                        "synthetic explicit-author instruction")
        for sid in store.expected:
            store.confirm_initial(sid, "synthetic author")
        cls.annotations = cls.private / "e02"
        cls.statistics = cls.private / "e03"
        with patch.object(audit, "PRIVATE", cls.private), patch.object(audit, "validate_media", return_value=[]):
            audit.export_annotations(cls.database, cls.annotations)
        with patch.object(stats, "PRIVATE", cls.private):
            stats.generate_statistics(cls.annotations, cls.statistics)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.private = self.folder / "private"
        self.annotations = self.folder / "e02"
        self.statistics = self.folder / "e03"
        self.db = self.folder / "synthetic.sqlite3"
        shutil.copytree(type(self).annotations, self.annotations)
        shutil.copytree(type(self).statistics, self.statistics)
        shutil.copyfile(self.database, self.db)

    def tearDown(self):
        self.temp.cleanup()

    def create(self, **kwargs):
        with patch.object(freeze, "PRIVATE", self.private), patch.object(audit, "validate_media", return_value=[]):
            return freeze.freeze_reference(self.annotations, self.statistics, database=self.db,
                                           authorized_by="synthetic author", **kwargs)

    def test_roundtrip_expected_hash_version_and_source_bytes_preserved(self):
        before = self.db.read_bytes()
        result = self.create()
        self.assertEqual(self.db.read_bytes(), before)
        folder = Path(result["output"])
        loaded = freeze.load_reference(folder, expected_seal_sha256=result["seal_sha256"])
        self.assertEqual(len(loaded["initial_rows"]), 720)
        self.assertEqual(loaded["reference"]["sample_count"], 40)
        self.assertTrue(all(r["record"]["origin"]["kind"] == "author_status_codex_evidence"
                            for r in loaded["initial_rows"]))
        for source, subfolder in ((self.annotations, "annotations"), (self.statistics, "statistics")):
            self.assertEqual({p.name: p.read_bytes() for p in source.iterdir()},
                             {p.name: p.read_bytes() for p in (folder / subfolder).iterdir()})

    def test_existing_version_and_bad_version_refused(self):
        first = self.create()
        seal = (Path(first["output"]) / "seal.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.create()
        self.assertEqual((Path(first["output"]) / "seal.json").read_bytes(), seal)
        for version in ("../escape", "other/version", ""):
            with self.subTest(version=version), self.assertRaises(ValueError):
                self.create(version=version)

    def test_frozen_reference_self_contained_after_original_sources_removed(self):
        result = self.create()
        self.assertTrue(self.annotations.resolve().is_relative_to(self.folder.resolve()))
        self.assertTrue(self.statistics.resolve().is_relative_to(self.folder.resolve()))
        shutil.rmtree(self.annotations)
        shutil.rmtree(self.statistics)
        self.db.unlink()
        with patch.object(stats, "read_source", side_effect=AssertionError("external E02/E03 access")), \
                patch.object(audit, "read_database", side_effect=AssertionError("live DB access")), \
                patch.object(audit, "load_context", side_effect=AssertionError("live context access")), \
                patch.object(audit, "validate_media", side_effect=AssertionError("live media access")):
            self.assertEqual(freeze.verify_reference(result["output"], result["seal_sha256"])["item_count"], 720)

    def test_later_revision_does_not_change_primary_initial_labels(self):
        result = self.create()
        store = AnnotationStore(self.db)
        cid = next(iter(store.expected["sample_020"]))
        old = store.sample("sample_020")["records"][cid]["label"]
        store.save("sample_020", {**old, "status": "fail"}, "synthetic author", 1, "synthetic later correction")
        loaded = freeze.load_reference(result["output"], expected_seal_sha256=result["seal_sha256"])
        primary = next(r for r in loaded["initial_rows"] if r["sample_id"] == "sample_020"
                       and r["record"]["label"]["criterion_id"] == cid)
        self.assertEqual(primary["record"]["label"]["status"], "pass")
        history = store.export("sample_020")["history"]
        correction = next(r for r in history if r["criterion_id"] == cid and r["revision"] == 2)
        self.assertEqual(correction["change_reason"], "synthetic later correction")
        self.assertEqual(correction["payload"]["status"], "fail")

    def test_revisions_before_freeze_are_recorded_without_replacing_initial(self):
        store = AnnotationStore(self.db)
        cid = next(iter(store.expected["sample_020"]))
        old = store.sample("sample_020")["records"][cid]["label"]
        store.save("sample_020", {**old, "status": "fail"}, "synthetic author", 1, "synthetic before freeze")
        result = self.create()
        seal = freeze.read_json(Path(result["output"]) / "seal.json")
        self.assertEqual(len(seal["post_confirmation_changes_at_freeze"]), 1)
        self.assertEqual(freeze.verify_reference(result["output"])["item_count"], 720)

    def test_wrong_hash_wrong_version_or_missing_pin_rejected(self):
        result = self.create()
        with self.assertRaisesRegex(ValueError, "登记哈希"):
            freeze.verify_reference(result["output"], "0" * 64)
        with self.assertRaisesRegex(ValueError, "版本"):
            freeze.verify_reference(result["output"], expected_version="human_reference_v2")
        with self.assertRaisesRegex(ValueError, "冻结哈希"):
            freeze.load_reference(result["output"], expected_seal_sha256="")

    def test_registered_loader_pins_version_and_rejects_outside_path(self):
        result = self.create()
        registration = {"schema_version": 1, "relative_private_path": "human_references/" + freeze.VERSION,
                        **{key: result[key] for key in ("reference_version", "seal_sha256", "initial_labels_sha256",
                                                       "frozen_at", "sample_count", "item_count")}}
        path = self.folder / "registration.json"
        path.write_text(json.dumps(registration), encoding="utf-8")
        with patch.object(freeze, "PRIVATE", self.private):
            self.assertEqual(len(freeze.load_registered_reference(path)["initial_rows"]), 720)
            registration["relative_private_path"] = "../../outside"
            path.write_text(json.dumps(registration), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "private"):
                freeze.load_registered_reference(path)

    def test_tampered_label_and_rehashed_file_still_fail_registered_bindings(self):
        result = self.create()
        folder = Path(result["output"])
        target = folder / "annotations/initial_labels.jsonl"
        target.write_bytes(target.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "文件哈希"):
            freeze.verify_reference(folder)
        checks = freeze.read_json(folder / "checksums.json")
        checks["annotations/initial_labels.jsonl"] = audit.sha(target.read_bytes())
        (folder / "checksums.json").write_text(json.dumps(checks), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "来源文件绑定"):
            freeze.verify_reference(folder, result["seal_sha256"])

    def test_changed_live_initial_snapshot_rejects_stale_export(self):
        store = AnnotationStore(self.db)
        with store.connect() as db:
            db.execute("UPDATE initial_snapshots SET confirmed_by='different confirmer' WHERE sample_id='sample_020'")
        with self.assertRaisesRegex(ValueError, "初始快照"):
            self.create()
        self.assertFalse((self.private / "human_references" / freeze.VERSION).exists())

    def test_concurrent_edit_does_not_finish_freeze(self):
        original = audit.read_database
        count = []
        def changed(path):
            state = original(path)
            count.append(1)
            if len(count) > 1:
                state["labels"][0]["updated_at"] = "synthetic concurrent change"
            return state
        with patch.object(audit, "read_database", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "人工库变化"):
                self.create()
        self.assertFalse((self.private / "human_references" / freeze.VERSION / "checksums.json").exists())


if __name__ == "__main__":
    unittest.main()
