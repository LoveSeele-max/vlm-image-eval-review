import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_manifest
from build_manifest import annotation_view, build_rows, read_jsonl, validate_files

ROOT = Path(__file__).resolve().parents[2]


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.sources = read_jsonl(ROOT / "A03_候选图片来源清单.jsonl")
        self.inspections = read_jsonl(ROOT / "B01_图片检查清单.jsonl")
        self.catalog = json.loads((ROOT / "image-eval-review/data/prompt_requirements.json").read_text(encoding="utf-8"))

    def build(self, previous=None):
        return build_rows(self.sources, self.inspections, self.catalog, previous)

    def test_all_files_preserved_and_configuration_hidden_from_annotation_view(self):
        manifest, internal, references = self.build()
        self.assertEqual(len(manifest), 48)
        self.assertEqual(len(internal), 48)
        self.assertEqual(len({row["sample_id"] for row in manifest}), 48)
        self.assertEqual({row["source_record_id"] for row in internal},
                         {row["record_id"] for row in self.sources})
        self.assertEqual({row["sample_id"] for row in manifest}, {row["sample_id"] for row in internal})
        self.assertEqual(len(references), 2)
        self.assertEqual(len({row["candidate_sha256"] for row in manifest}), 40)
        for row in manifest:
            self.assertIsNone(row["group_id"])
            self.assertIsNone(row["split"])
            self.assertEqual(row["selection_status"], "pending_B03")
            self.assertEqual(row["requirements_status"], "draft_pending_C02")
            view = annotation_view({**row, "model_group": "must_not_show", "ai_result": "must_not_show"})
            self.assertEqual(set(view), {"sample_id", "candidate_asset_id", "reference_ids",
                                         "prompt", "requirements", "requirements_status"})
            self.assertNotIn("must_not_show", json.dumps(view))
            for field in ("source_path", "file_name", "batch", "seed", "model_group", "prompt_id"):
                self.assertNotIn(field, row)
        self.assertEqual(sum(row["byte_duplicate_candidate_of"] is not None for row in internal), 8)

    def test_ids_stable_after_reordering_and_reusing_existing_mapping(self):
        manifest, internal, references = self.build()
        self.sources.reverse()
        self.inspections.reverse()
        self.assertEqual(self.build(), (manifest, internal, references))
        self.assertEqual(self.build(list(reversed(internal)))[0:2], (manifest, internal))
        source_order = [row["source_record_id"] for row in internal]
        self.assertNotEqual(source_order, sorted(source_order))

    def test_aliases_share_requirements_and_provisional_condition_groups(self):
        manifest, internal, _ = self.build()
        by_sample = {row["sample_id"]: row for row in manifest}
        by_path = {row["source_path"]: row for row in internal}
        self.assertEqual(len({row["generation_condition_group_id"] for row in internal}), 20)
        for row in internal:
            if row["byte_duplicate_candidate_of"]:
                original = by_path[row["byte_duplicate_candidate_of"]]
                self.assertEqual(row["generation_condition_group_id"], original["generation_condition_group_id"])
                left, right = by_sample[row["sample_id"]], by_sample[original["sample_id"]]
                self.assertEqual(left["prompt"], right["prompt"])
                self.assertEqual(left["requirements"], right["requirements"])
                self.assertNotEqual(left["sample_id"], right["sample_id"])

    def test_requirements_have_quotes_and_do_not_add_reference_only_attributes(self):
        manifest, internal, _ = self.build()
        by_sample = {row["sample_id"]: row for row in manifest}
        for source in internal:
            row = by_sample[source["sample_id"]]
            ids = {item["id"] for item in row["requirements"]}
            self.assertEqual(len(ids), len(row["requirements"]))
            for item in row["requirements"]:
                self.assertIn(item["source_quote"], row["prompt"])
            self.assertFalse({"straw_hat", "white_background", "zqv"} & ids)
            if source["prompt_id"] in ("P0", "P1"):
                self.assertEqual(ids, {"photo_style", "figurine", "wooden_table"})
            elif source["prompt_id"] == "TRIGGER_ORIGINAL":
                self.assertEqual(ids, {"photo_style", "figurine"})

    def test_changed_input_pool_and_duplicate_previous_ids_are_rejected(self):
        _, internal, _ = self.build()
        with self.assertRaisesRegex(ValueError, "输入集合变化"):
            self.build(internal[:-1])
        internal[0]["sample_id"] = internal[1]["sample_id"]
        with self.assertRaisesRegex(ValueError, "sample_id"):
            self.build(internal)

    def test_missing_failed_mismatched_and_duplicate_inspections_are_rejected(self):
        originals = copy.deepcopy(self.inspections)
        for failure in ("missing", "failed", "mismatched", "duplicate"):
            self.inspections = copy.deepcopy(originals)
            if failure == "missing":
                self.inspections.pop(0)
            elif failure == "failed":
                self.inspections[0]["status"] = "failed"
            elif failure == "mismatched":
                self.inspections[0]["source_path"] = "different/path"
            else:
                self.inspections.append(self.inspections[0])
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                self.build()

    def test_unknown_prompt_wrong_alias_and_unsupported_requirement_are_rejected(self):
        sources, catalog = copy.deepcopy(self.sources), copy.deepcopy(self.catalog)
        for failure in ("unknown", "alias", "requirement"):
            self.sources, self.catalog = copy.deepcopy(sources), copy.deepcopy(catalog)
            if failure == "unknown":
                self.sources[0]["prompt_text"] += " altered"
            elif failure == "alias":
                self.sources[0]["prompt_id"] = "P2"
            else:
                self.catalog["requirements"]["wooden_table"]["source_quote"] = "not in the prompt"
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                self.build()

    def test_stale_or_failed_image_evidence_rejected_without_changing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.png"
            path.write_bytes(b"changed after inspection")
            record = {"inspection_id": "test", "source_path": str(path),
                      "status": "passed", "decode_ok": True, "path_valid": True, "sha256": "0" * 64}
            with self.assertRaisesRegex(ValueError, "哈希不一致"):
                validate_files([record])
            self.assertEqual(path.read_bytes(), b"changed after inspection")
            record["status"] = "failed"
            with self.assertRaisesRegex(ValueError, "检查未通过"):
                validate_files([record])

    def test_frozen_stage_files_block_draft_builder(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            for name in ("manifest.jsonl", "split.json"):
                path = data / name
                path.write_text("keep", encoding="utf-8")
                with patch.object(build_manifest, "DATA", data):
                    with self.assertRaisesRegex(ValueError, "拒绝"):
                        build_manifest.main()
                self.assertEqual(path.read_text(encoding="utf-8"), "keep")
                path.unlink()

    def test_input_hash_ignores_git_newline_conversion_but_image_hash_does_not(self):
        with tempfile.TemporaryDirectory() as folder:
            unix, windows = Path(folder) / "unix", Path(folder) / "windows"
            unix.write_bytes(b'{"x": 1}\n{"x": 2}\n')
            windows.write_bytes(b'{"x": 1}\r\n{"x": 2}\r\n')
            self.assertEqual(build_manifest.text_digest(unix), build_manifest.text_digest(windows))
            self.assertNotEqual(build_manifest.digest(unix), build_manifest.digest(windows))


if __name__ == "__main__":
    unittest.main()
