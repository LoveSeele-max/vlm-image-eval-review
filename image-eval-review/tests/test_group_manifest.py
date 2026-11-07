import copy
import itertools
import json
import shutil
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_manifest import read_jsonl, unique_index
from group_manifest import (
    DATA, ROOT, deduplicate, fingerprint, group_rows, load_inputs, review_context,
    review_targets, screen_pairs, validate_review,
)


class GroupingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Inspect the historical B02 snapshot even after B04 freezes production files.
        # The production loader still refuses to rebuild B03 once those files exist.
        cls.draft = read_jsonl(DATA / "manifest.draft.jsonl")
        cls.sources = read_jsonl(DATA / "sample_sources.jsonl")
        cls.retained, cls.exclusions = deduplicate(cls.draft, cls.sources)
        cls.fingerprints, cls.pairs = screen_pairs(cls.retained, cls.sources)
        cls.context = review_context(cls.draft, cls.sources, cls.fingerprints, cls.pairs)
        cls.review = json.loads((DATA / "near_duplicate_review.json").read_text(encoding="utf-8"))

    def decisions(self):
        return validate_review(self.review, self.context, review_targets(self.pairs), self.retained)

    def test_exact_partition_and_canonical_source_preference(self):
        rows = unique_index(self.draft, "sample_id")
        mapping = unique_index(self.sources, "sample_id")
        retained = {row["sample_id"] for row in self.retained}
        excluded = {row["excluded_sample_id"] for row in self.exclusions}
        self.assertEqual((len(retained), len(excluded)), (40, 8))
        self.assertFalse(retained & excluded)
        self.assertEqual(retained | excluded, set(rows))
        self.assertEqual(len({row["candidate_sha256"] for row in self.retained}), 40)
        for item in self.exclusions:
            left, right = item["excluded_sample_id"], item["retained_sample_id"]
            self.assertIn(right, retained)
            self.assertEqual(rows[left]["candidate_sha256"], rows[right]["candidate_sha256"])
            self.assertEqual(mapping[left]["byte_duplicate_candidate_of"], mapping[right]["source_path"])
            self.assertIsNone(mapping[right]["byte_duplicate_candidate_of"])
            self.assertEqual(item["reason"], "identical_raw_sha256")

    def test_preserves_original_fields_ids_and_unassigned_split(self):
        before = copy.deepcopy(self.draft)
        grouped, groups = group_rows(self.retained, self.sources, self.decisions())
        originals = unique_index(self.draft, "sample_id")
        self.assertEqual(self.draft, before)
        self.assertEqual(len(groups), 4)
        for row in grouped:
            expected = {**originals[row["sample_id"]], "group_id": row["group_id"],
                        "selection_status": "included_B03"}
            self.assertEqual(row, expected)
            self.assertIsNone(row["split"])

    def test_same_seed_condition_and_base_lora_contrasts_never_cross_groups(self):
        grouped, groups = group_rows(self.retained, self.sources, self.decisions())
        mapping = unique_index(self.sources, "sample_id")
        by_seed, by_condition = defaultdict(set), defaultdict(set)
        for row in grouped:
            source = mapping[row["sample_id"]]
            by_seed[source["seed"]].add(row["group_id"])
            by_condition[source["generation_condition_group_id"]].add(row["group_id"])
        self.assertEqual(set(by_seed), {42, 123, 456, 789})
        self.assertEqual(len(by_condition), 20)
        self.assertTrue(all(len(groups) == 1 for groups in [*by_seed.values(), *by_condition.values()]))
        for group in groups:
            self.assertEqual(group["sample_count"], 10)
            self.assertEqual(group["model_group_counts"], {"base": 5, "lora": 5})

    def test_all_pairs_are_screened_and_cross_seed_targets_are_reviewed(self):
        ids = [row["sample_id"] for row in self.retained]
        self.assertEqual({(row["left"], row["right"]) for row in self.pairs},
                         set(itertools.combinations(ids, 2)))
        self.assertEqual(len(self.pairs), 780)
        self.assertEqual(sum(row["screen_flag"] for row in self.pairs), 14)
        self.assertEqual(sum(row["screen_flag"] and not row["same_seed"] for row in self.pairs), 2)
        self.assertEqual(len(self.decisions()), 9)
        self.assertTrue(all(row["decision"] == "distinct_images" for row in self.decisions()))

    def test_reviewed_cross_seed_edges_merge_transitively(self):
        decisions = [
            {"left": "sample_003", "right": "sample_034", "decision": "same_group"},
            {"left": "sample_017", "right": "sample_034", "decision": "same_group"},
        ]
        grouped, groups = group_rows(self.retained, self.sources, decisions)
        self.assertEqual(sorted(group["sample_count"] for group in groups), [10, 30])
        mapping = unique_index(grouped, "sample_id")
        self.assertEqual(mapping["sample_003"]["group_id"], mapping["sample_017"]["group_id"])
        self.assertEqual(mapping["sample_003"]["group_id"], mapping["sample_034"]["group_id"])

    def test_missing_unknown_duplicate_pending_and_stale_reviews_rejected(self):
        for failure in ("missing", "unknown", "duplicate", "pending", "stale_image",
                        "stale_context", "no_reason", "schema", "reviewer"):
            review = copy.deepcopy(self.review)
            if failure == "missing":
                review["decisions"].pop()
            elif failure == "unknown":
                review["decisions"][0]["left"] = "sample_999"
            elif failure == "duplicate":
                review["decisions"].append(copy.deepcopy(review["decisions"][0]))
            elif failure == "pending":
                review["decisions"][0]["decision"] = "pending"
            elif failure == "stale_image":
                review["decisions"][0]["left_sha256"] = "0" * 64
            elif failure == "stale_context":
                review["context_sha256"] = "0" * 64
            elif failure == "schema":
                review["schema_version"] = 2
            elif failure == "reviewer":
                review["reviewer"] = "pending"
            else:
                review["decisions"][0]["reason"] = ""
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                validate_review(review, self.context, review_targets(self.pairs), self.retained)

    def test_conflicting_byte_duplicate_metadata_rejected(self):
        excluded_id = self.exclusions[0]["excluded_sample_id"]
        for field in ("prompt", "requirements", "reference_ids"):
            draft = copy.deepcopy(self.draft)
            row = next(row for row in draft if row["sample_id"] == excluded_id)
            row[field] = "changed" if field == "prompt" else []
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "冲突"):
                deduplicate(draft, self.sources)
        sources = copy.deepcopy(self.sources)
        next(row for row in sources if row["sample_id"] == excluded_id)["seed"] = -1
        with self.assertRaisesRegex(ValueError, "冲突"):
            deduplicate(self.draft, sources)

    def test_bad_joins_pointers_and_non_draft_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError, "集合不一致"):
            deduplicate(self.draft, self.sources[:-1])
        sources = copy.deepcopy(self.sources)
        next(row for row in sources if row["byte_duplicate_candidate_of"])["byte_duplicate_candidate_of"] = "unknown"
        with self.assertRaisesRegex(ValueError, "指针"):
            deduplicate(self.draft, sources)
        draft = copy.deepcopy(self.draft)
        draft[0]["split"] = "dev"
        with self.assertRaisesRegex(ValueError, "B02 草稿"):
            deduplicate(draft, self.sources)
        with self.assertRaisesRegex(ValueError, "sample_id"):
            deduplicate([*self.draft, self.draft[0]], self.sources)

    def test_deterministic_after_input_reordering_and_review_binding_changes(self):
        kept, excluded = deduplicate(list(reversed(self.draft)), list(reversed(self.sources)))
        self.assertEqual((kept, excluded), (self.retained, self.exclusions))
        self.assertEqual(group_rows(kept, list(reversed(self.sources)), list(reversed(self.decisions()))),
                         group_rows(self.retained, self.sources, self.decisions()))
        self.assertEqual(review_context(list(reversed(self.draft)), list(reversed(self.sources)),
                                        self.fingerprints, self.pairs), self.context)
        altered = copy.deepcopy(self.sources)
        altered[0]["seed"] = -1
        self.assertNotEqual(review_context(self.draft, altered, self.fingerprints, self.pairs), self.context)

    def test_frozen_files_and_changed_upstream_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("manifest.jsonl", "split.json"):
                frozen = root / name
                frozen.write_text("keep", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "B04"):
                    load_inputs(root, root)
                self.assertEqual(frozen.read_text(encoding="utf-8"), "keep")
                frozen.unlink()
            build = {"inputs_sha256": {"changed.txt": "0" * 64}}
            (root / "manifest.build.json").write_text(json.dumps(build), encoding="utf-8")
            (root / "changed.txt").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "证据哈希"):
                load_inputs(root, root)

    def test_tampered_draft_and_source_mapping_rejected_against_a03_b01(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            names = ("manifest.build.json", "manifest.draft.jsonl", "sample_sources.jsonl",
                     "prompt_requirements.json", "reference_assets.json")
            for target in ("manifest.draft.jsonl", "sample_sources.jsonl"):
                for name in names:
                    shutil.copyfile(DATA / name, data / name)
                rows = [json.loads(line) for line in (data / target).read_text(encoding="utf-8").splitlines()]
                if target == "manifest.draft.jsonl":
                    rows[0]["prompt"] = "unverified new prompt"
                else:
                    rows[0]["seed"] = -1
                (data / target).write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                with self.subTest(target=target), self.assertRaisesRegex(ValueError, "重建结果不一致"):
                    load_inputs(data, ROOT)

    def test_fingerprints_detect_copy_and_brightness_variant_without_deleting(self):
        with tempfile.TemporaryDirectory() as folder:
            original, altered = Path(folder) / "a.png", Path(folder) / "b.png"
            image = Image.new("RGB", (80, 80), (30, 40, 50))
            ImageDraw.Draw(image).rectangle((10, 20, 55, 60), fill=(150, 180, 110))
            image.save(original)
            ImageEnhance.Brightness(image).enhance(1.08).save(altered)
            left, right = fingerprint(original), fingerprint(altered)
            self.assertEqual(left, fingerprint(original))
            self.assertNotEqual(left["rgb"], right["rgb"])
            _, pairs = screen_pairs(
                [{"sample_id": "a", "candidate_sha256": "original"},
                 {"sample_id": "b", "candidate_sha256": "brightness_variant"}],
                [{"sample_id": "a", "source_path": str(original), "seed": 1},
                 {"sample_id": "b", "source_path": str(altered), "seed": 2}])
            self.assertTrue(pairs[0]["screen_flag"])
            self.assertTrue(original.exists() and altered.exists())


if __name__ == "__main__":
    unittest.main()
