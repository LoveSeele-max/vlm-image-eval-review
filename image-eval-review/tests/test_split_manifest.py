import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_manifest import DATA, read_jsonl
from split_manifest import freeze, load_verified_inputs, partition, validate_smoke_selection


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.groups, cls.bindings = load_verified_inputs()

    def test_whole_groups_disjoint_complete_and_nearest_feasible_ratio(self):
        rows, sides, _ = partition(self.rows, self.groups)
        dev, holdout = (set(sides[side]["sample_ids"]) for side in ("development", "holdout"))
        self.assertEqual((len(dev), len(holdout)), (10, 30))
        self.assertFalse(dev & holdout)
        self.assertEqual(dev | holdout, {row["sample_id"] for row in self.rows})
        self.assertEqual(sides["development"]["group_ids"], ["group_004"])
        self.assertEqual(len({row["candidate_sha256"] for row in rows}), 40)
        for group in self.groups:
            members = set(group["sample_ids"])
            self.assertTrue(members <= dev or members <= holdout)
        originals = {row["sample_id"]: row for row in self.rows}
        for row in rows:
            self.assertEqual(row, {**originals[row["sample_id"]], "split": row["split"]})

    def test_reordering_does_not_change_partition_or_group_ranking(self):
        self.assertEqual(partition(self.rows, self.groups),
                         partition(list(reversed(self.rows)), list(reversed(self.groups))))

    def test_invalid_groups_duplicates_or_existing_assignments_rejected(self):
        for failure in ("missing", "duplicate_member", "count", "duplicate_hash", "split"):
            rows, groups = copy.deepcopy(self.rows), copy.deepcopy(self.groups)
            if failure == "missing":
                groups.pop()
            elif failure == "duplicate_member":
                groups[0]["sample_ids"][0] = groups[1]["sample_ids"][0]
            elif failure == "count":
                groups[0]["sample_count"] -= 1
            elif failure == "duplicate_hash":
                rows[0]["candidate_sha256"] = rows[1]["candidate_sha256"]
            else:
                rows[0]["split"] = "development"
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                partition(rows, groups)

    def test_freeze_preserves_pending_user_selection_and_second_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            split, created = freeze(data, self.rows, self.groups, self.bindings)
            self.assertTrue(created)
            self.assertEqual(split["smoke_selection"],
                             {"status": "awaiting_user", "sample_ids": [], "selected_by": None})
            before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in data.iterdir()}
            second, created = freeze(data, self.rows, self.groups, self.bindings)
            self.assertFalse(created)
            self.assertEqual(second, split)
            self.assertEqual(before, {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                                      for path in data.iterdir()})

    def test_frozen_manifest_assignment_and_input_tampering_rejected(self):
        for failure in ("manifest", "split", "inputs"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                data = Path(folder)
                freeze(data, self.rows, self.groups, self.bindings)
                if failure == "manifest":
                    with (data / "manifest.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write("\n")
                else:
                    path = data / "split.json"
                    split = json.loads(path.read_text(encoding="utf-8"))
                    if failure == "split":
                        split["partitions"]["development"]["sample_ids"].pop()
                    else:
                        split["inputs_sha256"] = {}
                    path.write_text(json.dumps(split), encoding="utf-8")
                before = {path.name: path.read_bytes() for path in data.iterdir()}
                with self.assertRaisesRegex(ValueError, "拒绝覆盖"):
                    freeze(data, self.rows, self.groups, self.bindings)
                self.assertEqual(before, {path.name: path.read_bytes() for path in data.iterdir()})

    def test_partial_freeze_is_not_overwritten(self):
        for name in ("manifest.jsonl", "split.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                data = Path(folder)
                (data / name).write_text("keep", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "不完整"):
                    freeze(data, self.rows, self.groups, self.bindings)
                self.assertEqual([path.name for path in data.iterdir()], [name])
                self.assertEqual((data / name).read_text(encoding="utf-8"), "keep")

    def test_smoke_ids_require_three_distinct_development_samples_selected_by_user(self):
        _, sides, _ = partition(self.rows, self.groups)
        dev = sides["development"]["sample_ids"]
        valid = {"status": "selected", "sample_ids": dev[:3], "selected_by": "user"}
        validate_smoke_selection(valid, dev)
        for ids, actor in ((dev[:2], "user"), ([dev[0]] * 3, "user"),
                           (dev[:2] + sides["holdout"]["sample_ids"][:1], "user"),
                           (dev[:3], "assistant")):
            with self.subTest(ids=ids, actor=actor), self.assertRaises(ValueError):
                validate_smoke_selection({"status": "selected", "sample_ids": ids,
                                          "selected_by": actor}, dev)
        with self.assertRaises(ValueError):
            validate_smoke_selection({"status": "awaiting_user", "sample_ids": dev[:3],
                                      "selected_by": None}, dev)


if __name__ == "__main__":
    unittest.main()
