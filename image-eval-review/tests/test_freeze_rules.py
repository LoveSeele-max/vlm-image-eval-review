import copy
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_manifest import DATA, read_jsonl, text_digest
from evaluate_results import expected_for_sample, load_context
from freeze_rules import (
    PROMPT, formalize, freeze, read_json, render_prompt, verify_bundle,
)
from prepare_image import prepare_image


class FrozenRuleTests(unittest.TestCase):
    def test_bundle_coverage_and_formal_context(self):
        summary = verify_bundle()
        self.assertEqual(summary["sample_count"], 40)
        self.assertEqual(summary["active_criterion_types"], 26)
        self.assertEqual(summary["label_assignment_count"], 720)
        self.assertEqual(summary["required_assignment_count"], 640)
        policy, reference, evaluation, samples = load_context()
        self.assertEqual(policy["rule_version"], "rules_v1")
        self.assertEqual(policy["status"], "frozen")
        for sample in samples:
            expected = expected_for_sample(
                sample["sample_id"], reference, evaluation, samples)
            self.assertNotIn("wooden_table", expected)
            self.assertEqual(sum(not row["required"] for row in expected.values()), 2)

    def test_formalization_preserves_boundaries_requiredness_and_draft_sources(self):
        drafts = (
            read_json(DATA / "reference_spec.draft.json"),
            read_json(DATA / "evaluation_spec.draft.json"),
            read_jsonl(DATA / "sample_criteria.draft.jsonl"),
            read_json(DATA / "label_policy.draft.json"))
        before = copy.deepcopy(drafts)
        reference, evaluation, samples, policy = formalize(*drafts, "test-time")
        self.assertEqual(drafts, before)
        for original, formal in zip(drafts[0]["criteria"], reference["criteria"]):
            for field, value in original.items():
                if field != "pre_freeze_review":
                    self.assertEqual(formal[field], value)
        for field in ("structure_criteria", "prompt_criteria", "excluded_prompt_requirements"):
            self.assertEqual(evaluation[field], drafts[1][field])
        for original, formal in zip(drafts[2], samples):
            self.assertEqual({**original, "criteria_status": "frozen"}, formal)
        for field in ("active_label_statuses", "aggregation_order", "label_fields",
                      "auxiliary_policy", "not_applicable_policy"):
            self.assertEqual(policy[field], drafts[3][field])

    def test_prompt_covers_exact_sample_criteria_without_configuration_or_paths(self):
        _, reference, evaluation, samples = load_context()
        manifest = {row["sample_id"]: row for row in read_jsonl(DATA / "manifest.jsonl")}
        for sample in samples:
            sample_id = sample["sample_id"]
            prompt = render_prompt(sample_id)
            self.assertNotIn("{{RULE_VERSION}}", prompt)
            self.assertNotIn("{{SAMPLE_CONTEXT_JSON}}", prompt)
            self.assertIn("木桌", prompt)
            self.assertIn("图 3 是唯一候选图", prompt)
            context = json.loads(prompt.split("待核对材料及逐项规则：\n", 1)[1])
            expected = expected_for_sample(sample_id, reference, evaluation, samples)
            self.assertEqual(context["sample_id"], sample_id)
            self.assertEqual(context["original_prompt"], manifest[sample_id]["prompt"])
            ids = [row["id"] for row in context["active_criteria"]]
            self.assertEqual(set(ids), set(expected))
            self.assertEqual(len(ids), len(expected))
            self.assertNotIn("wooden_table", ids)
            for forbidden in ("generation_config", "source_path", "seed", "lora", "E:/"):
                self.assertNotIn(forbidden, json.dumps(context, ensure_ascii=False))
        with self.assertRaises(ValueError):
            render_prompt("sample_999")

    def test_schema_matches_c03_fields_and_disallows_unassigned_status(self):
        schema = read_json(DATA / "response_schema.json")
        label_schema = schema["properties"]["labels"]["items"]
        policy, reference, evaluation, samples = load_context()
        all_ids = {key for sample in samples for key in expected_for_sample(
            sample["sample_id"], reference, evaluation, samples)}
        self.assertEqual(set(label_schema["properties"]["criterion_id"]["enum"]), all_ids)
        self.assertEqual(label_schema["properties"]["status"]["enum"],
                         policy["active_label_statuses"])
        self.assertEqual(set(label_schema["required"]), set(policy["label_fields"]))
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(label_schema["additionalProperties"])

    def test_corrupted_artifact_prompt_or_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            shutil.copytree(DATA, data)
            prompt = Path(directory) / "judge_v1.txt"
            shutil.copyfile(PROMPT, prompt)
            self.assertEqual(verify_bundle(data=data, prompt=prompt)["status"], "verified")
            for target in (data / "reference_spec.json", data / "manifest.jsonl",
                           data / "image_input_policy.json", prompt):
                original = target.read_bytes()
                target.write_bytes(original + b" ")
                with self.subTest(target=target.name), self.assertRaises(ValueError):
                    verify_bundle(data=data, prompt=prompt)
                target.write_bytes(original)

    def test_partial_formal_bundle_never_falls_back_to_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            shutil.copyfile(DATA / "reference_spec.json", data / "reference_spec.json")
            # Route both the consumer and verifier to the isolated partial fixture.
            with patch("evaluate_results.DATA", data), patch(
                    "freeze_rules.verify_bundle", side_effect=lambda: verify_bundle(data=data)):
                with self.assertRaises(FileNotFoundError):
                    load_context()

    def test_existing_freeze_rejected_without_modifying_seal(self):
        before = (DATA / "ruleset.json").read_bytes()
        with self.assertRaises(ValueError):
            freeze()
        self.assertEqual((DATA / "ruleset.json").read_bytes(), before)

    def test_preprocessing_whole_image_scaling_and_no_upscale(self):
        policy = read_json(DATA / "image_input_policy.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.png"
            for role, size, expected in (
                    ("REF-MAIN", (3072, 4096), (1536, 2048)),
                    ("CANDIDATE", (512, 512), (512, 512)),
                    ("CANDIDATE", (256, 128), (256, 128)),
                    ("CANDIDATE", (1024, 512), (512, 256))):
                with self.subTest(role=role, size=size):
                    Image.new("RGB", size, (11, 34, 57)).save(path)
                    before = path.read_bytes()
                    png, receipt = prepare_image(
                        path, role, policy, hashlib.sha256(before).hexdigest())
                    self.assertEqual(path.read_bytes(), before)
                    self.assertEqual(receipt["output_size"], list(expected))
                    with Image.open(io.BytesIO(png)) as output:
                        self.assertEqual(output.size, expected)
                        self.assertEqual(output.mode, "RGB")
                        self.assertEqual(output.getpixel((0, 0)), (11, 34, 57))
                    self.assertEqual(png, prepare_image(path, role, policy)[0])
            with self.assertRaises(ValueError):
                prepare_image(path, "CANDIDATE", policy, "wrong-hash")
            with self.assertRaises(ValueError):
                prepare_image(path, "UNKNOWN", policy)
            with self.assertRaises(ValueError):
                prepare_image(path, "CANDIDATE", {**policy, "crop": True})

    def test_exif_orientation_applied_and_original_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rotated.jpg"
            image = Image.new("RGB", (80, 40), (22, 33, 44))
            exif = image.getexif()
            exif[274] = 6
            image.save(path, exif=exif)
            before = text_digest(DATA / "image_input_policy.json")
            original = path.read_bytes()
            png, receipt = prepare_image(
                path, "CANDIDATE", read_json(DATA / "image_input_policy.json"))
            self.assertEqual(receipt["output_size"], [40, 80])
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(before, text_digest(DATA / "image_input_policy.json"))
            with Image.open(io.BytesIO(png)) as result:
                self.assertEqual(result.getexif().get(274, 1), 1)


if __name__ == "__main__":
    unittest.main()
