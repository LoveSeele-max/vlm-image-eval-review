"""F04 compatibility tests use synthetic pixels and local fixtures; no model calls."""
import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prompt_versions as pv
import judge_client as client
from run_store import identity
from test_judge_client import synthetic_prepare


class PromptVersionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.prompts = self.folder / "prompts";self.prompts.mkdir()
        for name in ("judge_v1.txt", "judge_v2.txt"):
            shutil.copyfile(pv.PROMPTS / name, self.prompts / name)
        self.registry = self.folder / "registry.json"
        self.profile = {"model": "synthetic-model", "base_url": "https://example.invalid/v1",
                        "max_output_tokens": 8192, "timeout": 180.0, "max_attempts": 3}
        self.seal = pv.make_registry(self.profile, registry_path=self.registry, prompt_dir=self.prompts)

    def build(self, version, sid="sample_020", **kwargs):
        with patch.object(pv, "verify_registry", return_value=self.seal), \
                patch.object(pv, "REGISTRY", self.registry), patch.object(pv, "PROMPTS", self.prompts), \
                patch.object(client, "prepare_image", side_effect=synthetic_prepare):
            return pv.build_versioned_request(sid, kwargs.get("model", "synthetic-model"),
                kwargs.get("max_output_tokens", 8192), prompt_version=version)

    def test_v1_body_metadata_and_cache_identity_exactly_preserved(self):
        with patch.object(client, "prepare_image", side_effect=synthetic_prepare):
            original = client.build_request("sample_020", "synthetic-model")
        selected = self.build("judge_v1")
        self.assertEqual(original, selected)
        self.assertEqual(identity(original, self.profile["base_url"],180),
                         identity(selected, self.profile["base_url"],180))

    def test_v2_changes_only_text_and_explicit_version_provenance(self):
        v1, v2 = self.build("judge_v1"), self.build("judge_v2")
        body = copy.deepcopy(v2["body"])
        body["input"][0]["content"][0]["text"] = v1["body"]["input"][0]["content"][0]["text"]
        self.assertEqual(body, v1["body"])
        self.assertEqual(v2["expected"], v1["expected"])
        self.assertEqual(v2["policy"], v1["policy"])
        changed = {k for k in set(v1["metadata"]) | set(v2["metadata"]) if v1["metadata"].get(k) != v2["metadata"].get(k)}
        self.assertEqual(changed, {"prompt_version","prompt_sha256","request_body_sha256","prompt_template_sha256",
                                   "prompt_registry_sha256","prompt_builder_source_sha256"})
        self.assertNotEqual(identity(v1,self.profile["base_url"],180),identity(v2,self.profile["base_url"],180))
        self.assertEqual(v2["metadata"]["request_body_sha256"],client.digest(client.encoded(v2["body"])))

    def test_explicit_version_and_frozen_comparison_config_enforced(self):
        for version in ("v2", "judge_v3", ""):
            with self.assertRaises(ValueError):self.build(version)
        for kwargs in ({"model":"changed-model"}, {"max_output_tokens":16384}):
            with self.assertRaisesRegex(ValueError,"冻结比较配置"):
                self.build("judge_v2",**kwargs)
        with self.assertRaises(TypeError):pv.build_versioned_request("sample_020","synthetic-model")

    def test_context_and_original_prompt_identical_in_both_versions(self):
        v1,v2=self.build("judge_v1","sample_008"),self.build("judge_v2","sample_008")
        marker="待核对材料及逐项规则：\n"
        a=client.strict_json(v1["body"]["input"][0]["content"][0]["text"].split(marker)[-1])
        b=client.strict_json(v2["body"]["input"][0]["content"][0]["text"].split(marker)[-1])
        self.assertEqual(a,b)
        self.assertEqual(a["sample_id"],"sample_008")
        self.assertEqual(len(a["active_criteria"]),25)

    def test_raw_prompt_like_placeholders_cannot_change_template(self):
        v1=pv.PROMPT.read_text(encoding="utf-8")
        context=json.dumps({"sample_id":"sample_020","original_prompt":"{{RULE_VERSION}} {{SAMPLE_CONTEXT_JSON}}"},ensure_ascii=False)
        original=v1.replace("{{RULE_VERSION}}","rules_v1").replace("{{SAMPLE_CONTEXT_JSON}}",context)
        with patch.object(pv,"render_prompt",return_value=original),patch.object(pv,"PROMPTS",self.prompts):
            rendered=pv.render_version("sample_020","judge_v2",seal=self.seal)
        self.assertTrue(rendered.endswith(context+"\n"))

    def test_appendix_preserves_foundation_and_rejects_answers_or_missing_sections(self):
        v1=(self.prompts/'judge_v1.txt').read_text(encoding='utf-8')
        v2=(self.prompts/'judge_v2.txt').read_text(encoding='utf-8')
        self.assertIn("R06",pv.appendix(v1,v2))
        for text in (v2.replace("只输出要求的 JSON 对象。","改成自由文本。",1),
                     v2.replace("R01 ","sample_008 R01 ",1),v2.replace("R03 ","X03 ",1),
                     v2.replace(pv.END,""),v2.replace("R01 对", "R01 {{RULE_VERSION}} 对",1)):
            with self.assertRaises(ValueError):pv.appendix(v1,text)

    def test_modified_v2_or_builder_or_missing_schema_binding_rejected(self):
        target=self.prompts/'judge_v2.txt';original=target.read_bytes();target.write_bytes(original+b'changed')
        with self.assertRaisesRegex(ValueError,"模板冻结哈希"):
            pv.verify_registry(self.registry,prompt_dir=self.prompts)
        target.write_bytes(original)
        registry=json.loads(self.registry.read_text(encoding='utf-8'))
        registry['version_builder_text_sha256']='changed'
        self.registry.write_text(json.dumps(registry),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,"版本构建代码"):
            pv.verify_registry(self.registry,prompt_dir=self.prompts)
        registry['version_builder_text_sha256']=pv.text_digest(Path(pv.__file__))
        registry['fixed_data_text_sha256'].pop('response_schema.json')
        self.registry.write_text(json.dumps(registry),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,"绑定不完整"):
            pv.verify_registry(self.registry,prompt_dir=self.prompts)

    def test_existing_registry_not_overwritten(self):
        before=self.registry.read_bytes()
        with self.assertRaisesRegex(ValueError,"禁止覆盖"):
            pv.make_registry(self.profile,registry_path=self.registry,prompt_dir=self.prompts)
        self.assertEqual(self.registry.read_bytes(),before)

    def test_changed_model_profile_or_frozen_time_refused(self):
        original=json.loads(self.registry.read_text(encoding='utf-8'))
        modified=copy.deepcopy(original);modified['comparison_profile']['model']='another-model'
        self.registry.write_text(json.dumps(modified),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'配置冻结哈希'):
            pv.verify_registry(self.registry,prompt_dir=self.prompts)
        modified=copy.deepcopy(original);modified['frozen_at']='2026-11-09T12:00:00'
        self.registry.write_text(json.dumps(modified),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'时区'):
            pv.verify_registry(self.registry,prompt_dir=self.prompts)

    def test_archives_are_private_distinct_and_readback_verified(self):
        prepared=self.build('judge_v2')
        with patch.object(pv,'verify_registry',return_value=self.seal),patch.object(pv,'REGISTRY',self.registry), \
                patch.object(pv.client,'PRIVATE',self.folder),patch.object(pv,'build_versioned_request',return_value=prepared):
            a=pv.archive_request(prepared,base_url=self.profile['base_url'],output_root=self.folder/'private_inputs')
            b=pv.archive_request(prepared,base_url=self.profile['base_url'],output_root=self.folder/'private_inputs')
            self.assertNotEqual(a,b)
            self.assertEqual(pv.read_json(a/'prepared.json'),prepared)
            self.assertEqual(pv.read_json(a/'receipt.json')['api_calls'],0)
            with self.assertRaisesRegex(ValueError,'接口'):
                pv.archive_request(prepared,base_url='https://other.invalid/v1',output_root=self.folder/'private_inputs')
            modified=copy.deepcopy(prepared);modified['body']['stream']=True
            with self.assertRaisesRegex(ValueError,'构建不一致'):
                pv.archive_request(modified,base_url=self.profile['base_url'],output_root=self.folder/'private_inputs')

    def test_holdout_prepare_archive_stays_locked(self):
        prepared=self.build('judge_v2','sample_001')
        with patch.object(pv,'verify_registry',return_value=self.seal):
            with self.assertRaisesRegex(ValueError,'仅开放开发集'):
                pv.archive_request(prepared,base_url=self.profile['base_url'],output_root=self.folder/'private_inputs')


if __name__=='__main__':unittest.main()
