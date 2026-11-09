"""F04 frozen V1/V2 request selection, local archives only; legacy V1 remains unchanged."""
import argparse
import copy
import json
import re
from pathlib import Path
from uuid import uuid4

import judge_client as client
from annotation_store import encode, now
from datetime import datetime
from build_manifest import DATA, ROOT, text_digest
from freeze_rules import PROMPT, render_prompt, verify_bundle


PROMPTS = PROMPT.parent
REGISTRY = DATA / "prompt_versions.json"
START = "【V2解释补充开始】"
END = "【V2解释补充结束】"
ENGINE_NAMES = ("freeze_rules.py", "run_store.py", "run_tasks.py", "judge_client.py", "prepare_image.py", "evaluate_results.py")
FIXED_DATA_NAMES = ("ruleset.json", "manifest.jsonl", "split.json", "reference_spec.json", "evaluation_spec.json",
                    "sample_criteria.jsonl", "label_policy.json", "response_schema.json", "reference_assets.json",
                    "image_input_policy.json", "human_reference_registration.json", "v2_change_plan.json")


def read_json(path):
    return client.strict_json(Path(path).read_bytes())


def appendix(v1, v2):
    if v2.count(START) != 1 or v2.count(END) != 1:
        raise ValueError("V2 解释补充标记不完整。")
    before, rest = v2.split(START)
    addition, after = rest.split(END)
    if before + after.lstrip("\n") != v1:
        # The addition replaces the existing final two-newline separator with itself plus a separator.
        if before + after.removeprefix("\n\n") != v1:
            raise ValueError("V2 修改了 V1 的基础指令，须复核解释性边界。")
    for text in (v1, v2):
        if text.count("{{RULE_VERSION}}") != 1 or text.count("{{SAMPLE_CONTEXT_JSON}}") != 1:
            raise ValueError("模板占位符数量不符。")
    if re.search(r"sample_\d+", addition):
        raise ValueError("V2 模板禁止包含具体开发样本编号或答案。")
    if any(len(re.findall(r"(?m)^" + item + r"\b", addition)) != 1 for item in ("R01", "R02", "R03", "R04", "R05", "R06")):
        raise ValueError("V2 未恰好覆盖六类解释修改。")
    return START + addition + END


def make_registry(profile, *, registry_path=REGISTRY, prompt_dir=PROMPTS):
    """One-time registration; profile is established from the verified D04/F01 baseline by the CLI."""
    registry_path, prompt_dir = Path(registry_path), Path(prompt_dir)
    if registry_path.exists():
        raise ValueError("提示词版本登记已存在，禁止覆盖；请使用 --verify。")
    verify_bundle()
    v1 = (prompt_dir / "judge_v1.txt").read_text(encoding="utf-8")
    v2 = (prompt_dir / "judge_v2.txt").read_text(encoding="utf-8")
    extra = appendix(v1, v2)
    plan = read_json(DATA / "v2_change_plan.json")
    if (plan["analysis_partition"] != "development" or plan["target_prompt_version"] != "judge_v2"
            or sum(r["disagreement_count"] for r in plan["changes"]) != plan["disagreement_item_count"]):
        raise ValueError("F03 修改理由来源不符。")
    human = read_json(DATA / "human_reference_registration.json")
    rules = read_json(DATA / "ruleset.json")
    seal = {"schema_version": 1, "status": "frozen", "frozen_at": now(),
            "rule_version": "rules_v1", "versions": {
                "judge_v1": {"template": "judge_v1.txt", "text_sha256": text_digest(prompt_dir / "judge_v1.txt"),
                             "frozen_at": rules["frozen_at"], "compatibility": "legacy_request_bytes_and_cache_identity_unchanged"},
                "judge_v2": {"template": "judge_v2.txt", "text_sha256": text_digest(prompt_dir / "judge_v2.txt"),
                             "changes": [r["id"] for r in plan["changes"]], "scope": "explanation_only_appendix"}},
            "appendix_sha256": client.digest(extra.encode("utf-8")),
            "fixed_data_text_sha256": {name: text_digest(DATA / name) for name in FIXED_DATA_NAMES},
            "legacy_engine_text_sha256": {name: text_digest(ROOT / "image-eval-review" / name) for name in ENGINE_NAMES},
            "version_builder_text_sha256": text_digest(Path(__file__)),
            "human_reference_version": human["reference_version"], "human_reference_seal_sha256": human["seal_sha256"],
            "comparison_profile": {"model": profile["model"], "endpoint_base_sha256": client.digest(profile["base_url"].encode("utf-8")),
                "max_output_tokens": profile["max_output_tokens"], "timeout_seconds": profile["timeout"],
                "max_attempts": profile["max_attempts"], "stream": False, "store": False},
            "source_stage": "F03_development_only", "v2_model_calls": 0,
            "limits": "Frozen instructions and local parity checks do not establish V2 quality improvement or backend identity."}
    seal["comparison_profile_sha256"] = client.digest(encode(seal["comparison_profile"]).encode("utf-8"))
    with registry_path.open("xb") as stream:
        stream.write(client.encoded(seal) + b"\n")
    return verify_registry(registry_path, prompt_dir=prompt_dir)


def verify_registry(registry_path=REGISTRY, *, prompt_dir=PROMPTS):
    verify_bundle()
    seal = read_json(registry_path)
    if seal["schema_version"] != 1 or seal["status"] != "frozen" or seal["rule_version"] != "rules_v1":
        raise ValueError("未知或未冻结提示词版本登记。")
    if set(seal["versions"]) != {"judge_v1", "judge_v2"}:
        raise ValueError("提示词版本集合不符。")
    if datetime.fromisoformat(seal["frozen_at"]).tzinfo is None:
        raise ValueError("冻结时间缺少时区。")
    if client.digest(encode(seal["comparison_profile"]).encode("utf-8")) != seal["comparison_profile_sha256"]:
        raise ValueError("模型或请求配置冻结哈希不符。")
    if set(seal["fixed_data_text_sha256"]) != set(FIXED_DATA_NAMES):
        raise ValueError("规则、Schema 或人工参考绑定不完整。")
    for version in ("judge_v1", "judge_v2"):
        row = seal["versions"][version]
        if row["template"] != version + ".txt" or text_digest(Path(prompt_dir) / row["template"]) != row["text_sha256"]:
            raise ValueError(f"提示词模板冻结哈希不符：{version}")
    extra = appendix((Path(prompt_dir) / "judge_v1.txt").read_text(encoding="utf-8"),
                     (Path(prompt_dir) / "judge_v2.txt").read_text(encoding="utf-8"))
    if client.digest(extra.encode("utf-8")) != seal["appendix_sha256"]:
        raise ValueError("V2 解释补充哈希不符。")
    for name, digest in seal["fixed_data_text_sha256"].items():
        if text_digest(DATA / name) != digest:
            raise ValueError(f"比较标准或人工参考绑定已改变：{name}")
    if set(seal["legacy_engine_text_sha256"]) != set(ENGINE_NAMES):
        raise ValueError("原执行代码绑定不完整。")
    for name, digest in seal["legacy_engine_text_sha256"].items():
        if text_digest(ROOT / "image-eval-review" / name) != digest:
            raise ValueError(f"原执行代码已改变：{name}")
    if text_digest(Path(__file__)) != seal["version_builder_text_sha256"]:
        raise ValueError("版本构建代码已改变，须显式复核。")
    return seal


def render_version(sample_id, prompt_version, *, seal=None):
    if prompt_version not in ("judge_v1", "judge_v2"):
        raise ValueError("必须显式选择 judge_v1 或 judge_v2。")
    if prompt_version == "judge_v1":
        return render_prompt(sample_id)
    seal = seal or verify_registry()
    v1 = PROMPT.read_text(encoding="utf-8")
    rendered = render_prompt(sample_id)
    prefix, suffix = v1.split("{{SAMPLE_CONTEXT_JSON}}")
    prefix = prefix.replace("{{RULE_VERSION}}", seal["rule_version"])
    if not rendered.startswith(prefix) or not rendered.endswith(suffix):
        raise ValueError("V1 渲染结构变化，无法复用相同上下文。")
    context = rendered[len(prefix):len(rendered) - len(suffix) if suffix else None]
    # Parse and reconstruct to verify this is the exact original context, never an injected instruction.
    if not isinstance(client.strict_json(context), dict):
        raise ValueError("规则上下文非法。")
    template = (PROMPTS / "judge_v2.txt").read_text(encoding="utf-8")
    return template.replace("{{RULE_VERSION}}", seal["rule_version"]).replace("{{SAMPLE_CONTEXT_JSON}}", context)


def build_versioned_request(sample_id, model, max_output_tokens=8192, *, prompt_version):
    if prompt_version not in ("judge_v1", "judge_v2"):
        raise ValueError("未知提示词版本。")
    seal = verify_registry()
    profile = seal["comparison_profile"]
    if model != profile["model"] or max_output_tokens != profile["max_output_tokens"]:
        raise ValueError("请求模型或输出上限与冻结比较配置不同。")
    prepared = client.build_request(sample_id, model, max_output_tokens)
    if prompt_version == "judge_v1":
        return prepared  # Exactly preserve original metadata/body and D03 V1 cache keys.
    result = copy.deepcopy(prepared)
    prompt = render_version(sample_id, prompt_version, seal=seal)
    result["body"]["input"][0]["content"][0]["text"] = prompt
    result["metadata"].update(prompt_version="judge_v2", prompt_sha256=client.digest(prompt.encode("utf-8")),
        request_body_sha256=client.digest(client.encoded(result["body"])),
        prompt_template_sha256=seal["versions"]["judge_v2"]["text_sha256"],
        prompt_registry_sha256=text_digest(REGISTRY), prompt_builder_source_sha256=text_digest(Path(__file__)))
    return result


def archive_request(prepared, *, base_url, output_root=client.PRIVATE / "versioned_inputs"):
    seal = verify_registry()
    if client.digest(base_url.encode("utf-8")) != seal["comparison_profile"]["endpoint_base_sha256"]:
        raise ValueError("接口与冻结比较配置不同。")
    version = prepared["metadata"]["prompt_version"]
    if version not in seal["versions"] or prepared["metadata"]["split"] != "development":
        raise ValueError("本地准备入口仅开放开发集，保留集留待 G01 门禁。")
    current = build_versioned_request(prepared["metadata"]["sample_id"], prepared["body"]["model"],
                                      prepared["body"]["max_output_tokens"], prompt_version=version)
    if current != prepared:
        raise ValueError("存档请求与冻结版本构建不一致。")
    folder = Path(output_root) / (version + "_" + prepared["metadata"]["sample_id"] + "_" + uuid4().hex[:12])
    if not folder.resolve().is_relative_to(client.PRIVATE.resolve()):
        raise ValueError("完整请求必须保存在 private 目录。")
    folder.mkdir(parents=True, exist_ok=False)
    with (folder / "prepared.json").open("xb") as stream:
        stream.write(client.encoded(prepared) + b"\n")
    with (folder / "receipt.json").open("xb") as stream:
        stream.write(client.encoded({"created_at": now(), "prompt_version": version,
            "sample_id": prepared["metadata"]["sample_id"], "request_body_sha256": prepared["metadata"]["request_body_sha256"],
            "registry_sha256": text_digest(REGISTRY), "api_calls": 0, "mode": "local_preparation_only"}) + b"\n")
    if read_json(folder / "prepared.json") != prepared:
        raise ValueError("请求存档读回复验失败。")
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--freeze", action="store_true")
    modes.add_argument("--verify", action="store_true")
    modes.add_argument("--prepare", action="store_true")
    parser.add_argument("--sample-id")
    parser.add_argument("--prompt-version", choices=("judge_v1", "judge_v2"))
    args = parser.parse_args()
    try:
        if args.freeze:
            from development_v1 import baseline_profile
            from review_app import ReadOnlyRuns
            from freeze_human_reference import load_registered_reference
            load_registered_reference()
            make_registry(baseline_profile(ReadOnlyRuns()))
            mode = "frozen"
        elif args.prepare:
            if not args.sample_id or not args.prompt_version:
                raise ValueError("准备请求必须显式提供样本与提示词版本。")
            settings = client.settings_from_environment()
            prepared = build_versioned_request(args.sample_id, settings["model"], prompt_version=args.prompt_version)
            folder = archive_request(prepared, base_url=settings["base_url"])
            print(json.dumps({"output": str(folder), "prompt_version": args.prompt_version, "api_calls": 0}, ensure_ascii=True))
            return
        else:
            mode = "verified"
        seal = verify_registry()
        print(json.dumps({"status": mode, "frozen_at": seal["frozen_at"], "versions": list(seal["versions"]),
                          "registry_sha256": text_digest(REGISTRY), "api_calls": 0}, ensure_ascii=True, indent=2))
    except (OSError, ValueError, KeyError, TypeError):
        parser.exit(1, "提示词版本操作失败：请核对冻结模板、规则、比较配置或准备参数。\n")


if __name__ == "__main__":
    main()
