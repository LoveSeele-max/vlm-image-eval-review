"""C04 rule bundle creation/verification; no API calls or image judgments."""

import argparse
import copy
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from build_criteria import build_rows
from build_manifest import DATA, ROOT, read_jsonl, text_digest, unique_index, write_json
from prepare_image import prepare_image


PROMPT = ROOT / "image-eval-review/prompts/judge_v1.txt"
VERSION = "rules_v1"
ARTIFACT_NAMES = (
    "reference_spec.json", "evaluation_spec.json", "sample_criteria.jsonl", "label_policy.json",
    "response_schema.json", "image_input_policy.json", "reference_readability.json",
    "reference_inputs.check.json",
)
SOURCE_NAMES = (
    "reference_spec.draft.json", "evaluation_spec.draft.json", "sample_criteria.draft.jsonl",
    "label_policy.draft.json", "manifest.jsonl", "split.json", "reference_assets.json",
    "prompt_requirements.json",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def response_schema(criterion_ids):
    return {
        "type": "object", "additionalProperties": False,
        "required": ["sample_id", "labels"],
        "properties": {
            "sample_id": {"type": "string"},
            "labels": {
                "type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["criterion_id", "status", "candidate_region",
                                 "candidate_observation", "reason"],
                    "properties": {
                        "criterion_id": {"type": "string", "enum": sorted(criterion_ids)},
                        "status": {"type": "string", "enum": ["pass", "fail", "unjudgeable"]},
                        **{key: {"type": "string"} for key in (
                            "candidate_region", "candidate_observation", "reason")},
                    },
                },
            },
        },
    }


def formalize(reference, evaluation, samples, policy, frozen_at):
    """Copy historical drafts; leave their bytes and boundaries unchanged."""
    reference, evaluation, samples, policy = copy.deepcopy(
        (reference, evaluation, samples, policy))
    for obj in (reference, evaluation, policy):
        obj.update(status="frozen", frozen_at=frozen_at, rule_version=VERSION)
    reference["spec_id"] = "subject_features_v1"
    reference["status_definitions"] = reference.pop("provisional_status_definitions")
    reference["label_vocabulary_status"] = "defined_in_label_policy.json"
    reference["aggregation_status"] = "defined_in_label_policy.json"
    reference["evidence_schema_status"] = "defined_in_label_policy.json"
    reference["evidence_fields"] = [
        "criterion_id", "status", "candidate_region", "candidate_observation", "reason"]
    reference.pop("evidence_fields_proposed")
    reference["applicability"]["auxiliary"] = (
        "辅助项须完整记录 pass/fail/unjudgeable；其状态不单独否决，漏填属于工程不完整。")
    reference["general_boundaries"] = [
        text.replace("提示词与结构维度待 C02", "提示词与结构维度见 evaluation_spec.json")
        for text in reference["general_boundaries"]]
    reference.pop("pre_freeze_checks")
    reference["freeze_evidence"] = "reference_readability.json；D04 仍需验证实际请求。"
    for row in reference["criteria"]:
        if "pre_freeze_review" in row:
            row.pop("pre_freeze_review")
            row["freeze_review"] = "C04 整图长边 2048 下疤痕可辨，维持必需；见 reference_readability.json。"
    evaluation["spec_id"] = "structure_prompt_v1"
    evaluation["general_boundaries"][-1] = "标签、证据字段及聚合顺序统一见 label_policy.json。"
    evaluation["evidence_fields"] = evaluation.pop("evidence_fields_proposed")
    evaluation.pop("freeze_checks_pending")
    evaluation["downstream_requirements"] = [
        "D01/D02 仅消费当前有效条目；原文中的木桌不评测。",
        "D04 核对实际请求图片与冻结预处理一致；本地核验不等于服务端验证。",
        "报告注明选定提示词条目符合度（不含木桌）。",
    ]
    for sample in samples:
        sample["criteria_status"] = "frozen"
    policy["basis"] = "C03 用户认可的规则；C04 完成参考本地可判性核查并保存正式规则包。"
    policy["source_files_sha256"] = {}  # Bound after the formal copies have been written.
    return reference, evaluation, samples, policy


def freeze():
    if (DATA / "ruleset.json").exists() or any(
            (DATA / name).exists() for name in (
                "reference_spec.json", "evaluation_spec.json", "sample_criteria.jsonl",
                "label_policy.json", "response_schema.json")):
        raise ValueError("正式文件已存在，拒绝覆盖；使用 --verify 或显式制定新版本。")
    from evaluate_results import load_context, expected_for_sample
    policy, reference, evaluation, samples = load_context(use_frozen=False)
    manifest = read_jsonl(DATA / "manifest.jsonl")
    rebuilt = build_rows(manifest, read_json(DATA / "prompt_requirements.json"), evaluation)
    if rebuilt != samples:
        raise ValueError("有效条目草稿不等于原始要求减去木桌。")
    if text_digest(DATA / "reference_assets.json") != reference["reference_assets_text_sha256"]:
        raise ValueError("参考图登记与 C01 绑定不一致。")
    image_policy = read_json(DATA / "image_input_policy.json")
    receipts = []
    private_dir = ROOT / "image-eval-review/private/C04"
    private_dir.mkdir(parents=True, exist_ok=True)
    for asset in read_json(DATA / "reference_assets.json"):
        png, receipt = prepare_image(
            Path(asset["source_path"]), asset["reference_id"], image_policy, asset["sha256"])
        (private_dir / f"{asset['reference_id']}_2048.png").write_bytes(png)
        receipts.append(receipt)
    frozen_at = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    reference, evaluation, samples, policy = formalize(
        reference, evaluation, samples, policy, frozen_at)
    all_ids = set()
    for sample in samples:
        all_ids.update(expected_for_sample(sample["sample_id"], reference, evaluation, samples))
    if "wooden_table" in all_ids:
        raise ValueError("木桌意外进入有效条目。")
    # Write the seal last. Partial output cannot be loaded as a frozen bundle.
    write_json(DATA / "reference_spec.json", reference)
    write_json(DATA / "evaluation_spec.json", evaluation)
    write_json(DATA / "sample_criteria.jsonl", samples, jsonl=True)
    policy["source_files_sha256"] = {
        name: text_digest(DATA / name) for name in (
            "reference_spec.json", "evaluation_spec.json", "sample_criteria.jsonl", "manifest.jsonl")}
    write_json(DATA / "label_policy.json", policy)
    write_json(DATA / "response_schema.json", response_schema(all_ids))
    write_json(DATA / "reference_inputs.check.json", {
        "rule_version": VERSION, "image_policy_id": image_policy["policy_id"],
        "receipts": receipts, "api_calls": 0,
        "scope": "本地参考输入生成与视觉检查；不证明服务端识别能力",
    })
    seal = {
        "schema_version": 1, "rule_version": VERSION, "status": "frozen", "frozen_at": frozen_at,
        "sample_count": len(samples), "active_criterion_types": len(all_ids),
        "sources_sha256": {name: text_digest(DATA / name) for name in SOURCE_NAMES},
        "artifacts_sha256": {name: text_digest(DATA / name) for name in ARTIFACT_NAMES},
        "judge_v1_sha256": text_digest(PROMPT),
        "code_sha256": {name: text_digest(ROOT / "image-eval-review" / name) for name in (
            "freeze_rules.py", "prepare_image.py", "evaluate_results.py")},
        "pending": ["D01 人工标注页", "D02 正式调用与严格解析",
                    "D03 运行记录", "D04 真实图片输入和接口参数验证"],
        "api_calls": 0, "image_labels_created": 0,
    }
    write_json(DATA / "ruleset.json", seal)
    return verify_bundle()


def verify_bundle(data=DATA, prompt=PROMPT, code_dir=None):
    """Read-only verification of all sealed sources, artifacts and consumers."""
    code_dir = code_dir or ROOT / "image-eval-review"
    seal = read_json(data / "ruleset.json")
    if seal.get("status") != "frozen" or seal.get("rule_version") != VERSION:
        raise ValueError("未知或未冻结规则包。")
    if set(seal["sources_sha256"]) != set(SOURCE_NAMES):
        raise ValueError("来源绑定不完整。")
    if set(seal["artifacts_sha256"]) != set(ARTIFACT_NAMES):
        raise ValueError("正式产物绑定不完整。")
    if set(seal["code_sha256"]) != {
            "freeze_rules.py", "prepare_image.py", "evaluate_results.py"}:
        raise ValueError("实现绑定不完整。")
    for category in ("sources_sha256", "artifacts_sha256"):
        for name, expected in seal[category].items():
            if text_digest(data / name) != expected:
                raise ValueError(f"规则包文件哈希变化：{name}")
    if text_digest(prompt) != seal["judge_v1_sha256"]:
        raise ValueError("judge_v1.txt 已变化，需要单独记录提示词版本。")
    for name, expected in seal["code_sha256"].items():
        if text_digest(code_dir / name) != expected:
            raise ValueError(f"实现已变化，需显式复核规则兼容性：{name}")
    from evaluate_results import expected_for_sample, validate_policy
    reference = read_json(data / "reference_spec.json")
    evaluation = read_json(data / "evaluation_spec.json")
    policy = read_json(data / "label_policy.json")
    samples = read_jsonl(data / "sample_criteria.jsonl")
    validate_policy(policy)
    for obj in (reference, evaluation, policy):
        if (obj["status"] != "frozen" or obj["rule_version"] != VERSION
                or obj["frozen_at"] != seal["frozen_at"]):
            raise ValueError("正式规则的版本或冻结状态不一致。")
    expected_bindings = {
        name: text_digest(data / name) for name in (
            "reference_spec.json", "evaluation_spec.json", "sample_criteria.jsonl", "manifest.jsonl")}
    if policy["source_files_sha256"] != expected_bindings:
        raise ValueError("正式标签政策的来源绑定不一致。")
    manifest = read_jsonl(data / "manifest.jsonl")
    if set(unique_index(samples, "sample_id")) != set(unique_index(manifest, "sample_id")):
        raise ValueError("正式条目未恰好覆盖样本清单。")
    inventories = {row["sample_id"]: expected_for_sample(
        row["sample_id"], reference, evaluation, samples) for row in samples}
    all_ids = {key for rows in inventories.values() for key in rows}
    if (read_json(data / "response_schema.json") != response_schema(all_ids)
            or "wooden_table" in all_ids
            or seal["sample_count"] != len(inventories)
            or seal["active_criterion_types"] != len(all_ids)):
        raise ValueError("正式条目、schema 或计数不一致。")
    return {
        "rule_version": VERSION, "status": "verified",
        "sample_count": len(inventories), "active_criterion_types": len(all_ids),
        "label_assignment_count": sum(map(len, inventories.values())),
        "required_assignment_count": sum(
            row["required"] for rows in inventories.values() for row in rows.values()),
        "smoke_label_counts": {key: len(inventories[key]) for key in (
            "sample_020", "sample_029", "sample_042")},
        "api_calls": 0,
    }


def load_frozen_context():
    verify_bundle()
    return (
        read_json(DATA / "label_policy.json"), read_json(DATA / "reference_spec.json"),
        read_json(DATA / "evaluation_spec.json"), read_jsonl(DATA / "sample_criteria.jsonl"))


def render_prompt(sample_id):
    policy, reference, evaluation, samples = load_frozen_context()
    sample = unique_index(samples, "sample_id").get(sample_id)
    if sample is None:
        raise ValueError("样本未登记。")
    manifest = unique_index(read_jsonl(DATA / "manifest.jsonl"), "sample_id")
    context = {
        "sample_id": sample_id, "original_prompt": manifest[sample_id]["prompt"],
        "reference_images": [{"reference_id": row["reference_id"], "role": row["role"]}
                             for row in reference["reference_images"]],
        "subject_boundaries": reference["general_boundaries"],
        "evaluation_boundaries": evaluation["general_boundaries"],
        "active_criteria": [
            *[dict(row, dimension="subject_features") for row in reference["criteria"]],
            *[dict(row, dimension="structural_integrity")
              for row in evaluation["structure_criteria"]],
            *[dict(row, dimension="prompt_adherence") for row in sample["prompt_criteria"]],
        ],
    }
    template = PROMPT.read_text(encoding="utf-8")
    if template.count("{{RULE_VERSION}}") != 1 or template.count("{{SAMPLE_CONTEXT_JSON}}") != 1:
        raise ValueError("V1 模板占位符不完整。")
    # Substitute data last, so raw prompt text can never alter the template.
    return template.replace("{{RULE_VERSION}}", policy["rule_version"]).replace(
        "{{SAMPLE_CONTEXT_JSON}}", json.dumps(context, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="只读核验已冻结规则包")
    args = parser.parse_args()
    print(json.dumps(verify_bundle() if args.verify else freeze(), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
