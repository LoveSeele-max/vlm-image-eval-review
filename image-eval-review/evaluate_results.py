"""C03: validate parsed labels and aggregate using trusted, sample-specific criteria.

No network, image scoring, persistence or raw API parsing is performed here.
"""

import json
from pathlib import Path

from build_manifest import DATA, ROOT, read_jsonl, text_digest, unique_index, write_json


AGGREGATION_ORDER = [
    "engineering_failure", "required_fail", "required_unjudgeable", "all_required_pass"
]
DIMENSIONS = ["subject_features", "structural_integrity", "prompt_adherence"]


def validate_policy(policy):
    """Refuse unsupported policy changes instead of silently changing decisions."""
    expected = {
        "active_label_statuses": ["pass", "fail", "unjudgeable"],
        "envelope_fields": ["sample_id", "labels"],
        "label_fields": [
            "criterion_id", "status", "candidate_region", "candidate_observation", "reason"],
        "evidence_fields": ["candidate_region", "candidate_observation", "reason"],
        "dimensions": DIMENSIONS,
        "aggregation_order": AGGREGATION_ORDER,
        "candidate_statuses": ["needs_review", "candidate_fail", "candidate_pass"],
        "engineering_statuses": ["task_not_succeeded", "invalid_result", "valid_result"],
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise ValueError(f"不支持的 C03 规则：{key}")
    if set(policy["labels"]) != {"pass", "fail", "unjudgeable", "not_applicable"}:
        raise ValueError("四类标签定义不完整。")


def expected_for_sample(sample_id, reference, evaluation, sample_rows):
    """Build requiredness/dimensions from rules, never from model labels."""
    sample = unique_index(sample_rows, "sample_id").get(sample_id)
    if sample is None:
        raise ValueError(f"未登记的样本：{sample_id}")
    structure = unique_index(evaluation["structure_criteria"], "id")
    prompts = unique_index(evaluation["prompt_criteria"], "id")
    if sample["structure_criterion_ids"] != list(structure):
        raise ValueError("样本结构条目与 C02 规范不一致。")
    expected = {}

    def add(row, dimension):
        key = row["id"]
        if not isinstance(key, str) or not key.strip() or key in expected:
            raise ValueError("条目 ID 为空或重复。")
        if type(row["required"]) is not bool:
            raise ValueError("条目必需性必须为布尔值。")
        expected[key] = {"dimension": dimension, "required": row["required"]}

    for row in reference["criteria"]:
        add(row, "subject_features")
    for key in sample["structure_criterion_ids"]:
        add(structure[key], "structural_integrity")
    excluded = {row["id"] for row in evaluation["excluded_prompt_requirements"]}
    for row in sample["prompt_criteria"]:
        key = row["id"]
        if key in excluded or key not in prompts or row["required"] != prompts[key]["required"]:
            raise ValueError("样本提示词条目与 C02 规范不一致。")
        add(row, "prompt_adherence")
    _validate_expected(expected)
    return expected


def _validate_expected(expected):
    if not isinstance(expected, dict) or not expected:
        raise ValueError("有效条目不能为空。")
    for key, row in expected.items():
        if (not isinstance(key, str) or not key.strip()
                or not isinstance(row, dict)
                or row.get("dimension") not in DIMENSIONS
                or type(row.get("required")) is not bool):
            raise ValueError("有效条目的编号、维度或必需性非法。")
    for dimension in DIMENSIONS:
        if not any(row["dimension"] == dimension and row["required"]
                   for row in expected.values()):
            raise ValueError(f"维度缺少必需条目：{dimension}")


def _aggregate(labels, expected, dimension=None):
    statuses = [
        labels[key]["status"] for key, row in expected.items()
        if row["required"] and (dimension is None or row["dimension"] == dimension)
    ]
    if "fail" in statuses:
        return "candidate_fail"
    if "unjudgeable" in statuses:
        return "needs_review"
    return "candidate_pass"


def evaluate_result(sample_id, result, expected, policy, task_status="succeeded"):
    """Return engineering errors separately; never fabricate visual fail labels."""
    validate_policy(policy)
    _validate_expected(expected)
    errors = []
    output = {
        "sample_id": sample_id,
        "policy_id": policy["policy_id"],
        "engineering_status": "valid_result",
        "candidate_status": "needs_review",
        "dimension_statuses": {key: "needs_review" for key in DIMENSIONS},
        "validation_errors": errors,
        "required_fail_ids": [],
        "required_unjudgeable_ids": [],
    }

    def error(code, criterion_id=None, field=None):
        item = {"code": code}
        if criterion_id is not None:
            item["criterion_id"] = criterion_id
        if field is not None:
            item["field"] = field
        errors.append(item)

    if task_status != "succeeded":
        error("task_not_succeeded")
        output["engineering_status"] = "task_not_succeeded"
        return output
    if not isinstance(result, dict) or set(result) != set(policy["envelope_fields"]):
        error("invalid_envelope")
        output["engineering_status"] = "invalid_result"
        return output
    if result["sample_id"] != sample_id:
        error("sample_id_mismatch")
    if not isinstance(result["labels"], list):
        error("invalid_labels")
        output["engineering_status"] = "invalid_result"
        return output
    labels = {}
    for row in result["labels"]:
        if not isinstance(row, dict):
            error("invalid_label")
            continue
        key = row.get("criterion_id")
        if not isinstance(key, str) or not key.strip():
            error("invalid_criterion_id")
            continue
        if key in labels:
            error("duplicate_criterion_id", key)
            continue
        labels[key] = row
        if key not in expected:
            error("unknown_criterion_id", key)
        if set(row) != set(policy["label_fields"]):
            error("invalid_label_fields", key)
        status = row.get("status")
        if status not in policy["active_label_statuses"]:
            error("not_applicable_forbidden" if status == "not_applicable"
                  else "invalid_status", key)
        for field in policy["evidence_fields"]:
            if not isinstance(row.get(field), str) or not row[field].strip():
                error("empty_evidence", key, field)
    for key in expected:
        if key not in labels:
            error("missing_criterion_id", key)
    if errors:
        output["engineering_status"] = "invalid_result"
        return output
    output["candidate_status"] = _aggregate(labels, expected)
    output["dimension_statuses"] = {
        key: _aggregate(labels, expected, key) for key in DIMENSIONS}
    for status, field in (("fail", "required_fail_ids"),
                          ("unjudgeable", "required_unjudgeable_ids")):
        output[field] = [
            key for key, row in expected.items()
            if row["required"] and labels[key]["status"] == status]
    return output


def load_context():
    policy_path = DATA / "label_policy.draft.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    validate_policy(policy)
    names = [
        "reference_spec.draft.json", "evaluation_spec.draft.json",
        "sample_criteria.draft.jsonl", "manifest.jsonl"]
    if set(policy["source_files_sha256"]) != set(names):
        raise ValueError("C03 来源文件绑定不完整。")
    for name in names:
        if text_digest(DATA / name) != policy["source_files_sha256"][name]:
            raise ValueError(f"C03 来源文件变化，需显式更新规则版本及绑定：{name}")
    reference = json.loads((DATA / names[0]).read_text(encoding="utf-8"))
    evaluation = json.loads((DATA / names[1]).read_text(encoding="utf-8"))
    samples = read_jsonl(DATA / names[2])
    manifest = read_jsonl(DATA / names[3])
    if set(unique_index(samples, "sample_id")) != set(unique_index(manifest, "sample_id")):
        raise ValueError("有效条目未恰好覆盖正式清单。")
    return policy, reference, evaluation, samples


def main():
    """Check metadata only, without creating image judgments."""
    policy, reference, evaluation, samples = load_context()
    expected = {
        row["sample_id"]: expected_for_sample(row["sample_id"], reference, evaluation, samples)
        for row in samples}
    sources = [DATA / name for name in policy["source_files_sha256"]]
    sources.extend([DATA / "label_policy.draft.json", Path(__file__).resolve()])
    summary = {
        "stage": "C03", "policy_id": policy["policy_id"], "status": policy["status"],
        "sample_count": len(expected),
        "label_assignment_count": sum(len(rows) for rows in expected.values()),
        "required_assignment_count": sum(row["required"] for rows in expected.values()
                                         for row in rows.values()),
        "auxiliary_assignment_count": sum(not row["required"] for rows in expected.values()
                                          for row in rows.values()),
        "smoke_label_counts": {key: len(expected[key])
                               for key in ("sample_020", "sample_029", "sample_042")},
        "inputs_sha256": {path.relative_to(ROOT).as_posix(): text_digest(path)
                          for path in sources},
        "checks": "40 条样本元数据及必需性核验；未产生图片标签",
        "api_calls": 0,
    }
    write_json(DATA / "label_policy.check.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
