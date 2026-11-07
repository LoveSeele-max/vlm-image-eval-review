"""Build C02 active criteria without rewriting original prompts or frozen samples."""

import copy
import json
from pathlib import Path

from build_manifest import DATA, ROOT, read_jsonl, text_digest, unique_index, write_json


def build_rows(manifest, catalog, spec):
    unique_index(manifest, "sample_id")
    prompt_index = unique_index(catalog["prompts"], "prompt")
    excluded = unique_index(spec["excluded_prompt_requirements"], "id")
    boundaries = unique_index(spec["prompt_criteria"], "id")
    structure = unique_index(spec["structure_criteria"], "id")
    catalog_ids = set(catalog["requirements"])
    if set(excluded) - catalog_ids or set(boundaries) != catalog_ids - set(excluded):
        raise ValueError("有效提示词条目与原始目录减去明确排除项不一致。")
    if set(structure) & catalog_ids:
        raise ValueError("结构与提示词条目编号冲突。")
    for row in [*boundaries.values(), *structure.values()]:
        if type(row["required"]) is not bool or not all(
                row.get(key) for key in ("pass_when", "fail_when", "unjudgeable_when")):
            raise ValueError("条目缺少必需性或判定边界。")
    result = []
    for sample in sorted(manifest, key=lambda row: row["sample_id"]):
        entry = prompt_index.get(sample["prompt"])
        if entry is None:
            raise ValueError("候选原始提示词未在来源目录登记。")
        expected = [{"id": key, **catalog["requirements"][key]} for key in entry["requirement_ids"]]
        if sample["requirements"] != expected:
            raise ValueError("候选原始要求与来源目录不一致。")
        active = []
        for requirement in sample["requirements"]:
            key = requirement["id"]
            if requirement["source_quote"] not in sample["prompt"]:
                raise ValueError("要求引文未出现在原始提示词中。")
            if key not in excluded:
                active.append({**copy.deepcopy(requirement), **copy.deepcopy(boundaries[key])})
        result.append({
            "sample_id": sample["sample_id"],
            "criteria_status": spec["status"],
            "structure_criterion_ids": list(structure),
            "prompt_criteria": active,
            "excluded_prompt_requirement_ids": [
                row["id"] for row in sample["requirements"] if row["id"] in excluded],
        })
    return result


def main():
    output = DATA / "sample_criteria.draft.jsonl"
    # C04 uses a new formal artifact. Do not overwrite a frozen rule set.
    if (DATA / "evaluation_spec.json").exists() or (DATA / "sample_criteria.jsonl").exists():
        raise ValueError("已存在正式规则或有效条目，拒绝覆盖草稿流程。")
    paths = [DATA / name for name in ("manifest.jsonl", "prompt_requirements.json",
                                     "evaluation_spec.draft.json")]
    rows = build_rows(read_jsonl(paths[0]), json.loads(paths[1].read_text(encoding="utf-8")),
                      json.loads(paths[2].read_text(encoding="utf-8")))
    write_json(output, rows, jsonl=True)
    summary = {
        "stage": "C02_draft", "sample_count": len(rows),
        "prompt_criterion_count": sum(len(row["prompt_criteria"]) for row in rows),
        "structure_assignment_count": sum(len(row["structure_criterion_ids"]) for row in rows),
        "excluded_prompt_assignment_count": sum(len(row["excluded_prompt_requirement_ids"]) for row in rows),
        "inputs_sha256": {path.relative_to(ROOT).as_posix(): text_digest(path)
                          for path in [*paths, Path(__file__).resolve()]},
        "outputs_sha256": {output.name: text_digest(output)},
        "status": "draft_pending_C03_C04", "api_calls": 0,
    }
    write_json(DATA / "criteria.build.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
