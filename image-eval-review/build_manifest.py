"""Build a B02 draft and separate administrative metadata; never call a model."""

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"
SOURCE = ROOT / "A03_候选图片来源清单.jsonl"
INSPECTIONS = ROOT / "B01_图片检查清单.jsonl"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unique_index(rows, key):
    index = {}
    for row in rows:
        if row[key] in index:
            raise ValueError(f"重复的 {key}：{row[key]}")
        index[row[key]] = row
    return index


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_digest(path):
    # Git normalizes these tracked text inputs to LF, including Windows-generated B01 reports.
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def validate_files(inspections):
    for row in inspections:
        if row.get("status") != "passed" or not row.get("decode_ok") or not row.get("path_valid"):
            raise ValueError(f"B01 检查未通过：{row['inspection_id']}")
        path = Path(row["source_path"])
        if not path.is_file() or digest(path) != row["sha256"]:
            raise ValueError(f"文件缺失或与 B01 哈希不一致：{row['inspection_id']}")


def build_rows(sources, inspections, catalog, previous_mapping=None):
    source_index = unique_index(sources, "record_id")
    if not source_index:
        raise ValueError("来源清单为空。")
    unique_index(sources, "source_path")
    checked = unique_index(inspections, "inspection_id")
    if {row["inspection_id"] for row in inspections if row["role"] == "candidate"} != set(source_index):
        raise ValueError("A03 与 B01 的候选编号集合不一致。")
    prompts = {}
    aliases = {}
    for entry in catalog["prompts"]:
        if entry["prompt"] in prompts or len(set(entry["requirement_ids"])) != len(entry["requirement_ids"]):
            raise ValueError("提示词目录存在重复原文或重复要求。")
        for requirement_id in entry["requirement_ids"]:
            requirement = catalog["requirements"][requirement_id]
            if not requirement["source_quote"] or requirement["source_quote"] not in entry["prompt"]:
                raise ValueError("要求缺少原始提示词中的直接依据。")
        prompts[entry["prompt"]] = entry
        for alias in entry["aliases"]:
            if alias in aliases:
                raise ValueError("提示词别名重复。")
            aliases[alias] = entry["prompt"]
    if previous_mapping is not None:
        old = unique_index(previous_mapping, "source_record_id")
        unique_index(previous_mapping, "sample_id")
        if set(old) != set(source_index):
            raise ValueError("输入集合变化，不能覆盖已有匿名编号；请建立新的数据版本。")
        identifiers = {key: row["sample_id"] for key, row in old.items()}
    else:
        # Break the original batch/base/LoRA ordering without encoding conditions in IDs.
        ordered = sorted(source_index, key=lambda key: hashlib.sha256(
            ("vlm-image-eval-review/B02/v1/" + key).encode("utf-8")).hexdigest())
        identifiers = {key: f"sample_{index:03d}" for index, key in enumerate(ordered, 1)}
    reference_order = ("REF-MAIN", "REF-SUPPLEMENT")
    reference_roles = ("reference_main", "reference_supplement")
    references = []
    for key, role in zip(reference_order, reference_roles):
        reference = checked[key]
        if reference["role"] != role or reference["status"] != "passed":
            raise ValueError("参考图角色或 B01 状态不正确。")
        references.append({
            "reference_id": key, "role": role, "source_path": reference["source_path"],
            "sha256": reference["sha256"], "width": reference["width"], "height": reference["height"],
        })
    conditions = sorted({(row["prompt_text"], row["seed"]) for row in sources})
    condition_ids = {condition: f"condition_{index:03d}" for index, condition in enumerate(conditions, 1)}
    manifest, internal = [], []
    for source in sources:
        key = source["record_id"]
        image = checked[key]
        if (image["source_path"] != source["source_path"] or image["status"] != "passed"
                or not image.get("decode_ok")):
            raise ValueError(f"来源与 B01 检查不匹配：{key}")
        if (source["model_group"] not in ("base", "lora") or type(source["seed"]) is not int
                or source["provenance_status"] not in ("verified", "verified_duplicate_candidate")):
            raise ValueError(f"生成配置或来源核对状态不合法：{key}")
        prompt = source["prompt_text"]
        if prompt not in prompts or aliases.get(source["prompt_id"]) != prompt:
            raise ValueError(f"原始提示词或别名与已核对目录不符：{key}")
        entry = prompts[prompt]
        sample_id = identifiers[key]
        manifest.append({
            "schema_version": 1, "sample_id": sample_id,
            "candidate_asset_id": sample_id, "candidate_sha256": image["sha256"],
            "candidate_width": image["width"], "candidate_height": image["height"],
            "reference_ids": list(reference_order),
            "prompt": prompt,
            "requirements": [
                {"id": requirement_id, **catalog["requirements"][requirement_id]}
                for requirement_id in entry["requirement_ids"]
            ],
            "requirements_status": catalog["status"],
            "group_id": None, "split": None, "selection_status": "pending_B03",
        })
        internal.append({
            "sample_id": sample_id, "source_record_id": key,
            "source_path": source["source_path"], "file_name": source["file_name"],
            "batch": source["batch"], "evidence_id": source["evidence_id"],
            "provenance_status": source["provenance_status"], "prompt_id": source["prompt_id"],
            "model_group": source["model_group"], "seed": source["seed"],
            "generation_condition_group_id": condition_ids[(prompt, source["seed"])],
            "generation_profile_id": "historical_flux_klein_checkpoint240",
            "non_visual_tokens": entry["non_visual_tokens"],
            "byte_duplicate_candidate_of": source["byte_duplicate_candidate_of"],
        })
    return (sorted(manifest, key=lambda row: row["sample_id"]),
            sorted(internal, key=lambda row: row["sample_id"]), references)


def annotation_view(row):
    """Explicit allowlist for a future annotation UI; no paths/config/results."""
    return {key: row[key] for key in (
        "sample_id", "candidate_asset_id", "reference_ids", "prompt", "requirements", "requirements_status")}


def write_json(path, value, jsonl=False):
    if jsonl:
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in value)
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def main():
    if (DATA / "manifest.jsonl").exists() or (DATA / "split.json").exists():
        raise ValueError("已存在后续正式清单或划分，拒绝用 B02 草稿覆盖数据流程。")
    sources, inspections = read_jsonl(SOURCE), read_jsonl(INSPECTIONS)
    validate_files(inspections)
    catalog_path = DATA / "prompt_requirements.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    mapping_path = DATA / "sample_sources.jsonl"
    previous = read_jsonl(mapping_path) if mapping_path.exists() else None
    manifest, internal, references = build_rows(sources, inspections, catalog, previous)
    profile = json.loads((DATA / "generation_config.json").read_text(encoding="utf-8"))
    if profile["profile_id"] != "historical_flux_klein_checkpoint240":
        raise ValueError("内部历史配置编号不匹配。")
    summary = {
        "stage": "B02_draft", "schema_version": 1,
        "sample_count": len(manifest), "reference_count": len(references),
        "unique_prompt_count": len({row["prompt"] for row in manifest}),
        "generation_condition_group_count": len({row["generation_condition_group_id"] for row in internal}),
        "byte_unique_candidate_count": len({row["candidate_sha256"] for row in manifest}),
        "duplicate_candidate_count": sum(row["byte_duplicate_candidate_of"] is not None for row in internal),
        "requirement_count_distribution": dict(sorted(Counter(
            len(row["requirements"]) for row in manifest).items())),
        "input_hash_policy": "UTF-8 text with LF newlines; image hashes use raw bytes",
        "inputs_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): text_digest(path)
                          for path in (SOURCE, INSPECTIONS, catalog_path, DATA / "generation_config.json",
                                       Path(__file__).resolve())},
        "selection_status": "pending_B03", "grouping_status": "pending_B03",
        "split_status": "pending_B04", "requirements_status": catalog["status"],
        "annotation_view_fields": list(annotation_view(manifest[0])),
    }
    write_json(DATA / "manifest.draft.jsonl", manifest, jsonl=True)
    write_json(mapping_path, internal, jsonl=True)
    write_json(DATA / "reference_assets.json", references)
    write_json(DATA / "manifest.build.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
