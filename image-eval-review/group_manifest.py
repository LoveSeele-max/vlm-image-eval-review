"""B03: exclude byte duplicates, screen similar images, and group without splitting."""

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import PIL
from PIL import Image, ImageDraw, ImageOps

from build_manifest import (
    DATA, ROOT, INSPECTIONS, SOURCE, build_rows, read_jsonl, text_digest,
    unique_index, validate_files, write_json,
)

POLICY = {
    "version": "B03_v1",
    "canonical": "prefer no duplicate pointer, then ascending A03 record ID",
    "grouping": "join same seed families, same condition groups, and reviewed same_group pairs",
    "fingerprints": "EXIF transpose; RGB; LANCZOS; dHash64 left>right; aHash64 pixel>=mean",
    "rgb_mae": "mean absolute byte difference of RGB 32x32, rounded to 3 decimals",
    "screen": {"dhash_max": 10, "ahash_max": 4, "ahash_rgb_mae_max": 30, "rgb_mae_max": 18},
    "cross_seed_review": "union of flagged pairs, top 5 dHash pairs, top 5 RGB MAE pairs",
    "near_duplicate_action": "review and group only; exclude only raw SHA-256 duplicates",
}
def object_digest(value):
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def load_inputs(data=DATA, root=ROOT):
    for name in ("manifest.jsonl", "split.json"):
        if (data / name).exists():
            raise ValueError("已存在 B04 正式清单或划分，拒绝重建 B03。")
    build = json.loads((data / "manifest.build.json").read_text(encoding="utf-8"))
    for relative, expected in build["inputs_sha256"].items():
        path = root / relative
        if not path.is_file() or text_digest(path) != expected:
            raise ValueError(f"B02 输入证据哈希已变化：{relative}")
    draft = read_jsonl(data / "manifest.draft.jsonl")
    sources = read_jsonl(data / "sample_sources.jsonl")
    inspections = read_jsonl(root / INSPECTIONS.name)
    validate_files(inspections)
    catalog = json.loads((data / "prompt_requirements.json").read_text(encoding="utf-8"))
    expected_draft, expected_sources, references = build_rows(
        read_jsonl(root / SOURCE.name), inspections, catalog, sources)
    actual_references = json.loads((data / "reference_assets.json").read_text(encoding="utf-8"))
    if (draft != expected_draft or sources != expected_sources or references != actual_references
            or build["sample_count"] != len(draft)):
        raise ValueError("B02 草稿或映射与原始证据重建结果不一致。")
    return draft, sources


def deduplicate(draft, sources):
    rows = unique_index(draft, "sample_id")
    mapping = unique_index(sources, "sample_id")
    unique_index(sources, "source_record_id")
    by_path = unique_index(sources, "source_path")
    if not rows or set(rows) != set(mapping):
        raise ValueError("候选与来源编号集合不一致或为空。")
    buckets = defaultdict(list)
    for sample_id, row in rows.items():
        if row["split"] is not None or row["group_id"] is not None or row["selection_status"] != "pending_B03":
            raise ValueError("输入必须是未分组、未划分的 B02 草稿。")
        source = mapping[sample_id]
        if type(source["seed"]) is not int:
            raise ValueError("seed 必须为整数。")
        pointer = source["byte_duplicate_candidate_of"]
        if pointer is not None:
            target = by_path.get(pointer)
            if target is None or rows[target["sample_id"]]["candidate_sha256"] != row["candidate_sha256"]:
                raise ValueError("字节重复指针缺失或哈希不匹配。")
        buckets[row["candidate_sha256"]].append(row)
    retained, exclusions = [], []
    semantic_fields = (
        "prompt", "requirements", "requirements_status", "reference_ids",
        "candidate_width", "candidate_height",
    )
    condition_fields = ("seed", "model_group", "generation_condition_group_id")
    for sha, members in sorted(buckets.items()):
        members.sort(key=lambda row: (
            bool(mapping[row["sample_id"]]["byte_duplicate_candidate_of"]),
            mapping[row["sample_id"]]["source_record_id"],
        ))
        canonical = members[0]
        retained.append(canonical)
        for duplicate in members[1:]:
            if (any(duplicate[key] != canonical[key] for key in semantic_fields)
                    or any(mapping[duplicate["sample_id"]][key] != mapping[canonical["sample_id"]][key]
                           for key in condition_fields)):
                raise ValueError("相同字节的提示词、要求或条件存在冲突，拒绝自动排除。")
            exclusions.append({
                "excluded_sample_id": duplicate["sample_id"],
                "retained_sample_id": canonical["sample_id"],
                "excluded_source_record_id": mapping[duplicate["sample_id"]]["source_record_id"],
                "retained_source_record_id": mapping[canonical["sample_id"]]["source_record_id"],
                "sha256": sha, "reason": "identical_raw_sha256",
            })
    return (sorted(retained, key=lambda row: row["sample_id"]),
            sorted(exclusions, key=lambda row: row["excluded_sample_id"]))


def fingerprint(path):
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        gray = image.convert("L")
        dh = gray.resize((9, 8), Image.Resampling.LANCZOS).tobytes()
        ah = gray.resize((8, 8), Image.Resampling.LANCZOS).tobytes()
        total = sum(ah)
        dhash = sum((dh[y * 9 + x] > dh[y * 9 + x + 1]) << (63 - y * 8 - x)
                    for y in range(8) for x in range(8))
        ahash = sum((pixel * 64 >= total) << (63 - index)
                    for index, pixel in enumerate(ah))
        rgb = image.resize((32, 32), Image.Resampling.LANCZOS).tobytes()
    return {"dhash64": f"{dhash:016x}", "ahash64": f"{ahash:016x}", "rgb": rgb}


def screen_pairs(retained, sources):
    mapping = unique_index(sources, "sample_id")
    fingerprints, private_rgb = [], {}
    for row in sorted(retained, key=lambda item: item["sample_id"]):
        value = fingerprint(Path(mapping[row["sample_id"]]["source_path"]))
        private_rgb[row["sample_id"]] = value.pop("rgb")
        fingerprints.append({
            "sample_id": row["sample_id"], "sha256": row["candidate_sha256"], **value,
        })
    pairs = []
    thresholds = POLICY["screen"]
    for left, right in itertools.combinations(fingerprints, 2):
        lid, rid = left["sample_id"], right["sample_id"]
        dh = (int(left["dhash64"], 16) ^ int(right["dhash64"], 16)).bit_count()
        ah = (int(left["ahash64"], 16) ^ int(right["ahash64"], 16)).bit_count()
        rgb = round(sum(abs(a - b) for a, b in zip(private_rgb[lid], private_rgb[rid])) / 3072, 3)
        pairs.append({
            "left": lid, "right": rid, "dhash_distance": dh, "ahash_distance": ah,
            "rgb_mae": rgb, "same_seed": mapping[lid]["seed"] == mapping[rid]["seed"],
            "screen_flag": (dh <= thresholds["dhash_max"] or rgb <= thresholds["rgb_mae_max"]
                            or (ah <= thresholds["ahash_max"] and rgb <= thresholds["ahash_rgb_mae_max"])),
        })
    return fingerprints, pairs


def review_targets(pairs):
    cross = [row for row in pairs if not row["same_seed"]]
    selected = [row for row in cross if row["screen_flag"]]
    for field in ("dhash_distance", "rgb_mae"):
        selected += sorted(cross, key=lambda row: (row[field], row["left"], row["right"]))[:5]
    return sorted({(row["left"], row["right"]) for row in selected})


def review_context(draft, sources, fingerprints, pairs):
    return object_digest({
        "draft": sorted(draft, key=lambda row: row["sample_id"]),
        "sources": sorted(sources, key=lambda row: row["sample_id"]),
        "policy": POLICY, "pillow_version": PIL.__version__,
        "fingerprints": fingerprints, "pairs": pairs,
    })


def validate_review(review, context_sha, targets, retained):
    if review.get("schema_version") != 1 or review.get("reviewer") in (None, "", "pending"):
        raise ValueError("复核版本或复核者记录不完整。")
    if review.get("context_sha256") != context_sha:
        raise ValueError("近重复复核上下文哈希过期。")
    rows = unique_index(retained, "sample_id")
    decisions = review["decisions"]
    keys = [(row["left"], row["right"]) for row in decisions]
    if len(set(keys)) != len(keys) or set(keys) != set(targets):
        raise ValueError("近重复复核名单缺失、重复或含未知图片对。")
    for item in decisions:
        if (item["decision"] not in ("distinct_images", "same_group") or not item.get("reason")
                or item.get("left_sha256") != rows[item["left"]]["candidate_sha256"]
                or item.get("right_sha256") != rows[item["right"]]["candidate_sha256"]):
            raise ValueError("近重复复核未完成或图片哈希过期。")
    return decisions


def group_rows(retained, sources, decisions):
    mapping = unique_index(sources, "sample_id")
    ids = sorted(row["sample_id"] for row in retained)
    parent = {sample_id: sample_id for sample_id in ids}

    def find(key):
        while key != parent[key]:
            key = parent[key]
        return key

    def join(left, right):
        a, b = sorted((find(left), find(right)))
        parent[b] = a

    for field in ("seed", "generation_condition_group_id"):
        representatives = {}
        for key in ids:
            value = mapping[key][field]
            if value in representatives:
                join(key, representatives[value])
            representatives[value] = key
    for item in decisions:
        if item["decision"] == "same_group":
            join(item["left"], item["right"])
    components = defaultdict(list)
    for key in ids:
        components[find(key)].append(key)
    groups, sample_groups = [], {}
    for index, members in enumerate(sorted(components.values()), 1):
        group_id = f"group_{index:03d}"
        sample_groups.update(dict.fromkeys(members, group_id))
        groups.append({
            "group_id": group_id, "sample_ids": members, "sample_count": len(members),
            "seeds": sorted({mapping[key]["seed"] for key in members}),
            "generation_condition_group_ids": sorted({
                mapping[key]["generation_condition_group_id"] for key in members}),
            "model_group_counts": dict(sorted(Counter(mapping[key]["model_group"] for key in members).items())),
        })
    grouped = [{**row, "group_id": sample_groups[row["sample_id"]],
                "selection_status": "included_B03"} for row in sorted(retained, key=lambda row: row["sample_id"])]
    # These are build-time invariants, not assumptions recorded as successful tests.
    for field in ("seed", "generation_condition_group_id"):
        values = defaultdict(set)
        for key in ids:
            values[mapping[key][field]].add(sample_groups[key])
        if any(len(value) != 1 for value in values.values()):
            raise ValueError("分组拆散了 seed 家族或同条件对照。")
    return grouped, groups


def prepare_review(sources, retained, pairs, context_sha):
    """Generate local evidence only; decisions must be filled after visual inspection."""
    folder = ROOT / "private/b03"
    folder.mkdir(parents=True, exist_ok=True)
    targets = review_targets(pairs)
    rows, mapping = unique_index(retained, "sample_id"), unique_index(sources, "sample_id")
    skeleton = {
        "schema_version": 1, "context_sha256": context_sha,
        "reviewer": "pending", "scope": "data grouping only; no evaluation labels",
        "decisions": [{
            "left": left, "right": right,
            "left_sha256": rows[left]["candidate_sha256"], "right_sha256": rows[right]["candidate_sha256"],
            "decision": "pending", "reason": "",
        } for left, right in targets],
    }
    write_json(folder / "review.pending.json", skeleton)
    # Keep previews local and ignored. No original images are changed.
    for name, tiles, columns in (
        ("cross_seed_review.jpg", [key for pair in targets for key in pair], 4),
        ("all_unique.jpg", [row["sample_id"] for row in sorted(
            retained, key=lambda row: (mapping[row["sample_id"]]["seed"],
                                       mapping[row["sample_id"]]["source_record_id"]))], 10),
    ):
        if not tiles:
            continue
        sheet = Image.new("RGB", (columns * 256, ((len(tiles) + columns - 1) // columns) * 278), "white")
        draw = ImageDraw.Draw(sheet)
        for index, key in enumerate(tiles):
            x, y = (index % columns) * 256, (index // columns) * 278
            with Image.open(mapping[key]["source_path"]) as image:
                thumb = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (248, 248))
                sheet.paste(thumb, (x + 4, y + 4))
            draw.text((x + 4, y + 252), f"{key} / seed {mapping[key]['seed']}", fill="black")
        sheet.save(folder / name)
    print(f"Local review evidence: {folder}; targets: {len(targets)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-review", action="store_true", help="write local preview and pending review; do not group")
    args = parser.parse_args()
    draft, sources = load_inputs()
    retained, exclusions = deduplicate(draft, sources)
    fingerprints, pairs = screen_pairs(retained, sources)
    context_sha = review_context(draft, sources, fingerprints, pairs)
    if args.prepare_review:
        prepare_review(sources, retained, pairs, context_sha)
        return
    review_path = DATA / "near_duplicate_review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    decisions = validate_review(review, context_sha, review_targets(pairs), retained)
    grouped, groups = group_rows(retained, sources, decisions)
    group_document = {
        "schema_version": 1, "stage": "B03_grouped", "policy": POLICY, "groups": groups,
        "split_status": "pending_B04",
        "limitation": "shared prompts across seed groups; no unseen-prompt generalization claim",
    }
    outputs = {
        "manifest.grouped.jsonl": (grouped, True), "exclusions.jsonl": (exclusions, True),
        "groups.json": (group_document, False), "image_fingerprints.jsonl": (fingerprints, True),
        "near_duplicate_pairs.jsonl": (pairs, True),
    }
    inputs = [
        DATA / name for name in ("manifest.draft.jsonl", "sample_sources.jsonl",
                                 "reference_assets.json", "manifest.build.json", "near_duplicate_review.json")
    ] + [Path(__file__).resolve()]
    for name, (value, jsonl) in outputs.items():
        write_json(DATA / name, value, jsonl=jsonl)
    summary = {
        "schema_version": 1, "stage": "B03_grouped", "policy": POLICY,
        "candidate_count": len(draft), "included_count": len(grouped), "excluded_count": len(exclusions),
        "group_count": len(groups), "group_sizes": [row["sample_count"] for row in groups],
        "pair_count": len(pairs), "screen_flag_count": sum(row["screen_flag"] for row in pairs),
        "cross_seed_flag_count": sum(row["screen_flag"] and not row["same_seed"] for row in pairs),
        "cross_seed_review_count": len(decisions),
        "confirmed_cross_seed_merge_count": sum(row["decision"] == "same_group" for row in decisions),
        "pillow_version": PIL.__version__, "review_context_sha256": context_sha,
        "inputs_sha256": {path.relative_to(ROOT).as_posix(): text_digest(path) for path in inputs},
        "outputs_sha256": {name: text_digest(DATA / name) for name in outputs},
        "split_status": "pending_B04", "requirements_status": "draft_pending_C02",
        "source_images_modified": False, "api_calls": 0,
    }
    write_json(DATA / "grouping.build.json", summary)
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("policy", "inputs_sha256", "outputs_sha256")}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
