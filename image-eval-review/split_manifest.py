"""Freeze a deterministic group split; the user chooses smoke samples separately."""

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from datetime import datetime
from fractions import Fraction
from pathlib import Path

from build_manifest import DATA, ROOT, read_jsonl, text_digest, unique_index, validate_files, write_json

NAMESPACE = "vlm-image-eval-review/B04/v1/"
POLICY = {
    "version": "B04_v1",
    "target_development_fraction": "1/3",
    "selection": "enumerate nonempty proper group subsets; minimize absolute fraction error",
    "tie_break": "ascending SHA-256(namespace + group_id), then group_id",
    "namespace": NAMESPACE,
    "smoke_selection": "user chooses exactly 3 distinct development IDs later",
}


def validate_partition_inputs(rows, groups):
    indexed = unique_index(rows, "sample_id")
    group_index = unique_index(groups, "group_id")
    if len(group_index) < 2 or len(group_index) > 16:
        raise ValueError("当前小规模划分支持 2–16 个完整组。")
    if len({row["candidate_sha256"] for row in rows}) != len(rows):
        raise ValueError("输入仍有字节重复。")
    members = {}
    for group in groups:
        if group["sample_count"] != len(group["sample_ids"]) or not group["sample_ids"]:
            raise ValueError("组成员计数不一致或为空。")
        for key in group["sample_ids"]:
            if key in members or key not in indexed:
                raise ValueError("组成员重复或未知。")
            members[key] = group["group_id"]
    if set(members) != set(indexed):
        raise ValueError("组成员未完整覆盖候选。")
    for key, row in indexed.items():
        if (row["group_id"] != members[key] or row["split"] is not None
                or row["selection_status"] != "included_B03"):
            raise ValueError("输入必须是组别一致、尚未划分的 B03 清单。")


def partition(rows, groups):
    validate_partition_inputs(rows, groups)
    groups = sorted(groups, key=lambda row: row["group_id"])
    scores = {
        row["group_id"]: hashlib.sha256((NAMESPACE + row["group_id"]).encode("utf-8")).hexdigest()
        for row in groups
    }
    candidates = []
    for count in range(1, len(groups)):
        for subset in itertools.combinations(groups, count):
            size = sum(group["sample_count"] for group in subset)
            ranking = tuple(sorted((scores[group["group_id"]], group["group_id"]) for group in subset))
            candidates.append((abs(Fraction(size, len(rows)) - Fraction(1, 3)), ranking, subset))
    chosen = min(candidates, key=lambda item: (item[0], item[1]))[2]
    development_groups = {group["group_id"] for group in chosen}
    manifest = [{**row, "split": "development" if row["group_id"] in development_groups else "holdout"}
                for row in sorted(rows, key=lambda row: row["sample_id"])]
    sides = {}
    for side in ("development", "holdout"):
        members = [row for row in manifest if row["split"] == side]
        sides[side] = {
            "group_ids": sorted({row["group_id"] for row in members}),
            "sample_ids": [row["sample_id"] for row in members],
            "sample_count": len(members), "fraction": f"{len(members)}/{len(rows)}",
        }
    return manifest, sides, scores


def load_verified_inputs(data=DATA, root=ROOT):
    bindings = {}
    for filename in ("manifest.build.json", "grouping.build.json"):
        path = data / filename
        build = json.loads(path.read_text(encoding="utf-8"))
        bindings[path.relative_to(root).as_posix()] = text_digest(path)
        for relative, expected in build["inputs_sha256"].items():
            target = root / relative
            if not target.is_file() or text_digest(target) != expected:
                raise ValueError(f"上游输入哈希不一致：{relative}")
            bindings[relative] = expected
        for name, expected in build.get("outputs_sha256", {}).items():
            target = data / name
            if not target.is_file() or text_digest(target) != expected:
                raise ValueError(f"B03 产物哈希不一致：{name}")
            bindings[target.relative_to(root).as_posix()] = expected
    validate_files(read_jsonl(root / "B01_图片检查清单.jsonl"))
    rows = read_jsonl(data / "manifest.grouped.jsonl")
    groups = json.loads((data / "groups.json").read_text(encoding="utf-8"))["groups"]
    validate_partition_inputs(rows, groups)
    mapping = unique_index(read_jsonl(data / "sample_sources.jsonl"), "sample_id")
    # Validate the relationships relevant to leakage, independently of the B03 summary.
    for field in ("seed", "generation_condition_group_id"):
        assigned = defaultdict(set)
        for row in rows:
            assigned[mapping[row["sample_id"]][field]].add(row["group_id"])
        if any(len(value) != 1 for value in assigned.values()):
            raise ValueError("上游分组拆散了 seed 家族或同条件对照。")
    script = Path(__file__).resolve()
    bindings[script.relative_to(ROOT).as_posix()] = text_digest(script)
    return rows, groups, bindings


def validate_smoke_selection(selection, development_ids):
    ids = selection["sample_ids"]
    if selection["status"] == "awaiting_user":
        if ids or selection["selected_by"] is not None:
            raise ValueError("待用户指定的联调名单必须为空。")
    elif selection["status"] == "selected":
        if (len(ids) != 3 or len(set(ids)) != 3 or not set(ids) <= set(development_ids)
                or selection["selected_by"] != "user"):
            raise ValueError("联调名单必须由用户指定 3 条不同开发集样本。")
    else:
        raise ValueError("未知联调指定状态。")


def freeze(data, rows, groups, bindings):
    manifest, sides, scores = partition(rows, groups)
    manifest_path, split_path = data / "manifest.jsonl", data / "split.json"
    if manifest_path.exists() != split_path.exists():
        raise ValueError("冻结文件不完整，拒绝自动覆盖；请先核对中断原因。")
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if (split["policy"] != POLICY or split["inputs_sha256"] != bindings
                or split["partitions"] != sides or split["group_rank_sha256"] != scores
                or split["manifest_sha256"] != text_digest(manifest_path)
                or read_jsonl(manifest_path) != manifest or split["split_status"] != "frozen"):
            raise ValueError("冻结清单、划分或证据发生变化，拒绝覆盖。")
        validate_smoke_selection(split["smoke_selection"], sides["development"]["sample_ids"])
        return split, False
    split = {
        "schema_version": 1, "split_version": "seed_family_v1", "split_status": "frozen",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "policy": POLICY, "partitions": sides, "group_rank_sha256": scores,
        "inputs_sha256": bindings,
        "smoke_selection": {"status": "awaiting_user", "sample_ids": [], "selected_by": None},
        "requirements_status": "draft_pending_C02",
        "limitation": "single subject; shared prompt templates; only four seed families",
    }
    write_json(manifest_path, manifest, jsonl=True)
    split["manifest_sha256"] = text_digest(manifest_path)
    write_json(split_path, split)
    return split, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify existing frozen files without creating them")
    args = parser.parse_args()
    if args.verify and not (DATA / "split.json").exists():
        raise ValueError("尚无冻结划分可验证。")
    rows, groups, bindings = load_verified_inputs()
    split, created = freeze(DATA, rows, groups, bindings)
    print(json.dumps({
        "created": created, "split_status": split["split_status"],
        "partitions": split["partitions"], "smoke_selection": split["smoke_selection"],
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
