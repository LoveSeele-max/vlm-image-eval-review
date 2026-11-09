"""E03: count verified initial author labels; never query model results or change labels."""

import argparse
import csv
import io
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from annotation_store import PRIVATE, encode, now
from audit_annotations import require, sha, strict_json, verify_export, write_json
from build_manifest import DATA, text_digest
from freeze_rules import load_frozen_context


STATUSES = ("pass", "fail", "unjudgeable", "not_applicable")
SCOPES = {"development": "开发集", "holdout": "保留集", "all": "全量"}
DIMENSIONS = {"subject_features": "主体特征", "structural_integrity": "结构完整性",
              "prompt_adherence": "选定提示词条目符合度（不含木桌）"}
FILES = {"statistics.json", "标签统计说明.md", "overview.csv", "dimensions.csv", "criteria.csv"}


def read_source(folder):
    folder = Path(folder).resolve()
    verify_export(folder)
    fingerprints = {p.name: sha(p.read_bytes()) for p in folder.iterdir()}
    context = strict_json((folder / "context.json").read_text(encoding="utf-8"))
    rows = [strict_json(line) for line in (folder / "initial_labels.jsonl").read_text(
        encoding="utf-8").splitlines()]
    return context, rows, fingerprints


def criterion_catalog(context):
    _, reference, evaluation, _ = load_frozen_context()
    require(text_digest(DATA / "ruleset.json") == context["ruleset_sha256"],
            "当前规则与 E02 初始快照绑定不同。")
    catalog = {}
    for dimension, rules in (("subject_features", reference["criteria"]),
                             ("structural_integrity", evaluation["structure_criteria"]),
                             ("prompt_adherence", evaluation["prompt_criteria"])):
        for rule in rules:
            require(rule["id"] not in catalog, "规则条目 ID 重复。")
            catalog[rule["id"]] = {"name": rule.get("name", rule.get("description", rule["id"])),
                                  "dimension": dimension, "required": rule["required"]}
    return catalog


def summarize(rows, context, catalog):
    """Count assignments, not independent images. Zero denominators stay explicit."""
    expected, manifest = context["expected"], context["manifest"]
    pairs = set()
    for row in rows:
        sid = row["sample_id"]
        label = row["record"]["label"]
        cid = label["criterion_id"]
        require(sid in expected and cid in expected[sid], f"未知分配：{sid}/{cid}")
        require((sid, cid) not in pairs, f"重复分配：{sid}/{cid}")
        pairs.add((sid, cid))
        rule = expected[sid][cid]
        require(row["split"] == manifest[sid]["split"] and row["split"] in ("development", "holdout"),
                f"划分错配：{sid}/{cid}")
        require(row["dimension"] == rule["dimension"] and row["required"] is rule["required"],
                f"条目维度或必需性错配：{sid}/{cid}")
        require(label["status"] in STATUSES[:3], f"有效条目出现非法或不适用状态：{sid}/{cid}")
        require(cid in catalog and catalog[cid]["dimension"] == rule["dimension"]
                and catalog[cid]["required"] is rule["required"], f"条目目录错配：{cid}")
        require(row["rule_version"] == context["rule_version"], "规则版本错配。")
    require(pairs == {(sid, cid) for sid, rules in expected.items() for cid in rules},
            "初始标签未完整覆盖有效条目。")
    active_ids = sorted({cid for rules in expected.values() for cid in rules})
    overview, dimensions, criteria, sources = [], [], [], []

    def count(selected, scope, dimension="all", cid=None):
        counts = Counter(r["record"]["label"]["status"] for r in selected)
        people = {sid for sid, row in manifest.items() if scope == "all" or row["split"] == scope}
        result = {"scope": scope, "scope_name": SCOPES[scope], "dimension": dimension,
                  "dimension_name": DIMENSIONS.get(dimension, "全部维度"),
                  "sample_count": len(people), "assigned_image_count": len({r["sample_id"] for r in selected}),
                  "assigned_items": len(selected), "required_items": sum(r["required"] for r in selected),
                  "auxiliary_items": sum(not r["required"] for r in selected),
                  **{status: counts[status] for status in STATUSES},
                  "decidable_items": counts["pass"] + counts["fail"],
                  "human_fail_denominator": counts["fail"], "human_pass_denominator": counts["pass"]}
        limits = []
        if not selected:
            limits.append("未发放，本范围不计算本条指标")
        else:
            if counts["fail"] == 0:
                limits.append("没有人工不符例，错误通过比例不可计算")
            if counts["pass"] == 0:
                limits.append("没有人工符合例，错误拒绝比例不可计算")
            if result["decidable_items"] == 0:
                limits.append("人工均不可判断，决策覆盖率不可计算")
        result["metric_limits"] = "；".join(limits) or "人工分母非零，待模型结果后计算"
        if cid is not None:
            result = {"criterion_id": cid, "criterion_name": catalog[cid]["name"],
                      "required": catalog[cid]["required"], **result,
                      "not_assigned_images": len(people) - len(selected),
                      "no_pass": counts["pass"] == 0, "no_fail": counts["fail"] == 0}
        require(sum(result[s] for s in STATUSES) == result["assigned_items"], "状态合计不符。")
        return result

    for scope in SCOPES:
        selected = [r for r in rows if scope == "all" or r["split"] == scope]
        overview.append(count(selected, scope))
        for dimension in DIMENSIONS:
            dimensions.append(count([r for r in selected if r["dimension"] == dimension], scope, dimension))
        for cid in active_ids:
            subset = [r for r in selected if r["record"]["label"]["criterion_id"] == cid]
            criteria.append(count(subset, scope, catalog[cid]["dimension"], cid))
        sources.append({"scope": scope, "origin_counts": dict(Counter(
            r["record"]["origin"]["kind"] for r in selected)),
            "explicit_chat_assisted_items": sum(r["record"]["origin"].get("decision_source") ==
                                                 "explicit_user_chat_instruction" for r in selected)})
    for scope in SCOPES:
        total = next(r for r in overview if r["scope"] == scope)
        for key in ("assigned_items", "required_items", "auxiliary_items", *STATUSES):
            require(total[key] == sum(r[key] for r in dimensions if r["scope"] == scope), "维度合计不符。")
            require(total[key] == sum(r[key] for r in criteria if r["scope"] == scope), "条目合计不符。")
    for key in ("sample_count", "assigned_items", "required_items", "auxiliary_items", *STATUSES):
        require(overview[2][key] == overview[0][key] + overview[1][key], "划分合计不符。")
    return {"overview": overview, "dimensions": dimensions, "criteria": criteria,
            "sources": sources, "active_criterion_count": len(active_ids)}


def csv_text(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\r\n")
    writer.writeheader()
    writer.writerows([{k: str(v).lower() if type(v) is bool else v for k, v in r.items()} for r in rows])
    return buffer.getvalue()


def markdown(report):
    data = report["data"]
    lines = ["# 初始人工标签分布", "", f"统计时间：{report['created_at']}", "",
             "来源：已核验的 E02 初始快照。单位为条目分配次数，40 张图片内的多个条目不是独立图片样本。",
             "", "## 总体与维度", "",
             "| 范围 | 维度 | 图片数 | 发放项 | 符合 | 不符 | 不可判断 | 不适用 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in data["overview"] + data["dimensions"]:
        lines.append(f"| {row['scope_name']} | {row['dimension_name']} | {row['sample_count']} | "
                     f"{row['assigned_items']} | {row['pass']} | {row['fail']} | {row['unjudgeable']} | {row['not_applicable']} |")
    lines += ["", "图片数为该划分总图片数，各维度行不能相加得到独立图片数。",
              "", "## 具体条目", "",
              "未发放的提示词要求仅计入未发放图片数，不填成符合、不符、不可判断或不适用。木桌已排除。",
              "", "| 范围 | 条目 | 必需性 | 发放项 | 未发放图片 | 符合 | 不符 | 不可判断 | 指标限制 |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in data["criteria"]:
        lines.append(f"| {r['scope_name']} | {r['criterion_name']}（{r['criterion_id']}） | "
                     f"{'必需' if r['required'] else '辅助'} | {r['assigned_items']} | {r['not_assigned_images']} | "
                     f"{r['pass']} | {r['fail']} | {r['unjudgeable']} | {r['metric_limits']} |")
    lines += ["", "## 口径及来源限制", "",
              "- 符合（pass）称正例，不符（fail）称负例；这是作者参考判断，不是客观真值。",
              "- 决策覆盖率的人工分母为 pass+fail，错误通过比例的分母为人工 fail，错误拒绝比例的分母为人工 pass。分母为零标不可计算，不能填 0% 或 100%。",
              "- 条件一致率还依赖双方均可判断的模型结果，本阶段不计算，也不因人工分母非零就声称可计算。",
              "- unjudgeable 单独保留，不改成 fail，不静默排除。辅助项计入分布，必需性单独列出，不改变聚合标准。",
              "- 来源含作者在页面填写、作者明确指定的批量判断及助手复制或整理的文字，不称全量逐项独立撰写。",
              "- 单主体、共享提示词模板及仅四个 seed 家族限制保持不变，不依据分布补样、更换划分或修改标签。",
              "- 没有读取模型结果、调用 API 或执行 E04 整体人工参考冻结。", "", "| 范围 | 初始来源 | 项数 |",
              "| --- | --- | ---: |"]
    for r in data["sources"]:
        for kind, number in r["origin_counts"].items():
            lines.append(f"| {SCOPES[r['scope']]} | {kind} | {number} |")
        lines.append(f"| {SCOPES[r['scope']]} | 其中本次聊天明确指示的辅助填写 | {r['explicit_chat_assisted_items']} |")
    lines += ["", "上表“其中”是来源的子集，不能与其他行相加。CSV 列名使用稳定字段名，统计含义以本说明为准。", ""]
    return "\n".join(lines)


def verify_statistics(folder):
    folder = Path(folder)
    hashes = strict_json((folder / "checksums.json").read_text(encoding="utf-8"))
    require(set(hashes) == FILES, "统计文件清单不完整。")
    require({p.name for p in folder.iterdir()} == FILES | {"checksums.json"}, "统计目录缺失或含额外文件。")
    for name, value in hashes.items():
        require(sha((folder / name).read_bytes()) == value, f"统计文件哈希不符：{name}")
    report = strict_json((folder / "statistics.json").read_text(encoding="utf-8"))
    context, rows, source_hashes = read_source(report["source_export"])
    require(source_hashes == report["source_files_sha256"], "E02 来源文件变化。")
    require(summarize(rows, context, report["criterion_catalog"]) == report["data"], "统计与初始快照不一致。")
    for name in ("overview", "dimensions", "criteria"):
        require((folder / (name + ".csv")).read_bytes() == ("\ufeff" + csv_text(report["data"][name])).encode("utf-8"),
                f"{name}.csv 与统计不一致。")
    require((folder / "标签统计说明.md").read_text(encoding="utf-8") == markdown(report), "说明与统计不一致。")
    require(report["global_reference_frozen"] is False, "E03 不执行整体冻结。")
    return {"status": "verified_statistics", "overview": report["data"]["overview"],
            "criterion_rows": len(report["data"]["criteria"]), "api_calls": 0}


def generate_statistics(source_export, output=None):
    source_export = Path(source_export).resolve()
    context, rows, fingerprints = read_source(source_export)
    catalog = criterion_catalog(context)
    data = summarize(rows, context, catalog)
    folder = Path(output).resolve() if output else PRIVATE / "label_statistics" / (
        datetime.fromisoformat(now()).strftime("e03_%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8])
    folder = folder.resolve()
    require(folder.is_relative_to(PRIVATE.resolve()) and folder != PRIVATE.resolve(), "统计只能写入 private 子目录。")
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir(exist_ok=False)
    report = {"task": "E03", "schema_version": 1, "created_at": now(),
              "source_export": str(source_export), "source_files_sha256": fingerprints,
              "initial_only": True, "global_reference_frozen": False, "api_calls": 0,
              "generator_text_sha256": text_digest(Path(__file__)),
              "criterion_catalog": catalog, "data": data}
    write_json(folder / "statistics.json", report)
    for name in ("overview", "dimensions", "criteria"):
        with (folder / (name + ".csv")).open("x", encoding="utf-8-sig", newline="") as stream:
            stream.write(csv_text(data[name]))
    with (folder / "标签统计说明.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(markdown(report))
    require(read_source(source_export)[2] == fingerprints, "统计期间 E02 导出发生变化。")
    write_json(folder / "checksums.json", {name: sha((folder / name).read_bytes()) for name in sorted(FILES)})
    return {"output": str(folder), **verify_statistics(folder)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_export", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    try:
        if args.verify:
            require(not args.source_export and not args.output, "复验时不指定新输入或输出。")
            result = verify_statistics(args.verify)
        else:
            require(args.source_export is not None, "需要指定已核验的 E02 导出目录。")
            result = generate_statistics(args.source_export, args.output)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f"E03 统计失败：{exc}\n")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
