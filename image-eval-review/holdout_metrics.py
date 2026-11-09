"""G02: read-only, frozen-reference holdout metrics; no sending or label revision."""
import argparse
import csv
import io
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import judge_client as client
import holdout_runs as runs
from annotation_store import now
from audit_annotations import require
from build_manifest import DATA, text_digest
from evaluate_results import evaluate_result
from freeze_human_reference import load_registered_reference
from freeze_rules import load_frozen_context
from label_statistics import DIMENSIONS
from review_app import ReadOnlyRuns


VERSIONS = runs.VERSIONS
HUMAN = ("pass", "fail", "unjudgeable", "not_applicable")
AI = ("pass", "fail", "unjudgeable", "engineering_missing")
METRICS = ("decision_coverage", "conditional_agreement", "false_pass", "false_rejection")
FILES = {"metrics.json", "items.jsonl", "dimensions.csv", "overview.csv", "criteria.csv",
         "requiredness.csv", "paired.csv", "保留集主要指标.md"}


class MetricRuns(runs.HoldoutStore):
    """Reuse G01 binding/evidence readers while refusing every database write."""
    __init__ = ReadOnlyRuns.__init__
    connect = ReadOnlyRuns.connect


def ratio(numerator, denominator):
    require(type(numerator) is int and type(denominator) is int and 0 <= numerator <= denominator,
            "指标分子分母非法。")
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None,
            "status": "computed" if denominator else "zero_denominator"}


def display(value):
    n, d = value["numerator"], value["denominator"]
    return f"{n}/{d} = {100*n/d:.2f}%" if d else f"不可计算（{n}/{d}，分母为零）"


def counts(rows):
    """All count units are assigned items, not independent images."""
    matrix = Counter((r["human_status"], r["ai_status"] if r["ai_status"] is not None else "engineering_missing") for r in rows)
    require(set(matrix) <= {(h, a) for h in HUMAN for a in AI}, "未知统计状态。")
    h = Counter({s: sum(matrix[s, a] for a in AI) for s in HUMAN})
    a = Counter({s: sum(matrix[t, s] for t in HUMAN) for s in AI})
    binary = sum(matrix[h, a] for h in HUMAN[:2] for a in AI[:2])
    agreement = matrix["pass", "pass"] + matrix["fail", "fail"]
    result = {"assigned_items": len(rows), "assigned_images": len({r["sample_id"] for r in rows}),
        "required_items": sum(r["required"] for r in rows),
        "auxiliary_items": sum(not r["required"] for r in rows),
        **{"human_"+s: h[s] for s in HUMAN}, **{"ai_"+s: a[s] for s in AI},
        "human_binary_items": h["pass"] + h["fail"], "both_binary_items": binary,
        "agreements": agreement, "false_pass_items": matrix["fail", "pass"],
        "false_rejection_items": matrix["pass", "fail"],
        "human_binary_ai_unjudgeable": sum(matrix[h, "unjudgeable"] for h in HUMAN[:2]),
        "human_binary_engineering_missing": sum(matrix[h, "engineering_missing"] for h in HUMAN[:2]),
        "matrix": {h: {a: matrix[h, a] for a in AI} for h in HUMAN},
        "decision_coverage": ratio(binary, h["pass"] + h["fail"]),
        "conditional_agreement": ratio(agreement, binary),
        "false_pass": ratio(matrix["fail", "pass"], h["fail"]),
        "false_rejection": ratio(matrix["pass", "fail"], h["pass"])}
    require(sum(h.values()) == len(rows) == sum(a.values()), "状态合计不符。")
    require(result["human_binary_items"] == binary + result["human_binary_ai_unjudgeable"]
            + result["human_binary_engineering_missing"], "可决策/弃权/工程缺失分解不符。")
    require(binary == agreement + result["false_pass_items"] + result["false_rejection_items"],
            "二元混淆计数不符。")
    return result


def join_items(human_rows, results, expected, batches, snapshots):
    human = {}
    pairs = {(sid, cid) for sid, rules in expected.items() for cid in rules}
    for row in human_rows:
        sid = row["sample_id"]
        if row["split"] != "holdout":
            require(sid not in expected, "保留样本人工划分错配。")
            continue
        label = row["record"]["label"]
        key = (sid, label["criterion_id"])
        require(key in pairs and key not in human, "人工参考未知或重复分配。")
        rule = expected[sid][key[1]]
        require(row["dimension"] == rule["dimension"] and row["required"] is rule["required"]
                and row["rule_version"] == "rules_v1" and row["snapshot_sha256"] == snapshots[sid]
                and label["status"] in HUMAN[:3], "人工参考的维度、必需性、版本或状态不符。")
        human[key] = label["status"]
    require(set(human) == pairs, "冻结人工参考未覆盖完整保留集。")
    indexed, output = {}, []
    for result in results:
        version, sid = result["prompt_version"], result["sample_id"]
        require(version in VERSIONS and sid in expected and (version, sid) not in indexed,
                "结果版本、划分或唯一样本身份不符。")
        require(result["batch_id"] == batches[version], "模型结果批次错配。")
        indexed[version, sid] = result
    require(set(indexed) == {(v, sid) for v in VERSIONS for sid in expected}, "两版保留集未完整覆盖。")
    for version in VERSIONS:
        for sid, rules in expected.items():
            result = indexed[version, sid]
            status, labels = result["engineering_status"], result["labels"]
            require(status in ("succeeded", "failed"), "存在未结束的模型任务。")
            ai = {}
            if status == "succeeded":
                require(isinstance(labels, list) and result["result_attempt_id"] and not result["stop_code"],
                        "成功任务缺少结果绑定。")
                for label in labels:
                    cid = label["criterion_id"]
                    require(cid in rules and cid not in ai and label["status"] in AI[:3],
                            "模型标签未知、重复或非法。")
                    ai[cid] = label["status"]
                require(set(ai) == set(rules), "成功任务缺失条目。")
            else:
                require(labels is None and result["result_attempt_id"] is None and result["stop_code"],
                        "失败任务不得包含伪造视觉判断。")
            for cid, rule in rules.items():
                output.append({"prompt_version": version, "batch_id": result["batch_id"],
                    "sample_id": sid, "criterion_id": cid, "dimension": rule["dimension"],
                    "required": rule["required"], "human_status": human[sid, cid],
                    "ai_status": ai.get(cid), "engineering_status": status,
                    "error_code": result["stop_code"], "result_attempt_id": result["result_attempt_id"]})
    return output


def summarize(items, catalog):
    require(all(r["prompt_version"] in VERSIONS and r["dimension"] in DIMENSIONS and r["criterion_id"] in catalog and
                catalog[r["criterion_id"]]["dimension"] == r["dimension"] and
                catalog[r["criterion_id"]]["required"] is r["required"] for r in items), "目录错配。")
    overview, dimensions, criteria, requiredness, paired = [], [], [], [], []
    for version in VERSIONS:
        rows = [r for r in items if r["prompt_version"] == version]
        overview.append({"prompt_version": version, "dimension": "all", **counts(rows)})
        for dimension in DIMENSIONS:
            dimensions.append({"prompt_version": version, "dimension": dimension,
                **counts([r for r in rows if r["dimension"] == dimension])})
        for cid in sorted({r["criterion_id"] for r in rows}):
            criteria.append({"prompt_version": version, "criterion_id": cid, **catalog[cid],
                             **counts([r for r in rows if r["criterion_id"] == cid])})
        for dimension in ("all", "subject_features"):
            for required in (True, False):
                selected = [r for r in rows if r["required"] is required and
                            (dimension == "all" or r["dimension"] == dimension)]
                requiredness.append({"prompt_version": version, "dimension": dimension,
                                     "required": required, **counts(selected)})
        for key in ("assigned_items", "required_items", "auxiliary_items", *["human_"+s for s in HUMAN],
                    *["ai_"+s for s in AI], "human_binary_items", "both_binary_items", "agreements",
                    "false_pass_items", "false_rejection_items"):
            total = overview[-1][key]
            require(total == sum(r[key] for r in dimensions if r["prompt_version"] == version)
                    == sum(r[key] for r in criteria if r["prompt_version"] == version), "维度/条目合计不符。")
    indexed = {(r["prompt_version"], r["sample_id"], r["criterion_id"]): r for r in items}
    pairs = {(r["sample_id"], r["criterion_id"]) for r in items}
    require(len(indexed) == len(items) == 2*len(pairs), "两版条目身份不唯一或不齐全。")
    for dimension in ("all", *DIMENSIONS):
        chosen = []
        for sid, cid in sorted(pairs):
            a, b = [indexed[v, sid, cid] for v in VERSIONS]
            require(all(a[k] == b[k] for k in ("human_status", "dimension", "required")), "配对人工参考错配。")
            if ((dimension == "all" or a["dimension"] == dimension) and a["human_status"] in HUMAN[:2]
                    and a["ai_status"] in AI[:2] and b["ai_status"] in AI[:2]):
                chosen.append((a, b))
        transitions = Counter((a["ai_status"] == a["human_status"], b["ai_status"] == b["human_status"])
                              for a, b in chosen)
        n = len(chosen)
        paired.append({"dimension": dimension, "common_decidable_items": n,
            "assigned_images": len({a["sample_id"] for a, _ in chosen}),
            "both_agree": transitions[True, True], "both_disagree": transitions[False, False],
            "v1_only_agrees": transitions[True, False], "v2_only_agrees": transitions[False, True],
            "v1_agreement": ratio(sum(a["ai_status"] == a["human_status"] for a, _ in chosen), n),
            "v2_agreement": ratio(sum(b["ai_status"] == b["human_status"] for _, b in chosen), n)})
    return {"overview": overview, "dimensions": dimensions, "criteria": criteria,
            "requiredness": requiredness, "paired": paired}


def load_sources(folder=client.PRIVATE / "holdout_g01", store=None):
    folder = Path(folder).resolve()
    store = store or MetricRuns()
    require(folder == store.folder.resolve(), "需使用正式绑定的 G01 来源。")
    plan = store.binding()
    loaded = load_registered_reference()
    require(loaded["reference"] == plan["frozen_inputs"]["reference"], "人工参考与 G01 不同。")
    receipt = runs.pv.read_json(folder / "completion_receipt.json")
    require(receipt["task"] == "G01", "缺少 G01 完成回执。")
    for name in ("experiment.json", "results.json", "verification.json", "checksums.json"):
        require(client.digest((folder / name).read_bytes()) == receipt["artifact_sha256"][name],
                "G01 完成产物绑定变化。")
    checksums = runs.pv.read_json(folder / "checksums.json")
    for name, sha in checksums.items():
        path = ((store.path.parent / name) if name.startswith("runs\\") or name.startswith("runs/")
                else folder / name).resolve()
        require(path.is_relative_to(store.path.parent.resolve()) and client.digest(path.read_bytes()) == sha,
                "G01 输入/响应/解析/校验文件缺失或哈希变化。")
    status = runs.compact_status(store)
    verification = runs.pv.read_json(folder / "verification.json")
    require(status["both_versions_terminal"] and verification["both_versions_terminal"], "两版尚未结束。")
    require(status["versions"] == verification["versions"], "正式库与 G01 汇总不同。")
    results = runs.pv.read_json(folder / "results.json")
    worker = runs.HoldoutWorker(store)
    policy, reference, evaluation, samples = load_frozen_context()
    expected = {}
    result_index = {(r["prompt_version"], r["sample_id"]): r for r in results}
    for version, bid in plan["batches"].items():
        batch = store.batch(bid)
        for task in store.tasks(bid):
            prepared = worker.inputs(task, batch)
            sid = task["sample_id"]
            if sid in expected:
                require(expected[sid] == prepared["expected"], "两版有效条目不同。")
            expected[sid] = prepared["expected"]
            exported = result_index[version, sid]
            require(exported["engineering_status"] == task["status"] and
                    exported["result_attempt_id"] == task["result_attempt_id"] and
                    exported["stop_code"] == task["stop_code"], "模型导出与正式任务绑定不同。")
            with store.connect() as db:
                attempts = [dict(r) for r in db.execute("SELECT * FROM attempts WHERE task_id=?", (task["id"],))]
            for attempt in attempts:
                require(attempt["status"] in ("succeeded", "failed"), "存在未核对的中断尝试。")
                _, hashes, _ = worker.evidence(attempt, prepared, batch)
                require(hashes == json.loads(attempt["artifact_hashes_json"]), "尝试证据与正式库不同。")
            if task["status"] == "succeeded":
                attempt = next(a for a in attempts if a["id"] == task["result_attempt_id"])
                parsed = runs.pv.read_json(Path(attempt["evidence_dir"]) / "parsed.json")
                require(parsed["labels"] == exported["labels"] and
                        evaluate_result(sid, parsed, expected[sid], policy)["engineering_status"] == "valid_result",
                        "成功结果导出或条目校验不符。")
    items = join_items(loaded["initial_rows"], results, expected, plan["batches"], plan["frozen_inputs"]["snapshots"])
    require(len(items) == 1080, "固定保留集应为 540 项×两版。")
    for version in VERSIONS:
        require(sum(r["ai_status"] is None for r in items if r["prompt_version"] == version) ==
                verification["engineering_missing_items"].get(version, 0), "工程缺失与 G01 不符。")
    catalog = {}
    for dimension, rules in (("subject_features", reference["criteria"]),
            ("structural_integrity", evaluation["structure_criteria"]), ("prompt_adherence", evaluation["prompt_criteria"])):
        for rule in rules:
            catalog[rule["id"]] = {"criterion_name": rule.get("name", rule.get("description", rule["id"])),
                                  "dimension": dimension, "required": rule["required"]}
    fingerprints = {name: client.digest((folder / name).read_bytes()) for name in
                    ("completion_receipt.json", "experiment.json", "results.json", "verification.json", "checksums.json")}
    return items, catalog, {"reference": loaded["reference"], "g01_files_sha256": fingerprints,
        "g01_directory": str(folder), "batches": plan["batches"], "engineering": verification["versions"]}


def flat(row):
    result = {}
    for key, value in row.items():
        if key == "matrix":
            for h in HUMAN:
                for a in AI:
                    result[f"human_{h}_ai_{a}"] = value[h][a]
        elif isinstance(value, dict):
            for suffix in ("numerator", "denominator", "value", "status"):
                result[key+"_"+suffix] = value[suffix]
        else:
            result[key] = value
    return result


def csv_text(rows):
    rows = [flat(r) for r in rows]
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def markdown(report):
    lines = ["# 保留集 V1/V2 主要指标", "", f"计算时间：{report['created_at']}", "",
        "使用同一冻结保留集 30 张、每版 540 个已分配条目及 human_reference_v1 初始人工参考。",
        "主表按条目计数，主体特征包含必需与辅助项；不改变原必需性或候选聚合。", "",
        "| 维度 | 版本 | 人工可判断 | 双方可判断 | 一致数 | 决策覆盖率 | 条件一致率 | 错误通过比例 | 错误拒绝比例 |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |"]
    for r in report["data"]["dimensions"]:
        lines.append(f"| {DIMENSIONS[r['dimension']]} | {r['prompt_version']} | {r['human_binary_items']} | "
            f"{r['both_binary_items']} | {r['agreements']} | " + " | ".join(display(r[k]) for k in METRICS) + " |")
    lines += ["", "## 总体与状态分解", "",
        "| 维度 | 版本 | 发放项 | 人工 pass/fail/不可判断/不适用 | AI pass/fail/不可判断 | 工程缺失 |",
        "| --- | --- | ---: | --- | --- | ---: |"]
    for r in report["data"]["overview"] + report["data"]["dimensions"]:
        lines.append(f"| {DIMENSIONS.get(r['dimension'], '全部维度')} | {r['prompt_version']} | {r['assigned_items']} | "
            + "/".join(str(r['human_'+s]) for s in HUMAN) + " | "
            + "/".join(str(r['ai_'+s]) for s in AI[:3]) + f" | {r['ai_engineering_missing']} |")
    lines += ["", "总体条目加权汇总（非三个维度百分比的算术平均）：", ""]
    for r in report["data"]["overview"]:
        lines.append(f"- {r['prompt_version']}：覆盖 {display(r['decision_coverage'])}；条件一致 "
            f"{display(r['conditional_agreement'])}；错误通过 {display(r['false_pass'])}；错误拒绝 {display(r['false_rejection'])}。")
    lines += ["", "## 必需项与辅助项", "",
        "| 范围 | 版本 | 必需性 | 发放项 | 覆盖率 | 条件一致率 |",
        "| --- | --- | --- | ---: | --- | --- |"]
    for r in report["data"]["requiredness"]:
        lines.append(f"| {DIMENSIONS.get(r['dimension'], '全部维度')} | {r['prompt_version']} | "
            f"{'必需' if r['required'] else '辅助'} | {r['assigned_items']} | {display(r['decision_coverage'])} | "
            f"{display(r['conditional_agreement'])} |")
    lines += ["", "## 补充：两版在同一条目均可判断的子集", "",
        "下表只选择人工与两版均为 pass/fail 的相同条目。它有选择性覆盖，不代替全保留集主要指标。", "",
        "| 维度 | 共同条目 | V1 一致率 | V2 一致率 | 仅 V1 一致 | 仅 V2 一致 |",
        "| --- | ---: | --- | --- | ---: | ---: |"]
    for r in report["data"]["paired"]:
        lines.append(f"| {DIMENSIONS.get(r['dimension'], '全部维度')} | {r['common_decidable_items']} | "
            f"{display(r['v1_agreement'])} | {display(r['v2_agreement'])} | {r['v1_only_agrees']} | {r['v2_only_agrees']} |")
    lines += ["", "## 计算口径与限制", "",
        "- 可判断表示 pass/fail。决策覆盖率分母为人工 pass+fail，分子为其中 AI 也给出 pass/fail 的项数。",
        "- 条件一致率只在双方均可判断的条目上计算；人工不可判断/不适用不进入四项二元指标分母。",
        "- 错误通过比例 = 人工 fail 且 AI pass / 全部人工 fail；错误拒绝比例 = 人工 pass 且 AI fail / 全部人工 pass。",
        "- AI 不可判断和工程缺失分别保留。它们不计入二元错误分子，但仍留在人工 pass/fail 分母中。因此错误比例较低不能单独证明质量更好，必须结合覆盖率。",
        "- 零分母写不可计算，JSON 使用 null 并附 zero_denominator；不替换成 0% 或 100%。",
        "- 作者参考并非客观真值，含明确批量判断及助手整理的文字。主要结果只用已冻结初标；修订另列。",
        "- 每张图含多个相关条目，且图片只有四个 seed 家族、共享主体与提示词模板；条目数量不等于独立样本量。不做显著性或跨主体泛化结论。",
        "- 工程成功不等于图片质量通过；不会根据保留结果修改 V2、规则或人工标签。本次没有 API 调用，分歧页面解锁属于 G03。",
        "", "每条数据的状态、必需性、样本/条目及结果来源保存在私有 items.jsonl；criteria.csv 给出具体条目分母及指标。", ""]
    return "\n".join(lines)


def verify_export(folder, *, refresh_source=True):
    folder = Path(folder)
    checks = runs.pv.read_json(folder / "checksums.json")
    require(set(checks) == FILES and {p.name for p in folder.iterdir()} == FILES | {"checksums.json"}, "G02 文件清单不符。")
    require(all(client.digest((folder/name).read_bytes()) == sha for name, sha in checks.items()), "G02 文件哈希变化。")
    report = runs.pv.read_json(folder / "metrics.json")
    require(report["task"] == "G02" and report["generator_sha256"] == text_digest(Path(__file__)), "G02 版本绑定变化。")
    items = [client.strict_json(line) for line in (folder / "items.jsonl").read_bytes().splitlines()]
    require(summarize(items, report["catalog"]) == report["data"], "指标与逐条数据不同。")
    if refresh_source:
        source_items, catalog, source = load_sources()
        require(items == source_items and catalog == report["catalog"] and source == report["source"], "指标来源变化。")
    for name in ("dimensions", "overview", "criteria", "requiredness", "paired"):
        require((folder/(name+".csv")).read_bytes() == ("\ufeff"+csv_text(report["data"][name])).encode(), "CSV 与指标不同。")
        parsed = list(csv.DictReader(io.StringIO((folder/(name+".csv")).read_text(encoding="utf-8-sig"))))
        require(len(parsed) == len(report["data"][name]), "CSV 读回行数不同。")
    require((folder/"保留集主要指标.md").read_text(encoding="utf-8") == markdown(report), "报告与指标不同。")
    return {"status": "verified_G02", "items": len(items), "dimension_rows": len(report["data"]["dimensions"]),
            "criterion_rows": len(report["data"]["criteria"]), "api_calls": 0}


def generate():
    items, catalog, source = load_sources()
    folder = client.PRIVATE / "holdout_metrics" / (datetime.fromisoformat(now()).strftime("g02_%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "task": "G02", "created_at": now(), "generator_sha256": text_digest(Path(__file__)),
              "api_calls": 0, "source": source, "catalog": catalog, "data": summarize(items, catalog)}
    client.write_json(folder/"metrics.json", report)
    with (folder/"items.jsonl").open("xb") as stream:
        for row in items:
            stream.write((json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False)+"\n").encode())
    for name in ("dimensions", "overview", "criteria", "requiredness", "paired"):
        (folder/(name+".csv")).write_bytes(("\ufeff"+csv_text(report["data"][name])).encode())
    (folder/"保留集主要指标.md").write_text(markdown(report), encoding="utf-8", newline="\n")
    client.write_json(folder/"checksums.json", {name: client.digest((folder/name).read_bytes()) for name in sorted(FILES)})
    verified = verify_export(folder, refresh_source=False)
    client.write_json(client.PRIVATE/"g02_latest.json", {"output": str(folder), **verified})
    return {"output": str(folder), **verified}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    try:
        result = verify_export(args.verify) if args.verify else generate()
        print(json.dumps(result, ensure_ascii=True, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"G02 计算/核验失败：{type(exc).__name__}；请检查冻结来源及文件完整性。\n")


if __name__ == "__main__":
    main()
