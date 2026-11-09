"""E04: register a private initial human reference version; no label edits or model calls."""

import argparse
import json
import re
from pathlib import Path

import audit_annotations as audit
import label_statistics as stats
from annotation_store import DEFAULT_DB, PRIVATE, now
from build_manifest import DATA, text_digest


VERSION = "human_reference_v1"
ANNOTATION_FILES = audit.EXPORT_NAMES | {"checksums.json"}
STATISTICS_FILES = stats.FILES | {"checksums.json"}
PACKAGE_FILES = ({"seal.json", "creation_state.json", "使用说明.md"}
                 | {"annotations/" + name for name in ANNOTATION_FILES}
                 | {"statistics/" + name for name in STATISTICS_FILES})
USAGE = """# 冻结的初始人工参考

本目录是主要评测使用的正式初始人工参考，版本与冻结时间见 seal.json。
annotations/initial_labels.jsonl 保存主要参考；annotations/current_labels.jsonl
是 E02 导出时的当前记录，只作历史来源，不作为后续主要指标的替代输入。
creation_state.json 保留冻结当时人工库的当前记录、初始快照与修订原因。
statistics/ 保留冻结前完成的 E03 统计和来源、零分母限制。

E02/E03 文件原样复制，其中尚未整体冻结的状态文字是历史记录；E04 的正式
冻结状态由本目录 seal.json 指定，不通过改写旧记录制造更早的冻结时间。
快照含作者批量指定判断及助手辅助文字，不称全量逐项独立撰写或客观真值。

后续在页面修订时保留原值、新值与原因，不覆盖本目录。主要指标使用本版
初始参考，修订后的结果另列。此登记不解锁保留集模型执行或分歧显示。
使用复验命令并提供已登记的 seal.json SHA-256，确认版本和内容未变。
哈希用于发现变化，不是签名或操作系统层面的防写锁。
"""


def read_json(path):
    return audit.strict_json(Path(path).read_text(encoding="utf-8"))


def hashes(folder, names):
    return {name: audit.sha((Path(folder) / name).read_bytes()) for name in sorted(names)}


def validate_inputs(annotations, statistics):
    """Verify using the specified copies, independent of old absolute source locations."""
    result = audit.verify_export(annotations)
    context = read_json(Path(annotations) / "context.json")
    state = read_json(Path(annotations) / "database_rows.json")
    _, initial, _ = audit.audit_state(state, context)
    statistics = Path(statistics)
    checksum = read_json(statistics / "checksums.json")
    audit.require(set(checksum) == stats.FILES and {p.name for p in statistics.iterdir()} == STATISTICS_FILES,
                  "E03 统计文件清单不符。")
    audit.require(hashes(statistics, stats.FILES) == checksum, "E03 统计文件哈希不符。")
    report = read_json(statistics / "statistics.json")
    audit.require(report["initial_only"] is True and report["global_reference_frozen"] is False,
                  "需要已完成 E03 的初始标签统计。")
    audit.require(report["source_files_sha256"] == hashes(annotations, ANNOTATION_FILES),
                  "E03 与指定 E02 导出来源不一致。")
    computed = stats.summarize(initial, context, report["criterion_catalog"])
    audit.require(computed == report["data"], "E03 分布与初始参考不一致。")
    for name in ("overview", "dimensions", "criteria"):
        audit.require((statistics / (name + ".csv")).read_bytes() == (
            "\ufeff" + stats.csv_text(computed[name])).encode("utf-8"), "统计 CSV 内容不符。")
    audit.require((statistics / "标签统计说明.md").read_text(encoding="utf-8") == stats.markdown(report),
                  "统计说明与初始参考不一致。")
    return context, state, initial, result["summary"]


def verify_reference(folder, expected_seal_sha256=None, expected_version=VERSION):
    folder = Path(folder)
    seal_path = folder / "seal.json"
    seal_sha = audit.sha(seal_path.read_bytes())
    if expected_seal_sha256 is not None:
        audit.require(seal_sha == expected_seal_sha256, "冻结登记哈希不符。")
    seal = read_json(seal_path)
    audit.require(seal["schema_version"] == 1 and seal["reference_version"] == expected_version
                  and seal["status"] == "frozen" and seal["primary_label_source"] == "confirmed_initial_snapshots",
                  "冻结版本或状态不符。")
    audit.require(isinstance(seal["authorized_by"], str) and seal["authorized_by"].strip(), "冻结授权人缺失。")
    audit.timestamp(seal["frozen_at"])
    leaves = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    audit.require(leaves == PACKAGE_FILES | {"checksums.json"}, "冻结包文件不完整或存在额外文件。")
    audit.require(not any(p.is_symlink() for p in folder.rglob("*")), "冻结包不能含符号链接。")
    checksum = read_json(folder / "checksums.json")
    audit.require(set(checksum) == PACKAGE_FILES and hashes(folder, PACKAGE_FILES) == checksum,
                  "冻结包文件哈希不符。")
    audit.require(seal["source_files_sha256"] == {
        "annotations": hashes(folder / "annotations", ANNOTATION_FILES),
        "statistics": hashes(folder / "statistics", STATISTICS_FILES)}, "冻结来源文件绑定不符。")
    audit.require(seal["usage_sha256"] == audit.sha((folder / "使用说明.md").read_bytes()),
                  "冻结使用说明绑定不符。")
    context, state, initial, summary = validate_inputs(folder / "annotations", folder / "statistics")
    creation = read_json(folder / "creation_state.json")
    creation_summary, creation_initial, _ = audit.audit_state(creation, context)
    audit.require(creation["initial_snapshots"] == state["initial_snapshots"] and creation_initial == initial,
                  "冻结当时的初始快照与正式参考不符。")
    audit.require(seal["rule_version"] == context["rule_version"]
                  and seal["ruleset_sha256"] == context["ruleset_sha256"]
                  and seal["manifest_sha256"] == context["files_text_sha256"]["manifest.jsonl"]
                  and seal["split_sha256"] == context["files_text_sha256"]["split.json"], "冻结规则或划分绑定不符。")
    audit.require(seal["initial_labels_sha256"] == audit.sha((folder / "annotations/initial_labels.jsonl").read_bytes())
                  and seal["initial_snapshot_sha256"] == {r["sample_id"]: r["sha256"] for r in state["initial_snapshots"]},
                  "冻结初始标签或快照绑定不符。")
    audit.require(seal["summary"] == summary and seal["creation_state_sha256"] == audit.state_sha(creation)
                  and seal["post_confirmation_changes_at_freeze"] == creation_summary["post_confirmation_changes"],
                  "冻结记录计数或修订来源不符。")
    return {"status": "verified_frozen_reference", "reference_version": seal["reference_version"],
            "frozen_at": seal["frozen_at"], "seal_sha256": seal_sha,
            "initial_labels_sha256": seal["initial_labels_sha256"],
            "sample_count": summary["sample_count"], "item_count": summary["item_count"], "api_calls": 0}


def load_reference(folder, *, expected_seal_sha256, expected_version=VERSION):
    """Future metric consumers must supply the registered seal hash and use these initial rows."""
    audit.require(isinstance(expected_seal_sha256, str) and len(expected_seal_sha256) == 64,
                  "读取主要参考需要已登记的冻结哈希。")
    verified = verify_reference(folder, expected_seal_sha256, expected_version)
    rows = [audit.strict_json(line) for line in (Path(folder) / "annotations/initial_labels.jsonl").read_text(
        encoding="utf-8").splitlines()]
    return {"reference": verified, "initial_rows": rows}


def load_registered_reference(registration=DATA / "human_reference_registration.json"):
    """Resolve the committed version/hash registration, never the mutable labels table."""
    registered = read_json(registration)
    audit.require(registered["schema_version"] == 1, "未知人工参考登记格式。")
    folder = (PRIVATE / registered["relative_private_path"]).resolve()
    audit.require(folder.is_relative_to(PRIVATE.resolve()) and folder != PRIVATE.resolve(),
                  "人工参考登记路径必须位于 private 子目录。")
    loaded = load_reference(folder, expected_seal_sha256=registered["seal_sha256"],
                            expected_version=registered["reference_version"])
    for key in ("initial_labels_sha256", "frozen_at", "sample_count", "item_count"):
        audit.require(loaded["reference"][key] == registered[key], f"人工参考登记字段不符：{key}")
    return loaded


def freeze_reference(annotations, statistics, *, version=VERSION, database=DEFAULT_DB, authorized_by="水水"):
    audit.require(re.fullmatch(r"[a-z][a-z0-9_]{0,63}", version) is not None, "参考版本名非法。")
    audit.require(isinstance(authorized_by, str) and authorized_by.strip(), "冻结授权人不能为空。")
    annotations, statistics = Path(annotations).resolve(), Path(statistics).resolve()
    context, state, initial, summary = validate_inputs(annotations, statistics)
    current_context = audit.load_context()
    audit.require(context == current_context, "当前规则、来源或核验代码与 E02 绑定不同，需复核后再冻结。")
    audit.validate_media(context)
    current = audit.read_database(database)
    creation_summary, current_initial, _ = audit.audit_state(current, context)
    audit.require(current["initial_snapshots"] == state["initial_snapshots"] and current_initial == initial,
                  "当前初始快照与 E02 不同，不能冻结旧导出。")
    folder = PRIVATE / "human_references" / version
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir(exist_ok=False)
    source_hashes = {"annotations": hashes(annotations, ANNOTATION_FILES),
                     "statistics": hashes(statistics, STATISTICS_FILES)}
    for name, source, files in (("annotations", annotations, ANNOTATION_FILES),
                                ("statistics", statistics, STATISTICS_FILES)):
        (folder / name).mkdir()
        for filename in files:
            with (folder / name / filename).open("xb") as stream:
                stream.write((source / filename).read_bytes())
    seal = {"schema_version": 1, "task": "E04", "reference_version": version, "status": "frozen",
            "frozen_at": now(), "authorized_by": authorized_by.strip(),
            "primary_label_source": "confirmed_initial_snapshots",
            "rule_version": context["rule_version"], "ruleset_sha256": context["ruleset_sha256"],
            "manifest_sha256": context["files_text_sha256"]["manifest.jsonl"],
            "split_sha256": context["files_text_sha256"]["split.json"],
            "initial_labels_sha256": source_hashes["annotations"]["initial_labels.jsonl"],
            "initial_snapshot_sha256": {r["sample_id"]: r["sha256"] for r in state["initial_snapshots"]},
            "creation_state_sha256": audit.state_sha(current),
            "post_confirmation_changes_at_freeze": creation_summary["post_confirmation_changes"],
            "source_files_sha256": source_hashes, "summary": summary,
            "usage_sha256": audit.sha(USAGE.encode("utf-8")),
            "freezer_text_sha256": text_digest(Path(__file__)),
            "revision_policy": "Keep initial values, revised values and reasons separately; primary metrics use this reference.",
            "limits": ["Author reference, not objective ground truth.",
                       "Includes explicit batch author judgments and assistant wording.",
                       "No holdout model execution or disagreement unlock is granted by this registration."],
            "api_calls": 0}
    audit.write_json(folder / "creation_state.json", current)
    audit.write_json(folder / "seal.json", seal)
    with (folder / "使用说明.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(USAGE)
    audit.require(source_hashes == {"annotations": hashes(annotations, ANNOTATION_FILES),
                                    "statistics": hashes(statistics, STATISTICS_FILES)}, "冻结期间源导出变化。")
    audit.require(audit.read_database(database) == current, "冻结期间人工库变化；需重新核对。")
    audit.require(audit.load_context() == context, "冻结期间规则或来源变化。")
    audit.validate_media(context)
    audit.write_json(folder / "checksums.json", hashes(folder, PACKAGE_FILES))
    registered_sha = audit.sha((folder / "seal.json").read_bytes())
    return {"output": str(folder), **verify_reference(folder, registered_sha, version)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--statistics", type=Path)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--expected-seal-sha256")
    parser.add_argument("--verify-registered", action="store_true")
    args = parser.parse_args()
    try:
        if args.verify_registered:
            audit.require(not args.verify and not args.annotations and not args.statistics
                          and not args.expected_seal_sha256, "核验登记版本时不指定新来源。")
            result = load_registered_reference()["reference"]
        elif args.verify:
            audit.require(not args.annotations and not args.statistics, "复验不接受新来源。")
            result = verify_reference(args.verify, args.expected_seal_sha256, args.version)
        else:
            audit.require(args.annotations and args.statistics and not args.expected_seal_sha256,
                          "冻结需要指定 --annotations 和 --statistics。")
            result = freeze_reference(args.annotations, args.statistics, version=args.version, database=args.db)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f"E04 冻结或核验失败：{exc}\n")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
