"""E02: read-only annotation audit and private, non-overwriting export. No API calls."""

import argparse
import hashlib
import io
import json
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PIL import Image

from annotation_store import AnnotationStore, DEFAULT_DB, PRIVATE, encode, now
from build_manifest import DATA, read_jsonl, text_digest, unique_index
from evaluate_results import expected_for_sample
from freeze_rules import load_frozen_context
from split_manifest import freeze, load_verified_inputs


TABLE_ORDER = {"metadata": "key", "labels": "sample_id,criterion_id", "history": "id",
               "imports": "sha256", "initial_snapshots": "sample_id"}
EXPORT_NAMES = {"database_rows.json", "context.json", "initial_labels.jsonl",
                "current_labels.jsonl", "audit_report.json"}


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def strict_json(text):
    def pairs(items):
        obj = {}
        for key, value in items:
            require(key not in obj, f"JSON 重复键：{key}")
            obj[key] = value
        return obj
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda value: require(False, f"非法 JSON 常量：{value}"))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def state_sha(state):
    return sha(encode(state).encode("utf-8"))


def read_database(path):
    # Never instantiate AnnotationStore: its constructor initializes/writes metadata.
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=15)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")  # All five tables come from one SQLite read snapshot.
        require([r[0] for r in db.execute("PRAGMA integrity_check")] == ["ok"],
                "SQLite 完整性检查失败。")
        state = {table: [dict(row) for row in db.execute(
            f"SELECT * FROM {table} ORDER BY {order}")] for table, order in TABLE_ORDER.items()}
    return state


def load_context():
    policy, reference, evaluation, samples = load_frozen_context()
    require((DATA / "split.json").is_file(), "冻结划分缺失；不创建新划分。")
    rows, groups, bindings = load_verified_inputs()
    split, created = freeze(DATA, rows, groups, bindings)
    require(not created, "不允许重建划分。")
    manifest = unique_index(read_jsonl(DATA / "manifest.jsonl"), "sample_id")
    sources = unique_index(read_jsonl(DATA / "sample_sources.jsonl"), "sample_id")
    require(set(manifest) <= set(sources), "候选图片来源映射缺失。")
    expected = {sid: expected_for_sample(sid, reference, evaluation, samples) for sid in manifest}
    references = strict_json((DATA / "reference_assets.json").read_text(encoding="utf-8"))
    require([r["sha256"] for r in reference["reference_images"]] ==
            [r["sha256"] for r in references], "参考图规则和素材映射不一致。")
    return {"schema_version": 1, "rule_version": policy["rule_version"],
            "ruleset_sha256": text_digest(DATA / "ruleset.json"),
            "manifest": manifest, "expected": expected, "references": references,
            "candidates": [{"sample_id": sid, "source_path": sources[sid]["source_path"],
                            "sha256": row["candidate_sha256"], "width": row["candidate_width"],
                            "height": row["candidate_height"]} for sid, row in manifest.items()],
            "split": split, "files_text_sha256": {name: text_digest(DATA / name) for name in (
                "ruleset.json", "manifest.jsonl", "split.json", "sample_sources.jsonl",
                "reference_assets.json", "sample_criteria.jsonl", "label_policy.json")},
            "exporter_text_sha256": text_digest(Path(__file__)),
            "annotation_store_text_sha256": text_digest(Path(__file__).with_name("annotation_store.py"))}


def validate_media(context):
    receipts = []
    for role, rows in (("reference", context["references"]), ("candidate", context["candidates"])):
        for row in rows:
            identity = row.get("sample_id", row.get("reference_id"))
            path = Path(row["source_path"])
            raw = path.read_bytes()
            require(sha(raw) == row["sha256"], f"{identity}：源图片哈希不符。")
            with Image.open(io.BytesIO(raw)) as image:
                require(image.size == (row["width"], row["height"]), f"{identity}：尺寸不符。")
                require(getattr(image, "n_frames", 1) == 1, f"{identity}：图片并非单帧。")
                image.load()  # Fully decode pixels, including JPEGs whose verify() is a no-op.
            receipts.append({"asset_id": identity, "role": role, "source_path": str(path),
                             "sha256": row["sha256"], "size": [row["width"], row["height"]]})
    return receipts


def timestamp(text):
    result = datetime.fromisoformat(text)
    require(result.tzinfo is not None, "记录时间缺少时区。")
    return result


def record_from_row(row, historical=False):
    return {"label": strict_json(row["payload"]), "annotator": row["annotator"],
            "revision": row["revision"], "origin": strict_json(row["origin"]),
            "updated_at": row["recorded_at" if historical else "updated_at"]}


def audit_state(state, context):
    manifest, expected = context["manifest"], context["expected"]
    require(set(expected) == set(manifest), "规则与清单样本集合不符。")
    binding = {r["key"]: r["value"] for r in state["metadata"]}
    require(strict_json(binding.get("binding", "null")) == {
        "schema": 1, "rule_version": context["rule_version"],
        "ruleset_sha256": context["ruleset_sha256"]}, "人工库规则绑定不符。")
    validator = AnnotationStore.__new__(AnnotationStore)
    validator.expected = expected
    current = {}
    histories = {}

    def validate_record(sid, cid, record, draft=False):
        require(sid in expected and cid in expected[sid], f"{sid}/{cid}：未知条目。")
        require(set(record) == {"label", "annotator", "revision", "origin", "updated_at"},
                f"{sid}/{cid}：记录元数据字段不完整。")
        require(record["label"].get("criterion_id") == cid, f"{sid}/{cid}：条目编号错配。")
        try:
            validator.validate(sid, record["label"], allow_draft=draft)
        except ValueError as exc:
            raise AuditError(f"{sid}/{cid}：{exc}") from exc
        require(isinstance(record["annotator"], str) and record["annotator"].strip(),
                f"{sid}/{cid}：缺少标注人。")
        require(type(record["revision"]) is int and record["revision"] > 0,
                f"{sid}/{cid}：修订版本非法。")
        require(isinstance(record["origin"], dict) and record["origin"].get("kind"),
                f"{sid}/{cid}：缺少来源。")
        timestamp(record["updated_at"])

    for row in state["history"]:
        sid, cid = row["sample_id"], row["criterion_id"]
        record = record_from_row(row, historical=True)
        validate_record(sid, cid, record, draft=True)  # Historical drafts are preserved, not completed.
        require(isinstance(row["change_reason"], str) and row["change_reason"].strip(),
                f"{sid}/{cid}：历史记录缺少原因。")
        key = (sid, cid, record["revision"])
        require(key not in histories, f"{sid}/{cid}：重复历史版本。")
        histories[key] = record
    for row in state["labels"]:
        sid, cid = row["sample_id"], row["criterion_id"]
        record = record_from_row(row)
        validate_record(sid, cid, record)
        require((sid, cid) not in current, f"{sid}/{cid}：重复当前条目。")
        current[sid, cid] = record
        versions = sorted(v for s, c, v in histories if (s, c) == (sid, cid))
        require(versions == list(range(1, record["revision"] + 1)), f"{sid}/{cid}：历史版本不连续。")
        require(histories.get((sid, cid, record["revision"])) == record,
                f"{sid}/{cid}：当前记录与最新历史不符。")
    expected_keys = {(sid, cid) for sid, rows in expected.items() for cid in rows}
    require(set(current) == expected_keys,
            f"当前条目覆盖不完整；缺失 {sorted(expected_keys - set(current))}；额外 {sorted(set(current) - expected_keys)}")
    require(all((s, c) in current for s, c, _ in histories), "存在孤立历史记录。")
    for row in state["imports"]:
        for key, hash_key in (("raw_markdown", "sha256"), ("original_markdown", "original_sha256")):
            require(sha(row[key].encode("utf-8")) == row[hash_key], "历史导入原文哈希不符。")
        if row["supplement_log"] is not None:
            require(isinstance(strict_json(row["supplement_log"]), dict), "补写来源日志非法。")
    snapshots = unique_index(state["initial_snapshots"], "sample_id")
    require(set(snapshots) == set(manifest), "初始确认快照未恰好覆盖全量图片。")
    initial_rows, current_rows, changes = [], [], []
    progress = {}
    origins = Counter()
    for sid in sorted(manifest):
        snap = snapshots[sid]
        require(sha(snap["payload"].encode("utf-8")) == snap["sha256"], f"{sid}：快照哈希不符。")
        payload = strict_json(snap["payload"])
        require(set(payload) == {"sample_id", "rule_version", "ruleset_sha256", "candidate_sha256",
                                 "reference_sha256", "records"}, f"{sid}：快照字段不完整。")
        require(payload["sample_id"] == sid and payload["rule_version"] == context["rule_version"]
                and payload["ruleset_sha256"] == context["ruleset_sha256"], f"{sid}：快照版本错配。")
        require(payload["candidate_sha256"] == manifest[sid]["candidate_sha256"],
                f"{sid}：快照候选图绑定错配。")
        require(payload["reference_sha256"] == [r["sha256"] for r in context["references"]],
                f"{sid}：快照参考图绑定错配。")
        require(set(payload["records"]) == set(expected[sid]), f"{sid}：快照条目覆盖不完整。")
        require(isinstance(snap["confirmed_by"], str) and snap["confirmed_by"].strip(),
                f"{sid}：缺少确认人。")
        confirmed_at = timestamp(snap["confirmed_at"])
        part = manifest[sid]["split"]
        counts = progress.setdefault(part, {"samples": 0, "items": 0, "confirmed_snapshots": 0})
        counts["samples"] += 1
        counts["confirmed_snapshots"] += 1
        for cid in sorted(expected[sid]):
            record = payload["records"][cid]
            validate_record(sid, cid, record)
            require(timestamp(record["updated_at"]) <= confirmed_at, f"{sid}/{cid}：确认早于记录。")
            require(histories.get((sid, cid, record["revision"])) == record,
                    f"{sid}/{cid}：初始记录与历史来源不符。")
            base = {"sample_id": sid, "split": part, **expected[sid][cid],
                    "rule_version": context["rule_version"], "snapshot_sha256": snap["sha256"]}
            initial_rows.append({**base, "record": record})
            latest = current[sid, cid]
            current_rows.append({**base, "record": latest})
            origins[record["origin"]["kind"]] += 1
            counts["items"] += 1
            if latest != record:
                changes.append({"sample_id": sid, "criterion_id": cid,
                                "initial_revision": record["revision"], "current_revision": latest["revision"],
                                "label_changed": latest["label"] != record["label"]})
    return {"sample_count": len(manifest), "item_count": len(current),
            "initial_snapshot_count": len(snapshots), "history_count": len(histories),
            "progress": progress, "initial_origin_counts": dict(origins),
            "post_confirmation_changes": changes}, initial_rows, current_rows


def write_json(path, obj):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(encode(row) + "\n")


def verify_export(folder):
    folder = Path(folder)
    checksums = strict_json((folder / "checksums.json").read_text(encoding="utf-8"))
    require(set(checksums) == EXPORT_NAMES, "导出文件哈希清单不完整或含未知路径。")
    require({p.name for p in folder.iterdir()} == EXPORT_NAMES | {"checksums.json"},
            "导出目录包含缺失或额外文件。")
    for name, digest in checksums.items():
        require(sha((folder / name).read_bytes()) == digest, f"导出文件哈希不符：{name}")
    state, context, report = [strict_json((folder / name).read_text(encoding="utf-8"))
                              for name in ("database_rows.json", "context.json", "audit_report.json")]
    summary, initial, current = audit_state(state, context)
    for name, expected in (("initial_labels.jsonl", initial), ("current_labels.jsonl", current)):
        actual = [strict_json(line) for line in (folder / name).read_text(encoding="utf-8").splitlines()]
        require(actual == expected, f"{name} 与数据库快照不一致。")
    require(report["summary"] == summary and report["database_state_sha256"] == state_sha(state),
            "核验报告与导出内容不一致。")
    require(report["status"] == "verified_export" and report["global_reference_frozen"] is False,
            "导出状态非法；E02 不执行整体冻结。")
    return {"status": "verified_export", "summary": summary, "api_calls": 0}


def export_annotations(db_path=DEFAULT_DB, output=None):
    context = load_context()
    state = read_database(db_path)
    summary, initial, current = audit_state(state, context)
    media = validate_media(context)
    # CLI exports are restricted to the ignored private tree, including custom destinations.
    folder = Path(output) if output else PRIVATE / "annotation_exports" / (
        datetime.fromisoformat(now()).strftime("e02_%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8])
    folder = folder.resolve()
    require(folder.is_relative_to(PRIVATE.resolve()) and folder != PRIVATE.resolve(),
            "导出目录必须位于本机 private 子目录。")
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir(exist_ok=False)  # Reserve a new directory, including on repeated runs.
    report = {"task": "E02", "schema_version": 1, "status": "verified_export", "exported_at": now(),
              "global_reference_frozen": False, "api_calls": 0, "database_open_mode": "ro",
              "database_state_sha256": state_sha(state), "summary": summary,
              "media_verified": media,
              "limits": ["Checks validate structure and provenance, not visual correctness or evidence quality.",
                         "Initial labels include explicit author batch judgments and assistant wording.",
                         "Checksums detect edits; they are not a signature or an E04 global freeze."]}
    write_json(folder / "database_rows.json", state)
    write_json(folder / "context.json", context)
    write_jsonl(folder / "initial_labels.jsonl", initial)
    write_jsonl(folder / "current_labels.jsonl", current)
    write_json(folder / "audit_report.json", report)
    # Re-read serialized bytes and reject writes made by another annotation window during export.
    require(read_database(db_path) == state, "导出期间人工库已变化；本目录未完成，需重新导出。")
    require(load_context() == context, "导出期间规则、映射或导出代码已变化；需重新导出。")
    require(validate_media(context) == media, "导出期间图片已变化；需重新导出。")
    write_json(folder / "checksums.json", {name: sha((folder / name).read_bytes())
                                           for name in sorted(EXPORT_NAMES)})
    verified = verify_export(folder)
    return {"output": str(folder), **verified}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-export", type=Path)
    args = parser.parse_args()
    try:
        require(not (args.verify_export and args.output), "核验现有导出时不能指定 --output。")
        result = verify_export(args.verify_export) if args.verify_export else export_annotations(args.db, args.output)
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f"E02 核验失败：{exc}\n")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
