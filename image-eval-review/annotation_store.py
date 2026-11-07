"""D01: local author labels, append-only history, Markdown import and snapshots."""

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from build_manifest import DATA, ROOT, read_jsonl, text_digest
from evaluate_results import expected_for_sample
from freeze_rules import load_frozen_context


PRIVATE = ROOT / "image-eval-review/private"
DEFAULT_DB = PRIVATE / "annotations.sqlite3"
DEFAULT_MARKDOWN = PRIVATE / "manual_annotation/人工初标记录表.md"
FIELDS = ("criterion_id", "status", "candidate_region", "candidate_observation", "reason")
STATUSES = ("pass", "fail", "unjudgeable")
ZH_STATUSES = {"符合": "pass", "不符": "fail", "不可判断": "unjudgeable"}


class AnnotationError(ValueError):
    pass


class ConflictError(AnnotationError):
    pass


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="microseconds")


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_markdown(text):
    """Accept this project's fixed six-column tables, including escaped <br>."""
    current = None
    result = {}
    for line in text.splitlines():
        match = re.fullmatch(r"## (sample_\d+)\s*", line)
        if match:
            current = match[1]
            if current in result:
                raise AnnotationError("记录中出现重复的样本章节。")
            result[current] = []
        elif line.startswith("## "):
            current = None
        if current and line.startswith("|"):
            match = re.search(r"`([a-z][a-z0-9_]+)`", line)
            if not match:
                continue
            cells = [cell.strip().replace(r"\|", "|")
                     for cell in re.split(r"(?<!\\)\|", line.strip())[1:-1]]
            if len(cells) != 6 or cells[1] not in ZH_STATUSES:
                raise AnnotationError("标注表存在未填状态或列数错误。")
            result[current].append({
                "criterion_id": match[1], "status": ZH_STATUSES[cells[1]],
                "candidate_region": cells[2], "candidate_observation": cells[3],
                "reason": cells[4],
            })
    if not result or any(not labels for labels in result.values()):
        raise AnnotationError("未找到完整的样本标注表。")
    return result


def _region(row):
    if row["id"] == "subject_under_eye_scar":
        return "候选对应眼下脸颊（按面部对应关系核对，不固定画面左右）"
    if row.get("check_regions"):
        return "、".join(row["check_regions"])
    if row.get("reference_evidence"):
        return "；".join(dict.fromkeys(item["region"] for item in row["reference_evidence"]))
    return {
        "photo_style": "整体光照、材质与空间", "figurine": "主体整体",
        "collectible_toy": "主体整体", "black_spiky_hair": "头发",
        "long_upright_tan_ears": "头顶双耳", "rabbit_ear_placement": "头顶及耳根",
        "dark_inner_ears": "头顶长耳内侧", "separate_face_ears": "头顶及脸侧耳部",
        "light_tan_skin": "脸部及可见皮肤", "large_anime_eyes": "面部眼睛",
        "red_jacket": "躯干与外套", "blue_shorts": "腰部至大腿",
        "sandals": "双脚及鞋面", "round_gold_base": "脚下底座",
    }.get(row["id"], "对应可见部位")


class AnnotationStore:
    def __init__(self, path=DEFAULT_DB):
        self.path = Path(path)
        self.policy, reference, evaluation, sample_rows = load_frozen_context()
        self.rule_version = self.policy["rule_version"]
        self.rule_digest = text_digest(DATA / "ruleset.json")
        self.manifest = {row["sample_id"]: row for row in read_jsonl(DATA / "manifest.jsonl")}
        self.smoke_ids = json.loads((DATA / "split.json").read_text(
            encoding="utf-8"))["smoke_selection"]["sample_ids"]
        self.criteria = {}
        self.expected = {}
        for sample in sample_rows:
            sid = sample["sample_id"]
            self.expected[sid] = expected_for_sample(sid, reference, evaluation, sample_rows)
            rows = []
            for dimension, source in (
                    ("subject_features", reference["criteria"]),
                    ("structural_integrity", evaluation["structure_criteria"]),
                    ("prompt_adherence", sample["prompt_criteria"])):
                for original in source:
                    # Whitelist display fields. Source filenames/config never reach the UI.
                    row = {
                        "id": original["id"],
                        "name": original.get("name", original.get("description", original["id"])),
                        "dimension": dimension, "required": original["required"],
                        "region": _region(original),
                        "pass_when": original["pass_when"], "fail_when": original["fail_when"],
                        "unjudgeable_when": original["unjudgeable_when"],
                        "allowed_variations": original.get("allowed_variations", []),
                        "not_required": original.get("not_required", []),
                        "boundary": original.get("boundary", ""),
                        "source_quote": original.get("source_quote", ""),
                    }
                    rows.append(row)
            self.criteria[sid] = rows
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS labels(
                    sample_id TEXT NOT NULL, criterion_id TEXT NOT NULL,
                    payload TEXT NOT NULL, annotator TEXT NOT NULL, revision INTEGER NOT NULL,
                    origin TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY(sample_id, criterion_id));
                CREATE TABLE IF NOT EXISTS history(
                    id INTEGER PRIMARY KEY, sample_id TEXT NOT NULL, criterion_id TEXT NOT NULL,
                    payload TEXT NOT NULL, annotator TEXT NOT NULL, revision INTEGER NOT NULL,
                    origin TEXT NOT NULL, change_reason TEXT NOT NULL, recorded_at TEXT NOT NULL,
                    UNIQUE(sample_id, criterion_id, revision));
                CREATE TABLE IF NOT EXISTS imports(
                    sha256 TEXT PRIMARY KEY, raw_markdown TEXT NOT NULL,
                    original_sha256 TEXT NOT NULL, original_markdown TEXT NOT NULL,
                    supplement_log TEXT, row_count INTEGER NOT NULL, imported_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS initial_snapshots(
                    sample_id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                    sha256 TEXT NOT NULL, confirmed_by TEXT NOT NULL, confirmed_at TEXT NOT NULL);
            """)
            binding = encode({"schema": 1, "rule_version": self.rule_version,
                              "ruleset_sha256": self.rule_digest})
            existing = db.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()
            if existing and existing[0] != binding:
                raise AnnotationError("标注库与当前规则版本不一致，请使用对应版本的标注库。")
            db.execute("INSERT OR IGNORE INTO metadata VALUES('binding', ?)", (binding,))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def validate(self, sid, label, allow_draft=True):
        if sid not in self.expected:
            raise AnnotationError("样本编号未登记。")
        if not isinstance(label, dict) or set(label) != set(FIELDS):
            raise AnnotationError("标签字段不完整或含额外字段。")
        if any(not isinstance(label[key], str) or len(label[key]) > 10000 for key in FIELDS):
            raise AnnotationError("标签字段必须为字符串，且每项不超过 10000 字符。")
        if label["criterion_id"] not in self.expected[sid]:
            raise AnnotationError("该条目不属于当前样本。")
        if label["status"] not in ((*STATUSES, "") if allow_draft else STATUSES):
            raise AnnotationError("请选择符合、不符或不可判断。")
        if not allow_draft and not self.complete(label):
            raise AnnotationError("该项的状态、检查部位、实际观察和判断原因须填写完整。")

    @staticmethod
    def complete(label):
        return label["status"] in STATUSES and all(
            label[key].strip() for key in ("candidate_region", "candidate_observation", "reason"))

    @staticmethod
    def _text(value, title):
        if not isinstance(value, str) or not value.strip() or len(value) > 200:
            raise AnnotationError(f"{title}须为 1–200 字的非空文字。")
        return value.strip()

    def _write(self, db, sid, label, annotator, revision, origin, change_reason):
        timestamp = now()
        params = (sid, label["criterion_id"], encode(label), annotator, revision,
                  encode(origin), timestamp)
        db.execute("""INSERT INTO labels VALUES(?,?,?,?,?,?,?)
                      ON CONFLICT(sample_id,criterion_id) DO UPDATE SET
                      payload=excluded.payload, annotator=excluded.annotator,
                      revision=excluded.revision, origin=excluded.origin,
                      updated_at=excluded.updated_at""", params)
        db.execute("""INSERT INTO history(sample_id,criterion_id,payload,annotator,revision,
                      origin,change_reason,recorded_at) VALUES(?,?,?,?,?,?,?,?)""",
                   (*params[:-1], change_reason, timestamp))

    def save(self, sid, label, annotator, expected_revision, change_reason=""):
        self.validate(sid, label)
        annotator = self._text(annotator, "标注人")
        if type(expected_revision) is not int or expected_revision < 0:
            raise AnnotationError("版本号非法，请重新加载记录。")
        if not isinstance(change_reason, str) or len(change_reason) > 2000:
            raise AnnotationError("修改原因须为不超过 2000 字的文字。")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                "SELECT * FROM labels WHERE sample_id=? AND criterion_id=?",
                (sid, label["criterion_id"])).fetchone()
            revision = current["revision"] if current else 0
            if revision != expected_revision:
                raise ConflictError("此项已在其他页面修改，请重新加载后再保存。")
            if current and json.loads(current["payload"]) == label and current["annotator"] == annotator:
                return revision
            initial = db.execute(
                "SELECT 1 FROM initial_snapshots WHERE sample_id=?", (sid,)).fetchone()
            if initial and not change_reason.strip():
                raise AnnotationError("本图已有初始快照，修改时请填写原因。")
            self._write(db, sid, label, annotator, revision + 1,
                        {"kind": "author_page_edit", "after_initial_confirmation": bool(initial)},
                        change_reason.strip() or "在标注页保存")
            return revision + 1

    def _records(self, db, sid):
        return {
            row["criterion_id"]: {
                "label": json.loads(row["payload"]), "annotator": row["annotator"],
                "revision": row["revision"], "origin": json.loads(row["origin"]),
                "updated_at": row["updated_at"],
            } for row in db.execute("SELECT * FROM labels WHERE sample_id=?", (sid,))}

    def _progress(self, db, sid):
        records = self._records(db, sid)
        return {"sample_id": sid, "complete": sum(
            self.complete(row["label"]) for row in records.values()),
            "saved": len(records), "total": len(self.expected[sid])}

    def overview(self):
        with self.connect() as db:
            progress = {sid: self._progress(db, sid) for sid in self.manifest}
            initials = {row["sample_id"] for row in db.execute(
                "SELECT sample_id FROM initial_snapshots")}
        return {
            "rule_version": self.rule_version, "default_annotator": "水水",
            "smoke_ids": list(self.smoke_ids),
            "samples": [{**progress[sid], "split": row["split"],
                         "confirmed": sid in initials} for sid, row in self.manifest.items()],
        }

    def sample(self, sid):
        if sid not in self.manifest:
            raise AnnotationError("样本编号未登记。")
        with self.connect() as db:
            saved = self._records(db, sid)
            initial = db.execute(
                "SELECT sha256,confirmed_by,confirmed_at FROM initial_snapshots WHERE sample_id=?",
                (sid,)).fetchone()
            progress = self._progress(db, sid)
        return {
            "sample_id": sid, "prompt": self.manifest[sid]["prompt"],
            "rule_version": self.rule_version, "progress": progress,
            "initial": dict(initial) if initial else None,
            "criteria": self.criteria[sid],
            "records": saved,
        }

    def confirm_initial(self, sid, annotator):
        if sid not in self.manifest:
            raise AnnotationError("样本编号未登记。")
        annotator = self._text(annotator, "确认人")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT sha256 FROM initial_snapshots WHERE sample_id=?", (sid,)).fetchone()
            if existing:
                return existing["sha256"]
            records = self._records(db, sid)
            if set(records) != set(self.expected[sid]) or not all(
                    self.complete(row["label"]) for row in records.values()):
                raise AnnotationError("请先补齐并保存本图全部条目，再确认初始记录。")
            payload = encode({
                "sample_id": sid, "rule_version": self.rule_version,
                "ruleset_sha256": self.rule_digest,
                "candidate_sha256": self.manifest[sid]["candidate_sha256"],
                "reference_sha256": [row["sha256"] for row in json.loads(
                    (DATA / "reference_assets.json").read_text(encoding="utf-8"))],
                "records": records,
            })
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            db.execute("INSERT INTO initial_snapshots VALUES(?,?,?,?,?)",
                       (sid, payload, digest, annotator, now()))
        return digest

    def export(self, sid):
        self.sample(sid)  # Validate the sample before constructing an export.
        with self.connect() as db:
            snapshot = db.execute(
                "SELECT * FROM initial_snapshots WHERE sample_id=?", (sid,)).fetchone()
            history = [dict(row) for row in db.execute(
                "SELECT * FROM history WHERE sample_id=? ORDER BY id", (sid,))]
        for row in history:
            row["payload"] = json.loads(row["payload"])
            row["origin"] = json.loads(row["origin"])
        return {
            "sample": self.sample(sid), "history": history,
            "initial_snapshot": {**dict(snapshot), "payload": json.loads(snapshot["payload"])}
            if snapshot else None,
        }

    def import_markdown(self, path, annotator="水水", supplement_path=None):
        """Atomic, idempotent import; never replace existing page edits."""
        raw = Path(path).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8")
        imported = parse_markdown(text)
        original_text, original_sha, supplement = text, sha, None
        originals = imported
        if supplement_path:
            supplement_path = Path(supplement_path)
            supplement = json.loads(supplement_path.read_text(encoding="utf-8"))
            if supplement["supplemented_sha256"] != sha:
                raise AnnotationError("填写表与补写日志哈希不一致，未执行导入。")
            # The snapshot must be a sibling, not an arbitrary path from metadata.
            name = supplement["snapshot"]
            if Path(name).name != name:
                raise AnnotationError("补写快照文件名非法。")
            original_raw = (supplement_path.parent / name).read_bytes()
            original_sha = hashlib.sha256(original_raw).hexdigest()
            if original_sha != supplement["source_sha256"]:
                raise AnnotationError("原始快照哈希不一致，未执行导入。")
            original_text = original_raw.decode("utf-8")
            originals = parse_markdown(original_text)
        annotator = self._text(annotator, "标注人")
        if set(imported) != set(originals):
            raise AnnotationError("原始记录与补写记录的样本集合不一致。")
        for sid, labels in imported.items():
            for label in labels:
                self.validate(sid, label, allow_draft=False)
            for label in originals[sid]:
                self.validate(sid, label, allow_draft=False)
            ids = [row["criterion_id"] for row in labels]
            old_ids = [row["criterion_id"] for row in originals[sid]]
            if (len(ids) != len(set(ids)) or set(ids) != set(self.expected[sid])
                    or len(old_ids) != len(set(old_ids)) or set(old_ids) != set(ids)):
                raise AnnotationError("导入条目有遗漏、重复或未知编号。")
            old_status = {row["criterion_id"]: row["status"] for row in originals[sid]}
            if any(row["status"] != old_status[row["criterion_id"]] for row in labels):
                raise AnnotationError("补写记录改变了作者判断，未执行导入。")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT row_count FROM imports WHERE sha256=?", (sha,)).fetchone()
            if existing:
                return {"imported": 0, "already_imported": True, "source_count": existing[0]}
            for sid in imported:
                if db.execute("SELECT 1 FROM labels WHERE sample_id=? LIMIT 1", (sid,)).fetchone():
                    raise ConflictError("样本已有页面记录，禁止导入覆盖。")
            count = 0
            for sid, labels in imported.items():
                original_map = {row["criterion_id"]: row for row in originals[sid]}
                for label in labels:
                    original = original_map[label["criterion_id"]]
                    self._write(db, sid, original, annotator, 1, {
                        "kind": "author_markdown_original", "source_sha256": original_sha,
                        "historical_time": "未记录", "imported_at": now(),
                    }, "保留原始人工填写")
                    if supplement:
                        self._write(db, sid, label, annotator, 2, {
                            "kind": "author_status_codex_evidence", "source_sha256": sha,
                            "original_sha256": original_sha, "evidence_editor": "Codex",
                            "supplement_date": supplement["supplement_date"],
                        }, "用户授权补写文字，判断未改变")
                    count += 1
            db.execute("INSERT INTO imports VALUES(?,?,?,?,?,?,?)", (
                sha, text, original_sha, original_text, encode(supplement) if supplement else None,
                count, now()))
        return {"imported": count, "already_imported": False, "source_count": count}
