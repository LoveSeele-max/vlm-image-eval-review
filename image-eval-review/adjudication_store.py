"""G03 author-confirmed decisions in a separate SQLite store, with append-only history."""
import copy
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import judge_client as client
from annotation_store import PRIVATE, encode, now, FIELDS, STATUSES, ConflictError


DEFAULT_DB = PRIVATE / "adjudications.sqlite3"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(value, name, limit=2000):
    require(isinstance(value, str) and 0 < len(value.strip()) <= limit, f"请填写{name}（最多 {limit} 字）。")
    return value.strip()


class AdjudicationStore:
    def __init__(self, binding, rows, database=DEFAULT_DB):
        self.path = Path(database).resolve()
        require(self.path.name.lower() not in ("annotations.sqlite3", "runs.sqlite3"), "复核库必须独立于初标和运行库。")
        self.binding = copy.deepcopy(binding)
        self.binding_sha = client.digest(encode(binding).encode())
        self.rows = {(r["sample_id"], r["criterion_id"]): copy.deepcopy(r) for r in rows}
        require(len(self.rows) == len(rows) and bool(rows), "复核条目必须非空且唯一。")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS review_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS review_history(
                    sample_id TEXT NOT NULL,criterion_id TEXT NOT NULL,revision INTEGER NOT NULL,
                    payload TEXT NOT NULL,sha256 TEXT NOT NULL,
                    PRIMARY KEY(sample_id,criterion_id,revision));
                CREATE TABLE IF NOT EXISTS review_current(
                    sample_id TEXT NOT NULL,criterion_id TEXT NOT NULL,revision INTEGER NOT NULL,
                    payload TEXT NOT NULL,sha256 TEXT NOT NULL,PRIMARY KEY(sample_id,criterion_id));
            """)
            for key, value in (("schema_version", "1"), ("binding", encode(binding))):
                old = db.execute("SELECT value FROM review_meta WHERE key=?", (key,)).fetchone()
                require(old is None or old[0] == value, "复核库绑定其他冻结实验，不能覆盖。")
                db.execute("INSERT OR IGNORE INTO review_meta VALUES(?,?)", (key, value))
        self.snapshot()

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def trusted(self, sid, cid):
        require((sid, cid) in self.rows, "仅允许已核验的保留集条目。")
        return self.rows[sid, cid]

    @staticmethod
    def row_digest(row):
        return client.digest(encode({k: row[k] for k in ("sample_id", "criterion_id", "human", "v1", "v2")}).encode())

    def _history(self, db):
        history, latest, previous = [], {}, {}
        for entry in db.execute("SELECT * FROM review_history ORDER BY sample_id,criterion_id,revision"):
            row = dict(entry)
            key = (row["sample_id"], row["criterion_id"])
            trusted = self.trusted(*key)
            require(client.digest(row["payload"].encode()) == row["sha256"], "复核历史哈希不符。")
            payload = json.loads(row["payload"])
            require(payload["revision"] == len(previous.get(key, [])) + 1 == row["revision"] and
                    payload["sample_id"] == key[0] and payload["criterion_id"] == key[1] and
                    payload["binding_sha256"] == self.binding_sha and
                    payload["source_row_sha256"] == self.row_digest(trusted) and
                    payload["confirmed_by_author"] is True and
                    payload["frozen_label"] == {"criterion_id": key[1], **trusted["human"]} and
                    payload["previous_sha256"] == (latest[key]["sha256"] if key in latest else None),
                    "复核历史链或冻结来源不符。")
            previous.setdefault(key, []).append(payload)
            latest[key] = row
            history.append(payload)
        current = {(r["sample_id"], r["criterion_id"]): dict(r) for r in db.execute("SELECT * FROM review_current")}
        require(current == latest, "复核当前值与历史链不一致。")
        return {key: json.loads(value["payload"]) for key, value in current.items()}, history

    def snapshot(self):
        with self.connect() as db:
            db.execute("BEGIN")
            return self._history(db)

    def save(self, sid, cid, *, decision, final_label, review_reason, annotator, expected_revision, confirmed):
        row = self.trusted(sid, cid)
        require(confirmed is True, "请先勾选已核对图片与固定规则，再确认保存。")
        require(decision in ("keep", "revise", "unresolved"), "请选择复核结论。")
        require(type(expected_revision) is int and expected_revision >= 0, "复核版本号非法。")
        annotator = text(annotator, "确认人", 100)
        review_reason = text(review_reason, "复核依据与处理理由")
        frozen = {"criterion_id": cid, **row["human"]}
        if decision == "keep":
            require(final_label is None, "维持初标不接受替换标签。")
            reviewed = copy.deepcopy(frozen)
        elif decision == "revise":
            require(isinstance(final_label, dict) and set(final_label) == set(FIELDS) and
                    final_label["criterion_id"] == cid and final_label["status"] in STATUSES, "修订标签格式非法。")
            reviewed = {k: text(final_label[k], "修订证据") for k in FIELDS}
            require(reviewed != frozen, "修订内容与初标相同，请选择维持初标。")
        else:
            require(final_label is None, "暂不能裁定不生成最终标签。")
            reviewed = None
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current, _ = self._history(db)
            old = current.get((sid, cid))
            if expected_revision != (old["revision"] if old else 0):
                raise ConflictError("此项已在其他页面更新，请重新加载后核对；当前表单尚未保存。")
            if old and all(old[k] == value for k, value in (("decision", decision), ("final_label", reviewed),
                    ("review_reason", review_reason), ("annotator", annotator))):
                return old
            previous = db.execute("SELECT sha256 FROM review_current WHERE sample_id=? AND criterion_id=?", (sid, cid)).fetchone()
            record = {"sample_id": sid, "criterion_id": cid, "revision": expected_revision+1,
                "decision": decision, "frozen_label": frozen, "final_label": reviewed,
                "review_reason": review_reason, "annotator": annotator, "confirmed_by_author": True,
                "confirmed_at": now(), "binding_sha256": self.binding_sha,
                "source_row_sha256": self.row_digest(row), "previous_sha256": previous[0] if previous else None}
            payload = encode(record)
            values = (sid, cid, record["revision"], payload, client.digest(payload.encode()))
            db.execute("INSERT INTO review_history VALUES(?,?,?,?,?)", values)
            db.execute("""INSERT INTO review_current VALUES(?,?,?,?,?) ON CONFLICT(sample_id,criterion_id)
                       DO UPDATE SET revision=excluded.revision,payload=excluded.payload,sha256=excluded.sha256""", values)
        return record

    def export(self):
        current, history = self.snapshot()
        return {"schema_version": 1, "task": "G03", "exported_at": now(), "binding": self.binding,
                "current_reviews": list(current.values()), "history": history,
                "resolved_items": sum(r["decision"] in ("keep", "revise") for r in current.values()),
                "unresolved_items": sum(r["decision"] == "unresolved" for r in current.values()),
                "primary_metrics_reference": "human_reference_v1 remains unchanged; revisions are separate."}
