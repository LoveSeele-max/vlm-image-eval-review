"""D03 durable batches, tasks, attempts, cache references and request reservations."""

import copy
import hashlib
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import judge_client as client
from api_connection import normalize_base_url


DEFAULT_DB = client.PRIVATE / "runs.sqlite3"
REQUEST_LIMIT = 100


class RunError(ValueError):
    pass


def identity(prepared, base_url, timeout):
    """Include exact model inputs, trusted rules, preprocessing and engine code."""
    metadata = {key: value for key, value in prepared["metadata"].items()
                if key != "initial_snapshot_sha256"}
    value = {
        "version": 1, "endpoint": normalize_base_url(base_url) + "/responses",
        "timeout_seconds": float(timeout), "body": prepared["body"],
        "expected": prepared["expected"], "policy": prepared["policy"], "metadata": metadata,
        "engine_sources": {name: client.text_digest(Path(__file__).with_name(name))
                           for name in ("run_store.py", "run_tasks.py", "judge_client.py",
                                        "prepare_image.py", "evaluate_results.py")},
    }
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


@contextmanager
def worker_lock(database):
    """One worker per DB. The OS releases this lock if the process dies."""
    path = Path(database).resolve().with_suffix(".worker.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RunError("已有进程正在运行或恢复任务，请勿同时启动。") from None
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class RunStore:
    def __init__(self, database=DEFAULT_DB):
        self.path = Path(database).resolve()
        self.root = self.path.with_suffix("")
        self.inputs = self.root / "inputs"
        self.evidence = self.root / "evidence"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.inputs.mkdir(parents=True, exist_ok=True)
        self.evidence.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS batches(
                    id TEXT PRIMARY KEY, created_at REAL NOT NULL, status TEXT NOT NULL,
                    stop_code TEXT, repeat INTEGER NOT NULL, max_attempts INTEGER NOT NULL,
                    timeout REAL NOT NULL, base_url TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS tasks(
                    id TEXT PRIMARY KEY, batch_id TEXT NOT NULL REFERENCES batches(id),
                    sample_id TEXT NOT NULL, cache_key TEXT NOT NULL,
                    input_path TEXT NOT NULL, input_sha256 TEXT NOT NULL,
                    spec_json TEXT NOT NULL, status TEXT NOT NULL,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    next_attempt_at REAL NOT NULL DEFAULT 0,
                    stop_code TEXT, result_attempt_id TEXT, cached_from TEXT,
                    initial_snapshot_sha256 TEXT, UNIQUE(batch_id,sample_id));
                CREATE TABLE IF NOT EXISTS attempts(
                    id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(id),
                    number INTEGER NOT NULL, status TEXT NOT NULL,
                    started_at REAL NOT NULL, finished_at REAL, elapsed_seconds REAL,
                    error_code TEXT, decision TEXT, http_status INTEGER,
                    requested_model TEXT NOT NULL, returned_model TEXT, usage_json TEXT,
                    evidence_dir TEXT NOT NULL, receipt_json TEXT, artifact_hashes_json TEXT,
                    budget_reserved INTEGER NOT NULL, request_started INTEGER,
                    UNIQUE(task_id,number));
                CREATE TABLE IF NOT EXISTS cache(
                    cache_key TEXT PRIMARY KEY, source_attempt_id TEXT NOT NULL REFERENCES attempts(id));
                CREATE TABLE IF NOT EXISTS events(
                    id INTEGER PRIMARY KEY, batch_id TEXT NOT NULL, task_id TEXT,
                    created_at REAL NOT NULL, kind TEXT NOT NULL, detail_json TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS tasks_batch ON tasks(batch_id,status);
            """)
            expected = {"schema_version": "1", "request_limit": str(REQUEST_LIMIT)}
            for key, value in expected.items():
                existing = db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
                if existing and existing[0] != value:
                    raise RunError("运行库版本或首轮请求限额不匹配，不能静默重置。")
                db.execute("INSERT OR IGNORE INTO metadata VALUES(?,?)", (key, value))

    @contextmanager
    def connect(self, immediate=False):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                if immediate:
                    db.execute("BEGIN IMMEDIATE")
                yield db
        finally:
            db.close()

    @staticmethod
    def event(db, batch_id, kind, task_id=None, detail=None):
        db.execute("INSERT INTO events(batch_id,task_id,created_at,kind,detail_json) VALUES(?,?,?,?,?)",
                   (batch_id, task_id, time.time(), kind, json.dumps(detail or {}, ensure_ascii=False)))

    def create(self, requests, base_url, *, timeout=180, max_attempts=3, repeat=False):
        base_url = normalize_base_url(base_url)
        if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
            raise RunError("每任务总尝试次数须为 1–3。")
        if type(timeout) not in (int, float) or not 0 < timeout <= 600:
            raise RunError("超时须大于 0 且不超过 600 秒。")
        ids = [row["metadata"]["sample_id"] for row in requests]
        if not requests or len(set(ids)) != len(ids):
            raise RunError("批次样本不能为空或重复。")
        batch_id = "batch-" + uuid.uuid4().hex
        stored = []
        for source in requests:
            prepared = copy.deepcopy(source)
            prepared["metadata"].pop("initial_snapshot_sha256", None)
            if prepared["metadata"]["split"] != "development":
                raise RunError("当前运行入口只开放开发集。")
            key = identity(prepared, base_url, timeout)
            path = self.inputs / (key + ".json")
            raw = client.encoded(prepared) + b"\n"
            if path.exists() and path.read_bytes() != raw:
                raise RunError("输入存档发生冲突，未覆盖。")
            if not path.exists():
                client.write_json(path, prepared)
            stored.append((prepared, key, path, client.digest(raw)))
        timestamp = time.time()
        with self.connect(immediate=True) as db:
            db.execute("INSERT INTO batches VALUES(?,?,?,?,?,?,?,?)",
                       (batch_id, timestamp, "pending", None, int(repeat),
                        max_attempts, timeout, base_url))
            for prepared, key, path, sha in stored:
                spec = {"metadata": prepared["metadata"], "model": prepared["body"]["model"],
                        "max_output_tokens": prepared["body"]["max_output_tokens"]}
                db.execute("""INSERT INTO tasks(
                    id,batch_id,sample_id,cache_key,input_path,input_sha256,spec_json,
                    status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                           ("task-" + uuid.uuid4().hex, batch_id,
                            prepared["metadata"]["sample_id"], key, str(path), sha,
                            json.dumps(spec, ensure_ascii=False), "pending", timestamp, timestamp))
            self.event(db, batch_id, "batch_created", detail={"sample_count": len(stored),
                                                            "repeat": bool(repeat)})
        return batch_id

    def batch(self, batch_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
        if not row:
            raise RunError("批次编号不存在。")
        return dict(row)

    def tasks(self, batch_id):
        self.batch(batch_id)
        with self.connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM tasks WHERE batch_id=? ORDER BY rowid", (batch_id,))]

    def stop(self, batch_id, code):
        with self.connect(immediate=True) as db:
            db.execute("UPDATE batches SET status='stopped',stop_code=? WHERE id=?", (code, batch_id))
            self.event(db, batch_id, "batch_stopped", detail={"code": code})

    def reserve(self, task, batch):
        """Reserve the budget transactionally before a request may leave the host."""
        with self.connect(immediate=True) as db:
            current = db.execute("SELECT status FROM tasks WHERE id=?", (task["id"],)).fetchone()
            if not current or current[0] != "pending":
                raise RunError("任务已被领取或状态发生变化。")
            number = db.execute("SELECT count(*) FROM attempts WHERE task_id=?",
                                (task["id"],)).fetchone()[0] + 1
            if number > batch["max_attempts"]:
                raise RunError("任务已达到尝试次数上限。")
            used = db.execute("SELECT COALESCE(sum(budget_reserved),0) FROM attempts").fetchone()[0]
            if used >= REQUEST_LIMIT:
                return None
            attempt_id = "attempt-" + uuid.uuid4().hex
            path = self.evidence / attempt_id
            db.execute("""INSERT INTO attempts(
                id,task_id,number,status,started_at,requested_model,evidence_dir,budget_reserved)
                VALUES(?,?,?,?,?,?,?,1)""",
                       (attempt_id, task["id"], number, "running", time.time(),
                        json.loads(task["spec_json"])["model"], str(path)))
            db.execute("UPDATE tasks SET status='running',updated_at=?,stop_code=NULL WHERE id=?",
                       (time.time(), task["id"]))
            self.event(db, batch["id"], "attempt_reserved", task["id"],
                       {"attempt_id": attempt_id, "number": number})
        return attempt_id, number

    def finish(self, task, attempt_id, receipt, hashes, decision, *, next_at=0, recovered=False):
        """Success may mean a visual fail; only engineering validity controls cache."""
        valid = receipt.get("engineering_status") == "valid_result" and not receipt.get("error_code")
        status = "succeeded" if valid else "failed"
        task_status = "succeeded" if valid else "pending" if decision == "retry" else "failed"
        with self.connect(immediate=True) as db:
            attempt = db.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone()
            if attempt["status"] != "running":
                raise RunError("尝试已经结束，禁止重复覆盖。")
            db.execute("""UPDATE attempts SET status=?,finished_at=?,elapsed_seconds=?,
                error_code=?,decision=?,http_status=?,returned_model=?,usage_json=?,
                receipt_json=?,artifact_hashes_json=?,request_started=?,budget_reserved=? WHERE id=?""",
                       (status, time.time(), receipt.get("elapsed_seconds"), receipt.get("error_code"),
                        decision, receipt.get("http_status"),
                        receipt.get("returned_model") if isinstance(receipt.get("returned_model"), str) else None,
                        json.dumps(receipt.get("usage")), json.dumps(receipt, ensure_ascii=False),
                        json.dumps(hashes), int(bool(receipt.get("request_started"))),
                        int(bool(receipt.get("request_started"))), attempt_id))
            db.execute("""UPDATE tasks SET status=?,updated_at=?,next_attempt_at=?,stop_code=?,
                result_attempt_id=?,cached_from=NULL WHERE id=?""",
                       (task_status, time.time(), next_at, receipt.get("error_code"),
                        attempt_id if valid else None, task["id"]))
            if valid:
                db.execute("""INSERT INTO cache VALUES(?,?) ON CONFLICT(cache_key)
                    DO UPDATE SET source_attempt_id=excluded.source_attempt_id""",
                           (task["cache_key"], attempt_id))
            self.event(db, task["batch_id"], "attempt_recovered" if recovered else "attempt_finished",
                       task["id"], {"attempt_id": attempt_id, "decision": decision})

    def interrupt(self, task, attempt, *, definitely_unsent=False, retry=True):
        with self.connect(immediate=True) as db:
            db.execute("""UPDATE attempts SET status='interrupted',finished_at=?,
                error_code='interrupted',decision=?,request_started=?,budget_reserved=? WHERE id=?""",
                       (time.time(), "retry" if retry else "stop_task",
                        0 if definitely_unsent else None, 0 if definitely_unsent else 1, attempt["id"]))
            db.execute("""UPDATE tasks SET status=?,stop_code='interrupted',
                next_attempt_at=0,updated_at=? WHERE id=?""",
                       ("pending" if retry else "failed", time.time(), task["id"]))
            self.event(db, task["batch_id"], "attempt_interrupted", task["id"],
                       {"attempt_id": attempt["id"], "request_may_have_been_sent": not definitely_unsent})

    def status(self, batch_id=None):
        with self.connect() as db:
            batches = [dict(row) for row in db.execute(
                "SELECT * FROM batches" + (" WHERE id=?" if batch_id else "") + " ORDER BY created_at",
                (batch_id,) if batch_id else ())]
            if batch_id and not batches:
                raise RunError("批次编号不存在。")
            result = []
            for batch in batches:
                tasks = []
                for task in db.execute("SELECT * FROM tasks WHERE batch_id=? ORDER BY rowid",
                                       (batch["id"],)):
                    attempts = [dict(row) for row in db.execute("""SELECT id,number,status,
                        error_code,decision,http_status,elapsed_seconds,request_started,budget_reserved,
                        requested_model,returned_model,usage_json,evidence_dir
                        FROM attempts WHERE task_id=? ORDER BY number""", (task["id"],))]
                    tasks.append({key: task[key] for key in (
                        "id", "sample_id", "status", "stop_code", "next_attempt_at",
                        "cache_key", "cached_from", "result_attempt_id")})
                    tasks[-1]["attempts"] = attempts
                    spec = json.loads(task["spec_json"])
                    tasks[-1]["requested_model"] = spec["model"]
                    tasks[-1]["max_output_tokens"] = spec["max_output_tokens"]
                    tasks[-1]["versions"] = {key: spec["metadata"][key] for key in (
                        "rule_version", "prompt_version", "ruleset_sha256",
                        "prompt_sha256", "request_body_sha256")}
                    tasks[-1]["retry_count"] = max(0, len(attempts) - 1)
                    tasks[-1]["elapsed_seconds"] = sum(row["elapsed_seconds"] or 0 for row in attempts)
                result.append({"id": batch["id"], "status": batch["status"],
                               "stop_code": batch["stop_code"], "repeat": bool(batch["repeat"]),
                               "max_attempts": batch["max_attempts"],
                               "timeout_seconds": batch["timeout"], "tasks": tasks})
            budget = dict(db.execute("""SELECT COALESCE(sum(budget_reserved),0) AS reserved,
                COALESCE(sum(CASE WHEN request_started=1 THEN 1 ELSE 0 END),0) AS known_started,
                COALESCE(sum(CASE WHEN request_started IS NULL AND budget_reserved=1
                                 THEN 1 ELSE 0 END),0) AS uncertain_or_running FROM attempts""").fetchone())
        return {"request_limit": REQUEST_LIMIT, "budget": budget, "batches": result}
