"""D03 serial worker: durable records, verified cache, bounded retries and recovery."""

import argparse
import copy
import json
import os
import random
import sqlite3
import sys
import time
from pathlib import Path

import judge_client as client
from evaluate_results import evaluate_result
from run_store import DEFAULT_DB, RunError, RunStore, identity, worker_lock


ARTIFACTS = ("request.json", "request_meta.json", "run.json", "provider.json",
             "answer.txt", "parsed.json", "validation.json")
RETRY_CODES = {"transport_error", "incomplete_http_body", "http_408", "http_409",
               "http_429", "http_500", "http_502", "http_503", "http_504",
               "invalid_json", "duplicate_json_key", "nonfinite_json_number"}
CONFIG_CODES = {"http_400", "http_401", "http_403", "http_404", "http_405", "http_422"}
QUOTA_CODES = {"insufficient_quota", "billing_hard_limit_reached", "credits_exhausted",
               "quota_exceeded", "account_deactivated"}


def decision(receipt, provider):
    if receipt.get("engineering_status") == "valid_result" and not receipt.get("error_code"):
        return "success"
    error = provider.get("error") if isinstance(provider, dict) else None
    provider_codes = {str(error.get(key, "")).lower() for key in ("code", "type")} if isinstance(error, dict) else set()
    if provider_codes & QUOTA_CODES or receipt.get("error_code") in CONFIG_CODES:
        return "stop_batch"
    if receipt.get("error_code") in RETRY_CODES or receipt.get("engineering_status") == "invalid_result":
        return "retry"
    if receipt.get("error_code") == "provider_error" and provider_codes & {
            "server_error", "rate_limit_exceeded", "temporarily_unavailable"}:
        return "retry"
    return "stop_task"


class Worker:
    def __init__(self, store, *, executor=None, snapshot_checker=None, clock=None, sleep=None):
        self.store = store
        self.executor = executor or client.execute_request
        self.snapshot_checker = snapshot_checker or client.require_initial_snapshot
        self.clock = clock or time.time
        self.sleep = sleep or time.sleep

    def inputs(self, task, batch):
        path = Path(task["input_path"]).resolve()
        if not path.is_relative_to(self.store.inputs.resolve()):
            raise RunError("输入存档位置非法。")
        raw = path.read_bytes()
        if client.digest(raw) != task["input_sha256"]:
            raise RunError("输入存档哈希变化，不能继续。")
        prepared = client.strict_json(raw)
        if identity(prepared, batch["base_url"], batch["timeout"]) != task["cache_key"]:
            raise RunError("输入或执行代码版本发生变化，请新建批次。")
        # Rebuild locally to check current frozen rules AND actual source bytes.
        current = client.build_request(task["sample_id"], prepared["body"]["model"],
                                       prepared["body"]["max_output_tokens"])
        if current != prepared:
            raise RunError("素材、规则或预处理结果发生变化，请新建批次。")
        return prepared

    def evidence(self, attempt, prepared, batch):
        folder = Path(attempt["evidence_dir"]).resolve()
        if folder != (self.store.evidence / attempt["id"]).resolve():
            raise RunError("尝试证据位置非法。")
        hashes = {name: client.digest((folder / name).read_bytes()) for name in ARTIFACTS}
        if (folder / "raw_response.bin").exists():
            hashes["raw_response.bin"] = client.digest((folder / "raw_response.bin").read_bytes())
        if attempt.get("artifact_hashes_json") and hashes != json.loads(attempt["artifact_hashes_json"]):
            raise RunError("成功证据缺失或哈希变化，不能复用。")
        receipt = client.strict_json((folder / "run.json").read_bytes())
        request = client.strict_json((folder / "request.json").read_bytes())
        metadata = client.strict_json((folder / "request_meta.json").read_bytes())
        metadata.pop("initial_snapshot_sha256", None)
        if (not receipt.get("completed_at") or receipt.get("run_id") != attempt["id"]
                or receipt.get("mode") != "single_request"
                or receipt.get("request_started") is not True
                or receipt.get("attempt") != attempt["number"]
                or receipt.get("endpoint") != batch["base_url"] + "/responses"
                or receipt.get("requested_model") != prepared["body"]["model"]
                or receipt.get("request_body_sha256") != prepared["metadata"]["request_body_sha256"]
                or request != prepared["body"] or metadata != prepared["metadata"]):
            raise RunError("尝试证据未完整结束，或与原始请求不一致。")
        raw = (folder / "raw_response.bin").read_bytes() if "raw_response.bin" in hashes else None
        if raw is not None and client.digest(raw) != receipt.get("saved_response_sha256"):
            raise RunError("响应正文哈希不一致。")
        if raw is None and receipt.get("saved_response_sha256"):
            raise RunError("响应正文缺失。")
        if type(receipt.get("http_status")) is int and 200 <= receipt["http_status"] < 300 and not receipt.get("response_truncated") and receipt.get(
                "error_code") not in {"transport_error", "incomplete_http_body"}:
            _, _, parsed, report = client.validate_response(raw, prepared)
            if receipt.get("engineering_status") == "valid_result":
                if parsed != client.strict_json((folder / "parsed.json").read_bytes()):
                    raise RunError("解析结果与响应不一致。")
        else:
            report = evaluate_result(prepared["metadata"]["sample_id"], None, prepared["expected"],
                                     prepared["policy"], task_status="failed")
            report["validation_errors"] = [{"code": receipt.get("error_code")}]
        if (report != client.strict_json((folder / "validation.json").read_bytes())
                or receipt.get("engineering_status") != report["engineering_status"]
                or receipt.get("candidate_status") != report["candidate_status"]):
            raise RunError("校验报告与响应不一致。")
        provider = client.strict_json((folder / "provider.json").read_bytes())
        return receipt, hashes, provider

    def finish(self, task, attempt, prepared, batch, *, recovered=False):
        receipt, hashes, provider = self.evidence(attempt, prepared, batch)
        action = decision(receipt, provider)
        if action == "retry" and attempt["number"] >= batch["max_attempts"]:
            action = "stop_task"
        retry_after = receipt.get("retry_after_seconds")
        retry_after = retry_after if type(retry_after) in (int, float) and 0 <= retry_after <= 86400 else 0
        delay = max(min(30, 2 ** attempt["number"] + random.uniform(0, 0.25)), retry_after)
        self.store.finish(task, attempt["id"], receipt, hashes, action,
                          next_at=self.clock() + delay if action == "retry" else 0,
                          recovered=recovered)
        if action == "stop_batch":
            self.store.stop(task["batch_id"], receipt.get("error_code") or "provider_configuration_error")
        return action

    @staticmethod
    def definitely_unsent(attempt):
        folder = Path(attempt["evidence_dir"])
        if not folder.exists():
            return True  # D02 creates this directory before it can send.
        try:
            receipt = client.strict_json((folder / "run.json").read_bytes())
            return receipt.get("request_started") is False
        except (OSError, ValueError, AttributeError):
            return False

    def recover_unlocked(self, batch_id):
        batch = self.store.batch(batch_id)
        for task in self.store.tasks(batch_id):
            if task["status"] != "running":
                continue
            with self.store.connect() as db:
                attempt = dict(db.execute(
                    "SELECT * FROM attempts WHERE task_id=? ORDER BY number DESC LIMIT 1",
                    (task["id"],)).fetchone())
            try:
                prepared = self.inputs(task, batch)
            except (OSError, ValueError, KeyError, TypeError):
                self.store.stop(batch_id, "stale_or_corrupt_input")
                return
            try:
                self.finish(task, attempt, prepared, batch, recovered=True)
            except (OSError, ValueError, KeyError, TypeError):
                self.store.interrupt(
                    task, attempt, definitely_unsent=self.definitely_unsent(attempt),
                    retry=attempt["number"] < batch["max_attempts"])
        if (self.store.batch(batch_id)["status"] != "stopped" and
                all(row["status"] in ("succeeded", "failed") for row in self.store.tasks(batch_id))):
            with self.store.connect(immediate=True) as db:
                db.execute("UPDATE batches SET status='completed' WHERE id=?", (batch_id,))

    def recover(self, batch_id):
        with worker_lock(self.store.path):
            self.recover_unlocked(batch_id)
        return self.store.status(batch_id)

    def try_cache(self, task, prepared, batch):
        if batch["repeat"]:
            return False
        with self.store.connect() as db:
            cached = db.execute("""SELECT a.* FROM cache c JOIN attempts a ON a.id=c.source_attempt_id
                JOIN tasks t ON t.id=a.task_id WHERE c.cache_key=? AND t.cache_key=?
                AND a.status='succeeded'""", (task["cache_key"], task["cache_key"])).fetchone()
        if not cached:
            return False
        try:
            receipt, _, _ = self.evidence(dict(cached), prepared, batch)
            if receipt["engineering_status"] != "valid_result" or receipt.get("error_code"):
                raise RunError("缓存结果未通过校验。")
        except (OSError, ValueError, KeyError, TypeError):
            with self.store.connect(immediate=True) as db:
                db.execute("DELETE FROM cache WHERE cache_key=?", (task["cache_key"],))
                self.store.event(db, task["batch_id"], "cache_invalidated", task["id"],
                                 {"source_attempt_id": cached["id"]})
            return False
        with self.store.connect(immediate=True) as db:
            db.execute("""UPDATE tasks SET status='succeeded',updated_at=?,stop_code=NULL,
                result_attempt_id=?,cached_from=?,next_attempt_at=0 WHERE id=?""",
                       (self.clock(), cached["id"], cached["id"], task["id"]))
            self.store.event(db, task["batch_id"], "cache_hit", task["id"],
                             {"source_attempt_id": cached["id"]})
        return True

    def run(self, batch_id, *, retry_failed=False, wait_limit=30):
        with worker_lock(self.store.path):
            return self.run_unlocked(batch_id, retry_failed=retry_failed, wait_limit=wait_limit)

    def run_unlocked(self, batch_id, *, retry_failed=False, wait_limit=30):
        self.recover_unlocked(batch_id)
        batch = self.store.batch(batch_id)
        if batch["status"] == "stopped" and not retry_failed:
            return self.store.status(batch_id)
        if retry_failed:
            with self.store.connect(immediate=True) as db:
                db.execute("""UPDATE tasks SET status='pending',stop_code=NULL,next_attempt_at=0
                    WHERE batch_id=? AND status='failed' AND
                    (SELECT count(*) FROM attempts WHERE task_id=tasks.id) < ?""",
                           (batch_id, batch["max_attempts"]))
                db.execute("UPDATE batches SET status='pending',stop_code=NULL WHERE id=?", (batch_id,))
                self.store.event(db, batch_id, "explicit_retry_failed")
        pending = [row for row in self.store.tasks(batch_id) if row["status"] == "pending"]
        prepared = {}
        # Check the whole pending batch before allowing its first request.
        for task in pending:
            try:
                prepared[task["id"]] = self.inputs(task, batch)
                snapshot = self.snapshot_checker(task["sample_id"])
            except (OSError, ValueError, KeyError, TypeError):
                self.store.stop(batch_id, "input_or_initial_snapshot_check_failed")
                return self.store.status(batch_id)
            with self.store.connect(immediate=True) as db:
                db.execute("UPDATE tasks SET initial_snapshot_sha256=? WHERE id=?", (snapshot, task["id"]))
        with self.store.connect(immediate=True) as db:
            db.execute("UPDATE batches SET status='running',stop_code=NULL WHERE id=?", (batch_id,))
        while True:
            pending = [row for row in self.store.tasks(batch_id) if row["status"] == "pending"]
            if not pending:
                with self.store.connect(immediate=True) as db:
                    db.execute("UPDATE batches SET status='completed' WHERE id=?", (batch_id,))
                return self.store.status(batch_id)
            ready = [task for task in pending if task["next_attempt_at"] <= self.clock()]
            if not ready:
                delay = min(task["next_attempt_at"] for task in pending) - self.clock()
                if delay > wait_limit:
                    with self.store.connect(immediate=True) as db:
                        db.execute("UPDATE batches SET status='pending',stop_code='retry_wait' WHERE id=?", (batch_id,))
                    return self.store.status(batch_id)
                self.sleep(max(0, min(delay, 1)))
                continue
            task = ready[0]
            request = prepared[task["id"]]
            if self.try_cache(task, request, batch):
                continue
            key = os.environ.get("VLM_API_KEY", "").strip()
            if not key or not key.isascii() or any(c.isspace() for c in key):
                self.store.stop(batch_id, "invalid_api_key_environment")
                return self.store.status(batch_id)
            reservation = self.store.reserve(task, batch)
            if not reservation:
                self.store.stop(batch_id, "request_limit_reached")
                return self.store.status(batch_id)
            attempt_id, number = reservation
            with self.store.connect() as db:
                attempt = dict(db.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone())
            sending = copy.deepcopy(request)
            sending["metadata"]["initial_snapshot_sha256"] = task["initial_snapshot_sha256"]
            try:
                self.executor(sending, batch["base_url"], send=True, output_root=self.store.evidence,
                              timeout=batch["timeout"], run_id=attempt_id, attempt_number=number)
                action = self.finish(task, attempt, request, batch)
            except (OSError, ValueError, KeyError, TypeError):
                self.store.interrupt(task, attempt, definitely_unsent=self.definitely_unsent(attempt), retry=False)
                self.store.stop(batch_id, "local_execution_or_evidence_error")
                return self.store.status(batch_id)
            if action == "stop_batch":
                return self.store.status(batch_id)


def send_one(prepared, base_url, timeout):
    """Keep the D02 --send entry point under the same D03 accounting."""
    store = RunStore()
    batch_id = store.create([prepared], base_url, timeout=timeout)
    return Worker(store).run(batch_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="只创建批次，不调用 API")
    create.add_argument("--sample-id", action="append", required=True)
    create.add_argument("--repeat", action="store_true", help="独立重复运行，绕过成功缓存")
    create.add_argument("--max-attempts", type=int, default=3)
    create.add_argument("--timeout", type=float, default=180)
    create.add_argument("--max-output-tokens", type=int, default=client.DEFAULT_OUTPUT_TOKENS)
    for name in ("run", "resume", "recover", "status"):
        command = commands.add_parser(name)
        command.add_argument("--batch-id", required=name != "status")
        if name in ("run", "resume"):
            command.add_argument("--retry-failed", action="store_true",
                                 help="显式重开未达到次数上限的失败任务，历史不删除")
    args = parser.parse_args()
    store = RunStore(args.db)
    if args.command == "create":
        settings = client.settings_from_environment()
        requests = [client.build_request(sid, settings["model"], args.max_output_tokens)
                    for sid in args.sample_id]
        batch_id = store.create(requests, settings["base_url"], timeout=args.timeout,
                                max_attempts=args.max_attempts, repeat=args.repeat)
        result = store.status(batch_id)
    elif args.command in ("run", "resume"):
        result = Worker(store).run(args.batch_id, retry_failed=args.retry_failed)
    elif args.command == "recover":
        result = Worker(store).recover(args.batch_id)
    else:
        result = store.status(args.batch_id)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 1 if any(batch["status"] == "stopped" or any(
        task["status"] == "failed" for task in batch["tasks"]) for batch in result["batches"]) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("运行已中断，尝试记录已保留。请使用 recover 检查，或 resume 继续。", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
        print("任务操作失败：请检查批次编号、规则、输入证据和运行库；未打印接口或密钥。", file=sys.stderr)
        sys.exit(2)
