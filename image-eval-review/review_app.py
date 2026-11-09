"""F02: read-only development comparison, frozen author reference and verified V1 evidence."""
import argparse
import json
import sqlite3
import threading
import webbrowser
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import judge_client as client
from annotation_store import now
from build_manifest import DATA, read_jsonl
from freeze_human_reference import load_registered_reference
from freeze_rules import load_frozen_context
from prepare_image import prepare_image
from run_store import DEFAULT_DB, RunError, RunStore
from run_tasks import Worker


DEFAULT_BATCH = "batch-2a50643355544ef49d9c8d5831d8ef61"
WEB = Path(__file__).resolve().parent / "web"
FILTERS = {"all", "disagreement", "unjudgeable", "engineering_failure"}
DIMENSIONS = {"all", "subject_features", "structural_integrity", "prompt_adherence"}


class LockedSample(ValueError):
    pass


class ReadOnlyRuns(RunStore):
    """Use existing read helpers without D03 initialization or metadata writes."""
    def __init__(self, database=DEFAULT_DB):
        self.path = Path(database).resolve()
        self.root = self.path.with_suffix("")
        self.inputs, self.evidence = self.root / "inputs", self.root / "evidence"

    @contextmanager
    def connect(self, immediate=False):
        if immediate:
            raise RunError("分歧页禁止写入运行库。")
        db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            yield db
        finally:
            db.close()

    def snapshot(self, batch_id):
        with self.connect() as db:
            batch = db.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
            if batch is None:
                raise RunError("开发集批次不存在。")
            tasks = [dict(r) for r in db.execute("SELECT * FROM tasks WHERE batch_id=? ORDER BY rowid", (batch_id,))]
            attempts = [dict(r) for r in db.execute("SELECT a.* FROM attempts a JOIN tasks t ON t.id=a.task_id "
                                                   "WHERE t.batch_id=? ORDER BY a.task_id,a.number", (batch_id,))]
            result_ids = {t["result_attempt_id"] for t in tasks if t["result_attempt_id"]}
            results = {}
            for identifier in result_ids:
                row = db.execute("SELECT * FROM attempts WHERE id=?", (identifier,)).fetchone()
                if row is None:
                    raise RunError("成功尝试来源缺失。")
                results[identifier] = dict(row)
        return {"batch": dict(batch), "tasks": tasks, "attempts": attempts, "results": results}


def public_label(label):
    return {key: label[key] for key in (
        "status", "candidate_region", "candidate_observation", "reason")}


def compare_sample(sid, criteria, human_records, ai_labels, *, task_status, error_code,
                   attempts, cached, candidate_status=None):
    """Project only display fields; failures remain missing model labels, never visual fail."""
    if set(human_records) != {r["id"] for r in criteria}:
        raise RunError("人工参考条目覆盖不符。")
    if ai_labels is not None and set(ai_labels) != set(human_records):
        raise RunError("模型条目覆盖不符。")
    if (task_status == "succeeded") != (ai_labels is not None):
        raise RunError("任务状态与模型结果不一致。")
    rows = []
    for criterion in criteria:
        cid = criterion["id"]
        author = human_records[cid]
        human = public_label(author["label"])
        ai = public_label(ai_labels[cid]) if ai_labels is not None else None
        origin = author["origin"]
        if origin.get("decision_source") == "explicit_user_chat_instruction":
            provenance = "作者明确指定的判断；文字由助手复制或整理"
        elif origin.get("kind") == "author_status_codex_evidence":
            provenance = "作者原始判断；证据文字含助手辅助补写"
        else:
            provenance = "作者页面填写"
        rows.append({"sample_id": sid, "criterion_id": cid,
                     **{key: criterion[key] for key in ("name", "dimension", "required", "pass_when", "fail_when", "unjudgeable_when")},
                     "boundary": criterion.get("boundary", ""),
                     "human": human, "ai": ai, "human_provenance": provenance,
                     "disagreement": ai is not None and human["status"] != ai["status"],
                     "unjudgeable": human["status"] == "unjudgeable" or (ai is not None and ai["status"] == "unjudgeable"),
                     "engineering_failure": ai is None})
    return {"sample_id": sid, "task_status": task_status, "error_code": error_code,
            "candidate_status": candidate_status, "cached": bool(cached), "attempts": attempts,
            "rows": rows, "counts": count_rows(rows)}


def count_rows(rows):
    return {"all": len(rows), "disagreement": sum(r["disagreement"] for r in rows),
            "unjudgeable": sum(bool(r["unjudgeable"]) for r in rows),
            "engineering_failure": sum(r["engineering_failure"] for r in rows)}


def select_rows(dataset, *, filter_name="all", sample="all", dimension="all", query=""):
    if filter_name not in FILTERS or dimension not in DIMENSIONS or len(query) > 200:
        raise ValueError("筛选参数非法。")
    if sample != "all" and sample not in {r["sample_id"] for r in dataset["samples"]}:
        raise LockedSample("本页只开放开发集，保留集尚未解锁。")
    selected = [row for item in dataset["samples"] for row in item["rows"]
                if (sample == "all" or row["sample_id"] == sample)
                and (dimension == "all" or row["dimension"] == dimension)
                and (not query or query.casefold() in (row["sample_id"] + " " + row["criterion_id"] + " " + row["name"]).casefold())]
    counts = count_rows(selected)
    rows = [row for row in selected if filter_name == "all" or bool(row[filter_name])]
    return {"meta": dataset["meta"], "summary": dataset["summary"],
            "samples": [{k: v for k, v in item.items() if k != "rows"} for item in dataset["samples"]],
            "filter_counts": counts, "matched_items": len(rows),
            "matched_samples": len({row["sample_id"] for row in rows}), "rows": rows}


def load_dataset(database=DEFAULT_DB, batch_id=DEFAULT_BATCH):
    frozen = load_registered_reference()
    policy, reference, evaluation, sample_specs = load_frozen_context()
    manifest = {r["sample_id"]: r for r in read_jsonl(DATA / "manifest.jsonl")}
    dev_ids = tuple(client.strict_json((DATA / "split.json").read_bytes())["partitions"]["development"]["sample_ids"])
    expected_humans = {sid: {} for sid in dev_ids}
    snapshot_hashes = {}
    for row in frozen["initial_rows"]:
        if row["split"] == "development":
            expected_humans[row["sample_id"]][row["record"]["label"]["criterion_id"]] = row["record"]
            snapshot_hashes[row["sample_id"]] = row["snapshot_sha256"]
    specs = {r["sample_id"]: r for r in sample_specs}
    store = ReadOnlyRuns(database)
    snapshot = store.snapshot(batch_id)
    batch, tasks = snapshot["batch"], snapshot["tasks"]
    if batch["repeat"] or tuple(t["sample_id"] for t in tasks) != dev_ids:
        raise LockedSample("查看批次必须是完整冻结开发集的普通批次。")
    binding_path = store.path.parent / "development_v1" / batch_id / "binding.json"
    binding = client.strict_json(binding_path.read_bytes())
    if binding["batch_id"] != batch_id or binding["sample_ids"] != list(dev_ids) or binding["reference"] != frozen["reference"]:
        raise RunError("开发集批次的人工参考绑定不一致。")
    worker = Worker(store)
    samples = []
    for task in tasks:
        sid = task["sample_id"]
        if manifest[sid]["split"] != "development":
            raise LockedSample("保留集未解锁。")
        if task["initial_snapshot_sha256"] and task["initial_snapshot_sha256"] != snapshot_hashes[sid]:
            raise RunError("运行时初始快照与正式参考不同。")
        prepared = worker.inputs(task, batch)
        meta = prepared["metadata"]
        if meta["prompt_version"] != "judge_v1" or meta["rule_version"] != policy["rule_version"]:
            raise RunError("分歧页只接受本次正式 V1。")
        profile = binding["profile"]
        if (batch["base_url"] != profile["base_url"] or batch["timeout"] != profile["timeout"]
                or batch["max_attempts"] != profile["max_attempts"]
                or prepared["body"]["model"] != profile["model"]
                or prepared["body"]["max_output_tokens"] != profile["max_output_tokens"]):
            raise RunError("查看结果的比较配置与 F01 绑定不一致。")
        histories = [a for a in snapshot["attempts"] if a["task_id"] == task["id"]]
        for attempt in histories:
            if attempt["status"] in ("succeeded", "failed"):
                worker.evidence(attempt, prepared, batch)
        ai = None
        candidate_status = None
        if task["status"] == "succeeded":
            attempt = snapshot["results"][task["result_attempt_id"]]
            receipt, _, _ = worker.evidence(attempt, prepared, batch)
            if attempt["status"] != "succeeded" or receipt.get("engineering_status") != "valid_result":
                raise RunError("成功结果证据无效。")
            folder = Path(attempt["evidence_dir"])
            parsed = client.strict_json((folder / "parsed.json").read_bytes())
            ai = {r["criterion_id"]: r for r in parsed["labels"]}
            candidate_status = receipt["candidate_status"]
        criteria = []
        for dim, rules in (("subject_features", reference["criteria"]),
                           ("structural_integrity", evaluation["structure_criteria"]),
                           ("prompt_adherence", specs[sid]["prompt_criteria"])):
            criteria.extend({**r, "dimension": dim, "name": r.get("name", r.get("description", r["id"]))} for r in rules)
        sample = compare_sample(sid, criteria, expected_humans[sid], ai,
            task_status=task["status"], error_code=task["stop_code"], cached=task["cached_from"],
            candidate_status=candidate_status,
            attempts=[{k: a[k] for k in ("number", "status", "http_status", "error_code", "elapsed_seconds")} for a in histories])
        sample["prompt"] = manifest[sid]["prompt"]
        samples.append(sample)
    if snapshot != store.snapshot(batch_id):
        raise RunError("读取期间批次发生变化，请重新加载。")
    if load_registered_reference()["reference"] != frozen["reference"]:
        raise RunError("读取期间冻结参考发生变化。")
    counts = count_rows([r for item in samples for r in item["rows"]])
    return {"meta": {"batch_id": batch_id, "prompt_version": "judge_v1",
                     "rule_version": policy["rule_version"], "reference_version": frozen["reference"]["reference_version"],
                     "reference_seal_sha256": frozen["reference"]["seal_sha256"], "loaded_at": now(),
                     "holdout_locked": True, "read_only": True},
            "summary": {**counts, "samples": len(samples),
                        "successful_samples": sum(s["task_status"] == "succeeded" for s in samples),
                        "failed_samples": sum(s["task_status"] == "failed" for s in samples),
                        "valid_model_items": counts["all"] - counts["engineering_failure"]}, "samples": samples}


class ReviewService:
    def __init__(self, database=DEFAULT_DB, batch_id=DEFAULT_BATCH):
        self.database, self.batch_id = database, batch_id
        self.dataset = load_dataset(database, batch_id)
        self.lock = threading.Lock()
        self.sources = {r["sample_id"]: r for r in read_jsonl(DATA / "sample_sources.jsonl")}
        self.manifest = {r["sample_id"]: r for r in read_jsonl(DATA / "manifest.jsonl")}
        self.references = {r["reference_id"]: r for r in client.strict_json((DATA / "reference_assets.json").read_bytes())}
        self.image_policy = client.strict_json((DATA / "image_input_policy.json").read_bytes())
        self.allowed = {s["sample_id"] for s in self.dataset["samples"]}

    def review(self, **kwargs):
        return select_rows(self.dataset, **kwargs)

    def reload(self):
        with self.lock:
            dataset = load_dataset(self.database, self.batch_id)
            self.dataset = dataset  # Replace only after every evidence check passed.
        return {"loaded_at": dataset["meta"]["loaded_at"]}

    def image(self, identifier):
        if identifier in self.references:
            row = self.references[identifier]
            return prepare_image(Path(row["source_path"]), identifier, self.image_policy, row["sha256"])[0]
        if identifier not in self.allowed:
            raise LockedSample("本页不提供保留集或未登记图片。")
        return prepare_image(Path(self.sources[identifier]["source_path"]), "CANDIDATE", self.image_policy,
                             self.manifest[identifier]["candidate_sha256"])[0]


def make_server(service, port=8502):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def safe_origin(self):
            valid = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            return self.headers.get("Host", "") in valid and (
                self.headers.get("Origin") is None or self.headers.get("Origin") in {"http://" + h for h in valid})

        def send(self, value, kind="application/json; charset=utf-8", status=200):
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            for key, val in (("Content-Type", kind), ("Content-Length", str(len(value))), ("Cache-Control", "no-store"),
                             ("X-Content-Type-Options", "nosniff"),
                             ("Content-Security-Policy", "default-src 'self'; img-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'; base-uri 'none'")):
                self.send_header(key, val)
            self.end_headers()
            self.wfile.write(value)

        def do_GET(self):
            if not self.safe_origin():
                return self.send({"error": "仅允许本机页面访问。"}, status=403)
            parsed = urlsplit(self.path)
            try:
                static = {"/": ("review.html", "text/html; charset=utf-8"),
                          "/review.js": ("review.js", "text/javascript; charset=utf-8"),
                          "/review.css": ("review.css", "text/css; charset=utf-8")}
                if parsed.path in static:
                    name, kind = static[parsed.path]
                    return self.send((WEB / name).read_bytes(), kind)
                if parsed.path == "/api/health":
                    return self.send({"app": "vlm-development-review", "version": 1})
                if parsed.path == "/api/review":
                    query = parse_qs(parsed.query, keep_blank_values=True)
                    if not set(query) <= {"filter", "sample", "dimension", "q"} or any(len(v) != 1 for v in query.values()):
                        raise ValueError("筛选参数非法；保留集未解锁。")
                    args = {key: query.get(key, [default])[0] for key, default in (
                        ("filter", "all"), ("sample", "all"), ("dimension", "all"), ("q", ""))}
                    return self.send(service.review(filter_name=args["filter"], sample=args["sample"],
                                                    dimension=args["dimension"], query=args["q"]))
                if parsed.path.startswith("/api/image/"):
                    return self.send(service.image(parsed.path.removeprefix("/api/image/")), "image/png")
                return self.send({"error": "页面不存在。"}, status=404)
            except LockedSample as exc:
                return self.send({"error": str(exc)}, status=403)
            except ValueError as exc:
                return self.send({"error": str(exc)}, status=400)
            except (OSError, KeyError, TypeError, sqlite3.Error):
                return self.send({"error": "本地结果读取失败，请核对证据或重新加载。"}, status=500)

        def do_POST(self):
            # The only action refreshes in-memory evidence; label edits, retries and sends do not exist here.
            if not self.safe_origin():
                return self.send({"error": "仅允许本机页面访问。"}, status=403)
            if self.path != "/api/reload":
                return self.send({"error": "此页只读，不提供标注或模型执行接口。"}, status=405)
            try:
                return self.send(service.reload())
            except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
                return self.send({"error": "重新核验失败，旧显示未被替换；请检查本机证据。"}, status=500)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH)
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    service = ReviewService(args.db, args.batch_id)
    server = make_server(service, args.port)
    address = f"http://127.0.0.1:{server.server_port}"
    print(f"开发集人机对照页已启动：{address}", flush=True)
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
        raise SystemExit("分歧页启动失败，请检查正式开发集批次、冻结参考与证据文件。") from None
