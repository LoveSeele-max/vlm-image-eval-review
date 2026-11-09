"""G01: explicitly gated, frozen V1/V2 holdout execution on the shared D03 store.

Legacy engine, prompts, cache identities and retry decisions remain unchanged.
This module grants holdout access only to an independently bound complete pair.
"""
import argparse
import copy
import json
import os
import sqlite3
import time
import uuid
from collections import Counter
from pathlib import Path

import judge_client as client
import prompt_versions as pv
from annotation_store import now
from build_manifest import DATA, text_digest
from development_v1 import billing_check
from freeze_human_reference import load_registered_reference
from run_store import DEFAULT_DB, REQUEST_LIMIT, RunError, RunStore, identity, worker_lock
from run_tasks import Worker


VERSIONS = ("judge_v1", "judge_v2")
PLAN_NAME = "experiment.json"


def gates():
    seal = pv.verify_registry()
    loaded = load_registered_reference()
    reference = loaded["reference"]
    if (reference["seal_sha256"] != seal["human_reference_seal_sha256"] or
            reference["reference_version"] != seal["human_reference_version"]):
        raise RunError("人工参考与 V2 冻结登记不同。")
    split = pv.read_json(DATA / "split.json")
    ids = tuple(split["partitions"]["holdout"]["sample_ids"])
    snapshots = {}
    for row in loaded["initial_rows"]:
        if row["split"] == "holdout":
            sid, digest = row["sample_id"], row["snapshot_sha256"]
            if sid in snapshots and snapshots[sid] != digest:
                raise RunError("同图冻结快照不一致。")
            snapshots[sid] = digest
    if len(ids) != 30 or len(set(ids)) != 30 or set(ids) != set(snapshots):
        raise RunError("必须使用全部固定的 30 张保留图及其冻结初标。")
    settings = client.settings_from_environment()
    profile = seal["comparison_profile"]
    if (settings["model"] != profile["model"] or
            client.digest(settings["base_url"].encode()) != profile["endpoint_base_sha256"]):
        raise RunError("接口或模型不同于冻结比较配置。")
    return {"sample_ids": list(ids), "snapshots": snapshots,
            "reference": reference, "profile": profile,
            "base_url": settings["base_url"],
            "registry_sha256": text_digest(pv.REGISTRY),
            "runner_sha256": text_digest(Path(__file__))}


def parity(v1, v2):
    body = copy.deepcopy(v2["body"])
    body["input"][0]["content"][0]["text"] = v1["body"]["input"][0]["content"][0]["text"]
    if body != v1["body"] or v1["expected"] != v2["expected"] or v1["policy"] != v2["policy"]:
        raise RunError("两版图片、上下文、规则或请求参数不一致。")
    if v1["metadata"]["images"] != v2["metadata"]["images"]:
        raise RunError("两版图片预处理不一致。")


class HoldoutStore(RunStore):
    @property
    def folder(self):
        return self.path.parent / "holdout_g01"

    @property
    def plan_path(self):
        return self.folder / PLAN_NAME

    def binding(self):
        plan = pv.read_json(self.plan_path)
        current = gates()
        if (plan["schema_version"] != 1 or plan["task"] != "G01" or
                plan["frozen_inputs"] != current or set(plan["batches"]) != set(VERSIONS)):
            raise RunError("G01 冻结输入或执行代码绑定已变化，拒绝继续。")
        if len(set(plan["batches"].values())) != 2:
            raise RunError("两版必须使用独立批次编号。")
        for version, bid in plan["batches"].items():
            batch, tasks = self.batch(bid), self.tasks(bid)
            profile = current["profile"]
            if (batch["repeat"] or batch["base_url"] != current["base_url"] or
                    batch["timeout"] != profile["timeout_seconds"] or
                    batch["max_attempts"] != profile["max_attempts"] or
                    [t["sample_id"] for t in tasks] != current["sample_ids"] or
                    any(json.loads(t["spec_json"])["metadata"]["prompt_version"] != version or
                        json.loads(t["spec_json"])["metadata"]["split"] != "holdout" for t in tasks)):
                raise RunError("批次不是绑定的完整同配置保留集版本。")
        return plan

    def prepare_pair(self):
        with worker_lock(self.path):
            if self.plan_path.exists():
                return self.binding()
            frozen = gates()
            billing = billing_check()
            status = self.status()
            if status["budget"]["uncertain_or_running"]:
                raise RunError("存在未核对的运行或中断请求，先恢复原任务。")
            if REQUEST_LIMIT - status["budget"]["reserved"] < 60:
                raise RunError("剩余名额不足两版正常的 60 次请求。")
            # Refuse accidental duplicate work after an interrupted plan publication.
            with self.connect() as db:
                if any(json.loads(r[0])["metadata"]["split"] == "holdout"
                       for r in db.execute("SELECT spec_json FROM tasks")):
                    raise RunError("已有保留任务但缺登记；保留证据，先核对，不新建重复实验。")
            batches = {version: "batch-" + uuid.uuid4().hex for version in VERSIONS}
            stored = []
            profile = frozen["profile"]
            # Build one image pair at a time, rather than keeping 60 base64 requests in RAM.
            for sid in frozen["sample_ids"]:
                pair = [pv.build_versioned_request(sid, profile["model"], profile["max_output_tokens"],
                         prompt_version=version) for version in VERSIONS]
                parity(*pair)
                for version, prepared in zip(VERSIONS, pair):
                    if prepared["metadata"]["sample_id"] != sid or prepared["metadata"]["split"] != "holdout":
                        raise RunError("请求样本身份与固定保留集不同。")
                    key = identity(prepared, frozen["base_url"], profile["timeout_seconds"])
                    path = self.inputs / (key + ".json")
                    raw = client.encoded(prepared) + b"\n"
                    if path.exists() and path.read_bytes() != raw:
                        raise RunError("输入文件冲突，拒绝覆盖。")
                    if not path.exists():
                        with path.open("xb") as stream:
                            stream.write(raw)
                    spec = {"metadata": prepared["metadata"], "model": prepared["body"]["model"],
                            "max_output_tokens": prepared["body"]["max_output_tokens"]}
                    stored.append((version, sid, key, path, client.digest(raw), spec))
            stamp = time.time()
            with self.connect(immediate=True) as db:
                for version, bid in batches.items():
                    db.execute("INSERT INTO batches VALUES(?,?,?,?,?,?,?,?)", (bid, stamp, "pending", None,
                        0, profile["max_attempts"], profile["timeout_seconds"], frozen["base_url"]))
                    self.event(db, bid, "g01_pair_created", detail={"version": version, "sample_count": 30})
                for version, sid, key, path, sha, spec in stored:
                    db.execute("""INSERT INTO tasks(id,batch_id,sample_id,cache_key,input_path,input_sha256,
                        spec_json,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                        ("task-" + uuid.uuid4().hex, batches[version], sid, key, str(path), sha,
                         json.dumps(spec, ensure_ascii=False), "pending", stamp, stamp))
            self.folder.mkdir(parents=True, exist_ok=True)
            plan = {"schema_version": 1, "task": "G01", "created_at": now(),
                    "frozen_inputs": frozen, "batches": batches,
                    "budget_before": status["budget"], "normal_new_requests": 60,
                    "billing": billing, "order": list(VERSIONS),
                    "policy": "Both versions use all fixed holdout images; no automatic reopening of final failures."}
            with self.plan_path.open("xb") as stream:
                stream.write(client.encoded(plan) + b"\n")
            return self.binding()


class HoldoutWorker(Worker):
    def __init__(self, store, **kwargs):
        if "snapshot_checker" in kwargs:
            raise RunError("保留集快照检查不能由调用者替换。")
        super().__init__(store, snapshot_checker=self.snapshot, **kwargs)
        executor = self.executor

        def guarded_executor(prepared, *args, **options):
            plan = self.store.binding()
            sid = prepared["metadata"]["sample_id"]
            if (sid not in plan["frozen_inputs"]["sample_ids"] or
                    prepared["metadata"]["prompt_version"] not in VERSIONS or
                    prepared["metadata"]["split"] != "holdout"):
                raise RunError("发送请求不在正式保留集版本中。")
            self.snapshot(sid)
            return executor(prepared, *args, **options)
        self.executor = guarded_executor

    def snapshot(self, sid):
        frozen = self.store.binding()["frozen_inputs"]
        if sid not in frozen["snapshots"]:
            raise RunError("非保留集样本。")
        return frozen["snapshots"][sid]

    def inputs(self, task, batch):
        plan = self.store.binding()
        versions = [v for v, bid in plan["batches"].items() if bid == batch["id"]]
        if len(versions) != 1 or task["sample_id"] not in plan["frozen_inputs"]["sample_ids"]:
            raise RunError("任务未绑定正式 G01 实验。")
        path = Path(task["input_path"]).resolve()
        if not path.is_relative_to(self.store.inputs.resolve()):
            raise RunError("输入位置非法。")
        raw = path.read_bytes()
        prepared = client.strict_json(raw)
        if (client.digest(raw) != task["input_sha256"] or
                identity(prepared, batch["base_url"], batch["timeout"]) != task["cache_key"]):
            raise RunError("输入存档或缓存绑定变化。")
        profile = plan["frozen_inputs"]["profile"]
        current = pv.build_versioned_request(task["sample_id"], profile["model"],
                    profile["max_output_tokens"], prompt_version=versions[0])
        if current != prepared:
            raise RunError("素材、规则、版本或参数已变化。")
        return prepared


def compact_status(store):
    plan = store.binding()
    result = {"task": "G01", "updated_at": now(), "plan_sha256": client.digest(store.plan_path.read_bytes()),
              "budget": store.status()["budget"], "versions": {}}
    for version, bid in plan["batches"].items():
        batch = store.status(bid)["batches"][0]
        attempts = [a for t in batch["tasks"] for a in t["attempts"]]
        counts = dict(Counter(t["status"] for t in batch["tasks"]))
        usage = Counter()
        for attempt in attempts:
            reported = json.loads(attempt["usage_json"]) if attempt["usage_json"] else None
            if isinstance(reported, dict):
                usage.update({k: reported[k] for k in ("input_tokens", "output_tokens", "total_tokens")
                              if type(reported.get(k)) is int})
        result["versions"][version] = {"batch_id": bid, "status": batch["status"],
            "stop_code": batch["stop_code"], "samples": len(batch["tasks"]), "counts": counts,
            "attempts": len(attempts), "retry_count": sum(t["retry_count"] for t in batch["tasks"]),
            "elapsed_seconds": round(sum(a["elapsed_seconds"] or 0 for a in attempts), 3),
            "reported_usage": dict(usage),
            "attempts_without_usage": sum(not a["usage_json"] or a["usage_json"] == "null" for a in attempts),
            "failures": [{"sample_id": t["sample_id"], "code": t["stop_code"], "attempts": t["attempts"]}
                         for t in batch["tasks"] if t["status"] == "failed"]}
    result["both_versions_terminal"] = all(
        v["status"] == "completed" and sum(v["counts"].get(s, 0) for s in ("succeeded", "failed")) == 30
        for v in result["versions"].values())
    return result


def verify_results(store):
    plan = store.binding()
    worker = HoldoutWorker(store)
    artifacts, rows, missing = {}, [], Counter()
    for version, bid in plan["batches"].items():
        batch = store.batch(bid)
        for task in store.tasks(bid):
            prepared = worker.inputs(task, batch)
            artifacts[str(Path(task["input_path"]).relative_to(store.path.parent))] = task["input_sha256"]
            with store.connect() as db:
                attempts = [dict(r) for r in db.execute("SELECT * FROM attempts WHERE task_id=? ORDER BY number", (task["id"],))]
            for attempt in attempts:
                if attempt["status"] in ("succeeded", "failed"):
                    _, hashes, _ = worker.evidence(attempt, prepared, batch)
                    if not attempt["artifact_hashes_json"] or json.loads(attempt["artifact_hashes_json"]) != hashes:
                        raise RunError("尝试证据哈希绑定缺失。")
                    for name, sha in hashes.items():
                        artifacts[str((Path(attempt["evidence_dir"]) / name).relative_to(store.path.parent))] = sha
            selected = next((a for a in attempts if a["id"] == task["result_attempt_id"]), None)
            labels = None
            if task["status"] == "succeeded":
                if selected is None:
                    raise RunError("成功任务缺少独立版本结果。")
                labels = pv.read_json(Path(selected["evidence_dir"]) / "parsed.json")["labels"]
            else:
                missing[version] += len(prepared["expected"])
            rows.append({"prompt_version": version, "batch_id": bid, "sample_id": task["sample_id"],
                         "engineering_status": task["status"], "result_attempt_id": task["result_attempt_id"],
                         "stop_code": task["stop_code"], "labels": labels})
    summary = compact_status(store)
    summary["engineering_missing_items"] = dict(missing)
    summary["verified_at"] = now()
    summary["evidence_files_verified"] = len(artifacts)
    summary["reference"] = plan["frozen_inputs"]["reference"]
    summary["quality_metrics"] = "Not computed in G01; G02 must use the frozen initial reference."
    secret = os.environ.get("VLM_API_KEY", "")
    client.write_json(store.folder / "results.json", rows, secret)
    client.write_json(store.folder / "verification.json", summary, secret)
    artifacts[PLAN_NAME] = client.digest(store.plan_path.read_bytes())
    artifacts["results.json"] = client.digest((store.folder / "results.json").read_bytes())
    artifacts["verification.json"] = client.digest((store.folder / "verification.json").read_bytes())
    client.write_json(store.folder / "checksums.json", artifacts, secret)
    return summary


def run_pair(store):
    plan = store.prepare_pair()
    worker = HoldoutWorker(store)
    for version in VERSIONS:
        bid = plan["batches"][version]
        while True:
            result = worker.run(bid)
            client.write_json(store.folder / "status.json", compact_status(store), os.environ.get("VLM_API_KEY", ""))
            batch = result["batches"][0]
            if batch["status"] == "pending" and batch["stop_code"] == "retry_wait":
                time.sleep(min(30, max(0, min(t["next_attempt_at"] for t in batch["tasks"]
                    if t["status"] == "pending") - time.time())))
                continue
            break
        if batch["status"] != "completed":
            break  # Configuration, quota, input or request-limit error stops the whole pair.
    return verify_results(store)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status", "verify"))
    args = parser.parse_args()
    try:
        store = HoldoutStore()  # Formal CLI never offers a new budget database.
        if args.command == "prepare":
            plan = store.prepare_pair()
            result = {"batches": plan["batches"], "normal_new_requests": 60, "api_calls": 0}
        elif args.command == "run":
            result = run_pair(store)
        elif args.command == "verify":
            result = verify_results(store)
        else:
            result = compact_status(store)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    except KeyboardInterrupt:
        print("运行中断，记录已保留；同一 run 命令会先恢复，不会自动重开最终失败。")
        return 130
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
        print("G01 核验或执行未完成；请核对冻结输入、批次和私有证据，未显示密钥或原始响应。")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
