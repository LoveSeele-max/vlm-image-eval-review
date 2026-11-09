"""F01 development V1 preparation/runner, preserving D03 cache engine and D04 inputs."""
import argparse
import json
import os

import judge_client as client
from annotation_store import now
from build_manifest import DATA
from freeze_human_reference import load_registered_reference
from run_store import DEFAULT_DB, REQUEST_LIMIT, RunError, RunStore, identity, worker_lock
from run_tasks import Worker


ANCHOR_BATCH = "batch-4d2824f5bb5d4e2ca617bf4fd67e6109"
BILLING = client.PRIVATE / "f01_billing_check.json"
STATUS_PATH = client.PRIVATE / "f01_run_status.json"


def development_reference():
    loaded = load_registered_reference()
    snapshots = {}
    for row in loaded["initial_rows"]:
        if row["split"] == "development":
            sid, digest = row["sample_id"], row["snapshot_sha256"]
            if sid in snapshots and snapshots[sid] != digest:
                raise RunError("同一开发样本的冻结快照绑定不一致。")
            snapshots[sid] = digest
    split = client.strict_json((DATA / "split.json").read_bytes())
    ids = tuple(split["partitions"]["development"]["sample_ids"])
    if set(ids) != set(snapshots) or len(ids) != len(set(ids)):
        raise RunError("开发集与正式人工参考不一致。")
    return ids, snapshots, loaded["reference"]


def snapshot_guard(snapshots, seal_sha256):
    def check(sid):
        if sid not in snapshots:
            raise RunError("F01 只允许冻结开发集。")
        current = load_registered_reference()["reference"]
        if current["seal_sha256"] != seal_sha256:
            raise RunError("正式人工参考版本已改变。")
        actual = client.require_initial_snapshot(sid)
        if actual != snapshots[sid]:
            raise RunError("当前逐图初始快照与冻结参考不一致。")
        return actual
    return check


def baseline_profile(store):
    batch = store.batch(ANCHOR_BATCH)
    tasks = store.tasks(ANCHOR_BATCH)
    split = client.strict_json((DATA / "split.json").read_bytes())
    if batch["repeat"] or batch["status"] != "completed" or set(t["sample_id"] for t in tasks) != set(
            split["smoke_selection"]["sample_ids"]):
        raise RunError("D04 基准批次不完整。")
    worker = Worker(store)
    profiles = []
    for task in tasks:
        if task["status"] != "succeeded":
            raise RunError("D04 基准任务未成功。")
        prepared = worker.inputs(task, batch)
        with store.connect() as db:
            row = db.execute("SELECT * FROM attempts WHERE id=?", (task["result_attempt_id"],)).fetchone()
        if row is None:
            raise RunError("D04 成功尝试证据缺失。")
        receipt, _, _ = worker.evidence(dict(row), prepared, batch)
        if receipt.get("engineering_status") != "valid_result":
            raise RunError("D04 基准证据未通过复验。")
        meta = prepared["metadata"]
        if meta["prompt_version"] != "judge_v1" or meta["rule_version"] != "rules_v1":
            raise RunError("D04 基准不是正式 V1。")
        profiles.append({"base_url": batch["base_url"], "timeout": batch["timeout"],
                         "max_attempts": batch["max_attempts"], "model": prepared["body"]["model"],
                         "max_output_tokens": prepared["body"]["max_output_tokens"]})
    if not profiles or any(profile != profiles[0] for profile in profiles):
        raise RunError("D04 比较配置不一致。")
    settings = client.settings_from_environment()
    if any(settings[key] != profiles[0][key] for key in ("base_url", "model")):
        raise RunError("当前接口或模型不同于 D04，不能作为相同配置继续。")
    return profiles[0]


def inspect_cache(store, prepared, batch):
    key = identity(prepared, batch["base_url"], batch["timeout"])
    with store.connect() as db:
        row = db.execute("SELECT a.* FROM cache c JOIN attempts a ON a.id=c.source_attempt_id "
                         "WHERE c.cache_key=?", (key,)).fetchone()
    if row is None:
        return {"reusable": False, "reason": "no_success_cache"}
    try:
        if row["status"] != "succeeded":
            raise RunError("缓存来源未成功。")
        receipt, _, _ = Worker(store).evidence(dict(row), prepared, batch)
        if receipt.get("engineering_status") != "valid_result":
            raise RunError("缓存证据无效。")
    except (OSError, ValueError, KeyError, TypeError):
        return {"reusable": False, "reason": "invalid_cache_evidence"}
    return {"reusable": True, "source_attempt_id": row["id"], "cache_key": key}


def prepare_batch(store, batch_id=None):
    ids, snapshots, reference = development_reference()
    profile = baseline_profile(store)
    guard = snapshot_guard(snapshots, reference["seal_sha256"])
    requests = []
    for sid in ids:
        guard(sid)
        requests.append(client.build_request(sid, profile["model"], profile["max_output_tokens"]))
    with worker_lock(store.path):
        if batch_id is None:
            for batch in reversed(store.status()["batches"]):
                if not batch["repeat"] and tuple(t["sample_id"] for t in batch["tasks"]) == ids:
                    batch_id = batch["id"]
                    break
        if batch_id is None:
            batch_id = store.create(requests, profile["base_url"], timeout=profile["timeout"],
                                    max_attempts=profile["max_attempts"])
        batch = store.batch(batch_id)
        tasks = store.tasks(batch_id)
        if (batch["repeat"] or tuple(t["sample_id"] for t in tasks) != ids
                or batch["base_url"] != profile["base_url"] or batch["timeout"] != profile["timeout"]
                or batch["max_attempts"] != profile["max_attempts"]):
            raise RunError("批次不是同一冻结配置的完整开发集 V1。")
        worker = Worker(store)
        cache = {}
        for task, request in zip(tasks, requests):
            if worker.inputs(task, batch) != request:
                raise RunError("开发集请求存档不一致。")
            cache[task["sample_id"]] = inspect_cache(store, request, batch)
        plan = {"task": "F01", "prepared_at": now(), "batch_id": batch_id,
                "sample_ids": list(ids), "reference": reference,
                "profile": profile, "cache": cache, "budget": store.status(batch_id)["budget"],
                "expected_new_requests_if_all_pending": sum(not r["reusable"] for r in cache.values()),
                "request_limit": REQUEST_LIMIT, "api_calls_during_preparation": 0}
        binding = {"batch_id": batch_id, "sample_ids": list(ids), "reference": reference, "profile": profile}
        folder = store.path.parent / "development_v1" / batch_id
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / "binding.json"
        if target.exists():
            if client.strict_json(target.read_bytes()) != binding:
                raise RunError("既有批次的整体参考或配置绑定不同，拒绝覆盖。")
        else:
            with target.open("xb") as stream:
                stream.write(client.encoded(binding) + b"\n")
        client.write_json(folder / "preflight.json", plan, "")
    return batch_id, plan, guard


def billing_check():
    record = client.strict_json(BILLING.read_bytes())
    if (record.get("source") != "user_report" or record.get("unit") != "CNY"
            or type(record.get("available_balance")) not in (int, float)
            or not 0 < record["available_balance"] <= 1000000
            or not record.get("checked_at")):
        raise RunError("先核对本次实际可用余额及单位，不把额度当作人民币。")
    return record


def run_batch(store, batch_id=None, on_prepared=None):
    billing = billing_check()
    batch_id, plan, guard = prepare_batch(store, batch_id)
    if on_prepared is not None:
        on_prepared(batch_id)
    result = Worker(store, snapshot_checker=guard).run(batch_id)
    # No labels or credentials enter the model request from the reference or billing record.
    secret = os.environ.get("VLM_API_KEY", "")
    client.write_json(STATUS_PATH, {"task": "F01", "batch_id": batch_id,
                                   "reference": plan["reference"], "billing": billing, "result": result}, secret)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--batch-id")
    args = parser.parse_args()
    store = RunStore()
    if args.prepare:
        batch_id, plan, _ = prepare_batch(store, args.batch_id)
        print(json.dumps({"batch_id": batch_id, "sample_count": len(plan["sample_ids"]),
                          "reusable_cache": sum(r["reusable"] for r in plan["cache"].values()),
                          "expected_new_requests": plan["expected_new_requests_if_all_pending"],
                          "budget": plan["budget"], "api_calls": 0}, ensure_ascii=True, indent=2))
    else:
        parser.error("请使用 --prepare；实际运行使用本机密码窗口。")


if __name__ == "__main__":
    main()
