import copy
import io
import json
import os
import queue
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import judge_client as client
from run_store import REQUEST_LIMIT, RunError, RunStore, identity, worker_lock
from run_tasks import Worker
from test_judge_client import build, response, synthetic_prepare, valid_answer


SECRET = "fake-d03-test-key"


class Clock:
    def __init__(self):
        self.now = time.time()
        self.waits = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.put(request)
        if self.server.responses:
            code, payload, headers = self.server.responses.pop(0)
        else:
            code, payload, headers = 200, response(self.server.prepared), {}
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(raw)))
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.store = RunStore(self.folder / "runs.sqlite3")
        self.clock = Clock()
        self.image_patch = patch.object(client, "prepare_image", side_effect=synthetic_prepare)
        self.image_patch.start()
        self.addCleanup(self.image_patch.stop)
        self.environment = patch.dict(os.environ, {"VLM_API_KEY": SECRET})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.prepared = build()

    def serve(self, responses=None):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.responses = list(responses or [])
        server.requests = queue.Queue()
        server.prepared = self.prepared
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server, f"http://127.0.0.1:{server.server_port}/v1"

    def worker(self, **kwargs):
        return Worker(self.store, snapshot_checker=lambda sid: "synthetic-initial-snapshot",
                      clock=self.clock.time, sleep=self.clock.sleep, **kwargs)

    def create(self, url, requests=None, **kwargs):
        return self.store.create(requests or [self.prepared], url, **kwargs)

    @staticmethod
    def task(result, index=0):
        return result["batches"][0]["tasks"][index]

    def attempt(self, task_id):
        with self.store.connect() as db:
            return dict(db.execute("SELECT * FROM attempts WHERE task_id=? ORDER BY number DESC LIMIT 1",
                                   (task_id,)).fetchone())

    def test_create_and_status_persist_versions_without_requests_or_credentials(self):
        batch = self.create("https://unused.example/v1")
        with patch.object(client, "build_opener") as opener:
            result = RunStore(self.store.path).status(batch)
        opener.assert_not_called()
        self.assertEqual(self.task(result)["status"], "pending")
        self.assertEqual(result["budget"]["reserved"], 0)
        with self.store.connect() as db:
            spec = json.loads(db.execute("SELECT spec_json FROM tasks").fetchone()[0])
        self.assertEqual(spec["metadata"]["rule_version"], "rules_v1")
        self.assertEqual(spec["model"], "synthetic-model")
        self.assertEqual(spec["max_output_tokens"], 8192)
        self.assertNotIn(SECRET.encode(), self.store.path.read_bytes())

    def test_success_records_attempt_usage_model_time_and_evidence_reference(self):
        server, url = self.serve()
        batch = self.create(url)
        result = self.worker().run(batch)
        task = self.task(result)
        self.assertEqual(result["batches"][0]["status"], "completed")
        self.assertEqual(task["status"], "succeeded")
        self.assertEqual(task["retry_count"], 0)
        attempt = task["attempts"][0]
        self.assertEqual(attempt["returned_model"], "returned-synthetic-model")
        self.assertEqual(json.loads(attempt["usage_json"])["total_tokens"], 300)
        self.assertGreaterEqual(attempt["elapsed_seconds"], 0)
        self.assertTrue((Path(attempt["evidence_dir"]) / "raw_response.bin").exists())
        self.assertEqual(server.requests.qsize(), 1)
        self.assertEqual(result["budget"]["known_started"], 1)

    def test_verified_cache_reused_with_no_key_no_attempt_and_no_budget_charge(self):
        server, url = self.serve()
        first = self.worker().run(self.create(url))
        second_batch = self.create(url)
        with patch.dict(os.environ, {"VLM_API_KEY": ""}):
            second = self.worker().run(second_batch)
        self.assertEqual(self.task(second)["cached_from"], self.task(first)["result_attempt_id"])
        self.assertEqual(self.task(second)["attempts"], [])
        self.assertEqual(second["budget"]["reserved"], 1)
        self.assertEqual(server.requests.qsize(), 1)

    def test_explicit_repeat_has_independent_batch_and_bypasses_cache(self):
        server, url = self.serve()
        first = self.worker().run(self.create(url))
        second = self.worker().run(self.create(url, repeat=True))
        self.assertNotEqual(first["batches"][0]["id"], second["batches"][0]["id"])
        self.assertIsNone(self.task(second)["cached_from"])
        self.assertEqual(len(self.task(second)["attempts"]), 1)
        self.assertEqual(server.requests.qsize(), 2)

    def test_cache_key_binds_inputs_versions_model_parameters_endpoint_and_engine(self):
        original = identity(self.prepared, "https://unused.example/v1", 180)
        self.assertEqual(identity(self.prepared, "https://unused.example/v1", 180.0), original)
        cases = [
            ("body", "model", "other-model"), ("body", "max_output_tokens", 4096),
            ("metadata", "rule_version", "rules_changed"),
            ("metadata", "prompt_version", "judge_v2"),
            ("metadata", "prompt_sha256", "changed"),
            ("metadata", "schema_sha256", "changed"),
            ("metadata", "image_policy_sha256", "changed"),
        ]
        for group, key, value in cases:
            candidate = copy.deepcopy(self.prepared)
            candidate[group][key] = value
            self.assertNotEqual(identity(candidate, "https://unused.example/v1", 180), original)
        candidate = copy.deepcopy(self.prepared)
        candidate["body"]["input"][0]["content"][1]["image_url"] += "changed"
        self.assertNotEqual(identity(candidate, "https://unused.example/v1", 180), original)
        candidate = copy.deepcopy(self.prepared)
        candidate["metadata"]["initial_snapshot_sha256"] = "different human version"
        self.assertEqual(identity(candidate, "https://unused.example/v1", 180), original)
        self.assertNotEqual(identity(self.prepared, "https://other.example/v1", 180), original)
        self.assertNotEqual(identity(self.prepared, "https://unused.example/v1", 120), original)
        with patch("run_store.client.text_digest", return_value="changed engine source"):
            self.assertNotEqual(identity(self.prepared, "https://unused.example/v1", 180), original)

    def test_missing_cache_evidence_is_rejected_and_original_history_retained(self):
        server, url = self.serve()
        first = self.worker().run(self.create(url))
        source = self.task(first)["attempts"][0]
        (Path(source["evidence_dir"]) / "parsed.json").unlink()
        second = self.worker().run(self.create(url))
        self.assertIsNone(self.task(second)["cached_from"])
        self.assertEqual(server.requests.qsize(), 2)
        self.assertEqual(self.store.status(first["batches"][0]["id"])["batches"][0]["tasks"][0]["status"],
                         "succeeded")
        with self.store.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM events WHERE kind='cache_invalidated'").fetchone()[0], 1)

    def test_changed_cache_response_is_rejected(self):
        server, url = self.serve()
        first = self.worker().run(self.create(url))
        source = self.task(first)["attempts"][0]
        path = Path(source["evidence_dir"]) / "raw_response.bin"
        path.write_bytes(path.read_bytes() + b" ")
        second = self.worker().run(self.create(url))
        self.assertEqual(server.requests.qsize(), 2)
        self.assertIsNone(self.task(second)["cached_from"])

    def test_visual_fail_is_engineering_success_and_cacheable(self):
        answer = valid_answer(self.prepared)
        answer["labels"][0]["status"] = "fail"
        payload = response(self.prepared, json.dumps(answer))
        server, url = self.serve([(200, payload, {})])
        first = self.worker().run(self.create(url))
        stored = self.attempt(self.task(first)["id"])
        self.assertEqual(json.loads(stored["receipt_json"])["candidate_status"], "candidate_fail")
        self.worker().run(self.create(url))
        self.assertEqual(server.requests.qsize(), 1)

    def test_429_retried_and_each_attempt_retains_independent_response(self):
        server, url = self.serve([(429, {"error": {"code": "rate_limit_exceeded"}}, {"Retry-After": "0"})])
        result = self.worker().run(self.create(url))
        task = self.task(result)
        self.assertEqual(task["status"], "succeeded")
        self.assertEqual([row["status"] for row in task["attempts"]], ["failed", "succeeded"])
        self.assertEqual(task["retry_count"], 1)
        self.assertTrue(all((Path(row["evidence_dir"]) / "raw_response.bin").exists()
                            for row in task["attempts"]))
        receipt = json.loads((Path(task["attempts"][1]["evidence_dir"]) / "run.json").read_bytes())
        self.assertEqual(receipt["attempt"], 2)
        self.assertEqual(result["budget"]["reserved"], 2)
        self.assertEqual(server.requests.qsize(), 2)

    def test_network_timeout_retried_once_then_success(self):
        server, url = self.serve()
        count = 0

        def executor(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 1:
                with patch.object(client, "build_opener") as opener:
                    opener.return_value.open.side_effect = URLError("synthetic timeout")
                    return client.execute_request(*args, **kwargs)
            return client.execute_request(*args, **kwargs)

        result = self.worker(executor=executor).run(self.create(url))
        self.assertEqual(self.task(result)["status"], "succeeded")
        self.assertEqual(self.task(result)["attempts"][0]["error_code"], "transport_error")
        self.assertEqual(server.requests.qsize(), 1)
        self.assertEqual(result["budget"]["reserved"], 2)

    def test_retry_exhaustion_is_persistent_and_explicit_resume_does_not_reset_cap(self):
        server, url = self.serve([(503, {"error": {"code": "server_error"}}, {})] * 3)
        batch = self.create(url)
        first = self.worker().run(batch)
        second = self.worker().run(batch, retry_failed=True)
        self.assertEqual(self.task(first)["status"], "failed")
        self.assertEqual(len(self.task(second)["attempts"]), 3)
        self.assertEqual(self.task(second)["retry_count"], 2)
        self.assertEqual(server.requests.qsize(), 3)
        self.assertGreaterEqual(sum(self.clock.waits), 6)

    def test_auth_failure_stops_batch_and_manual_retry_preserves_history(self):
        other = build("sample_029")
        other_response = response(other)
        server, url = self.serve([(401, {"error": {"code": "invalid_api_key"}}, {}),
                                 (200, response(self.prepared), {}), (200, other_response, {})])
        batch = self.create(url, [self.prepared, other])
        first = self.worker().run(batch)
        self.assertEqual(first["batches"][0]["status"], "stopped")
        self.assertEqual(self.task(first, 1)["status"], "pending")
        self.worker().run(batch)
        self.assertEqual(server.requests.qsize(), 1)
        resumed = self.worker().run(batch, retry_failed=True)
        self.assertEqual([row["status"] for row in resumed["batches"][0]["tasks"]],
                         ["succeeded", "succeeded"])
        self.assertEqual(len(self.task(resumed)["attempts"]), 2)
        self.assertEqual(server.requests.qsize(), 3)

    def test_explicit_quota_error_does_not_retry_429(self):
        server, url = self.serve([(429, {"error": {"code": "insufficient_quota"}}, {})])
        result = self.worker().run(self.create(url))
        self.assertEqual(result["batches"][0]["status"], "stopped")
        self.assertEqual(server.requests.qsize(), 1)

    def test_invalid_json_result_gets_bounded_retry(self):
        server, url = self.serve([(200, response(self.prepared, '```json\n{}\n```'), {})])
        result = self.worker().run(self.create(url))
        self.assertEqual(self.task(result)["status"], "succeeded")
        self.assertEqual(self.task(result)["attempts"][0]["error_code"], "invalid_json")
        self.assertEqual(server.requests.qsize(), 2)

    def test_semantic_failure_exhausts_retries_without_entering_cache(self):
        answer = valid_answer(self.prepared)
        answer["labels"].pop()
        payload = response(self.prepared, json.dumps(answer))
        server, url = self.serve([(200, payload, {})] * 3)
        result = self.worker().run(self.create(url))
        self.assertEqual(self.task(result)["status"], "failed")
        self.assertEqual(len(self.task(result)["attempts"]), 3)
        with self.store.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM cache").fetchone()[0], 0)
        self.assertEqual(server.requests.qsize(), 3)

    def test_refusal_stops_only_its_task_and_continues_remaining_samples(self):
        refusal = response(self.prepared)
        refusal["output"][1]["content"] = [{"type": "refusal", "refusal": "synthetic refusal"}]
        other = build("sample_029")
        server, url = self.serve([(200, refusal, {}), (200, response(other), {})])
        result = self.worker().run(self.create(url, [self.prepared, other]))
        self.assertEqual([task["status"] for task in result["batches"][0]["tasks"]],
                         ["failed", "succeeded"])
        self.assertEqual(result["batches"][0]["status"], "completed")
        self.assertEqual(server.requests.qsize(), 2)

    def test_retry_after_is_persisted_and_long_wait_returns_without_sleeping(self):
        server, url = self.serve([(429, {"error": {"code": "rate_limit_exceeded"}}, {"Retry-After": "120"})])
        batch = self.create(url)
        first = self.worker().run(batch)
        self.assertEqual(first["batches"][0]["stop_code"], "retry_wait")
        self.assertEqual(self.task(first)["status"], "pending")
        self.assertEqual(self.clock.waits, [])
        self.assertEqual(server.requests.qsize(), 1)
        self.clock.now += 121
        resumed = self.worker().run(batch)
        self.assertEqual(self.task(resumed)["status"], "succeeded")
        self.assertEqual(server.requests.qsize(), 2)

    def test_initial_snapshot_preflight_covers_whole_batch_before_first_request(self):
        server, url = self.serve()
        batch = self.create(url, [self.prepared, build("sample_029")])

        def gate(sid):
            if sid == "sample_029":
                raise ValueError("synthetic missing initial")
            return "snapshot"

        worker = Worker(self.store, snapshot_checker=gate)
        result = worker.run(batch)
        self.assertEqual(result["batches"][0]["status"], "stopped")
        self.assertEqual(result["budget"]["reserved"], 0)
        self.assertEqual(server.requests.qsize(), 0)

    def test_bad_key_does_not_reserve_a_request(self):
        server, url = self.serve()
        with patch.dict(os.environ, {"VLM_API_KEY": ""}):
            result = self.worker().run(self.create(url))
        self.assertEqual(result["budget"]["reserved"], 0)
        self.assertEqual(server.requests.qsize(), 0)
        self.assertEqual(result["batches"][0]["stop_code"], "invalid_api_key_environment")

    def test_interrupt_before_network_releases_unsent_reservation_and_can_resume(self):
        server, url = self.serve()
        batch = self.create(url)
        with self.assertRaises(KeyboardInterrupt):
            self.worker(executor=lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt())).run(batch)
        self.assertEqual(self.store.status(batch)["budget"]["reserved"], 1)
        recovered = self.worker().recover(batch)
        self.assertEqual(recovered["budget"]["reserved"], 0)
        self.assertEqual(self.task(recovered)["attempts"][0]["status"], "interrupted")
        resumed = self.worker().run(batch)
        self.assertEqual(self.task(resumed)["status"], "succeeded")
        self.assertEqual(len(self.task(resumed)["attempts"]), 2)
        self.assertEqual(server.requests.qsize(), 1)

    def test_uncertain_sent_interrupt_is_kept_in_budget(self):
        server, url = self.serve()
        batch = self.create(url)

        def interrupted(*args, **kwargs):
            with patch.object(client, "build_opener") as opener:
                opener.return_value.open.side_effect = KeyboardInterrupt()
                return client.execute_request(*args, **kwargs)

        with self.assertRaises(KeyboardInterrupt):
            self.worker(executor=interrupted).run(batch)
        recovered = self.worker().recover(batch)
        self.assertEqual(recovered["budget"]["uncertain_or_running"], 1)
        self.assertIsNone(self.task(recovered)["attempts"][0]["request_started"])
        resumed = self.worker().run(batch)
        self.assertEqual(resumed["budget"]["reserved"], 2)
        self.assertEqual(server.requests.qsize(), 1)

    def test_complete_files_are_adopted_after_real_process_exit_without_resending(self):
        server, url = self.serve()
        batch = self.create(url)
        code = """
import os, sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
import judge_client as client
from test_judge_client import synthetic_prepare
from run_store import RunStore
from run_tasks import Worker
client.prepare_image = synthetic_prepare
def finish_and_exit(*args, **kwargs):
    client.execute_request(*args, **kwargs)
    os._exit(17)
Worker(RunStore(sys.argv[3]), executor=finish_and_exit,
       snapshot_checker=lambda sid: 'synthetic-initial-snapshot').run(sys.argv[4])
"""
        process = subprocess.run(
            [sys.executable, "-c", code, str(Path(client.__file__).parent),
             str(Path(__file__).parent), str(self.store.path), batch],
            capture_output=True, timeout=20,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        self.assertEqual(process.returncode, 17, process.stderr.decode(errors="replace"))
        self.assertEqual(self.task(self.store.status(batch))["status"], "running")
        with patch.dict(os.environ, {"VLM_API_KEY": ""}):
            result = self.worker().recover(batch)
        self.assertEqual(self.task(result)["status"], "succeeded")
        self.assertEqual(result["budget"]["known_started"], 1)
        self.assertEqual(server.requests.qsize(), 1)
        cached = self.worker().run(self.create(url))
        self.assertIsNotNone(self.task(cached)["cached_from"])
        self.assertEqual(server.requests.qsize(), 1)

    def test_os_lock_rejects_second_worker_and_is_released_by_process_exit(self):
        code = """
import os, sys
sys.path.insert(0, sys.argv[1])
from run_store import worker_lock
with worker_lock(sys.argv[2]):
    print('READY', flush=True)
    input()
    os._exit(0)
"""
        process = subprocess.Popen(
            [sys.executable, "-c", code, str(Path(client.__file__).parent), str(self.store.path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            self.assertEqual(process.stdout.readline().strip(), "READY")
            with self.assertRaises(RunError), worker_lock(self.store.path):
                pass
            process.communicate("\n", timeout=10)
            with worker_lock(self.store.path):
                pass
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)

    def test_request_cap_reservations_atomic_and_no_network_after_limit(self):
        url = "https://unused.example/v1"
        for _ in range(REQUEST_LIMIT - 1):
            batch_id = self.create(url)
            task = self.store.tasks(batch_id)[0]
            attempt_id, _ = self.store.reserve(task, self.store.batch(batch_id))
            with self.store.connect() as db:
                attempt = dict(db.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone())
            self.store.interrupt(task, attempt, retry=False)
        batches = [self.create(url) for _ in range(2)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.store.reserve, self.store.tasks(bid)[0],
                                   self.store.batch(bid)) for bid in batches]
            reservations = [future.result() for future in futures]
        self.assertEqual(sum(value is not None for value in reservations), 1)
        final_batch = self.create(url)
        with patch.object(client, "build_opener") as opener:
            result = self.worker().run(final_batch)
        opener.assert_not_called()
        self.assertEqual(result["budget"]["reserved"], 100)
        self.assertEqual(result["batches"][0]["stop_code"], "request_limit_reached")

    def test_corrupt_input_stops_before_network(self):
        server, url = self.serve()
        batch = self.create(url)
        task = self.store.tasks(batch)[0]
        Path(task["input_path"]).write_text("{}")
        result = self.worker().run(batch)
        self.assertEqual(result["batches"][0]["status"], "stopped")
        self.assertEqual(result["budget"]["reserved"], 0)
        self.assertEqual(server.requests.qsize(), 0)

    def test_echoed_credentials_never_enter_sqlite_or_attempt_files(self):
        answer = valid_answer(self.prepared)
        answer["labels"][0]["reason"] = SECRET
        payload = response(self.prepared, json.dumps(answer))
        payload["model"] = SECRET
        server, url = self.serve([(200, payload, {})])
        result = self.worker().run(self.create(url))
        self.assertEqual(self.task(result)["status"], "succeeded")
        self.assertNotIn(SECRET.encode(), self.store.path.read_bytes())
        for file in self.store.root.rglob("*"):
            if file.is_file():
                self.assertNotIn(SECRET.encode(), file.read_bytes(), str(file))

    def test_limit_and_schema_metadata_cannot_silently_reset(self):
        with self.store.connect() as db:
            db.execute("UPDATE metadata SET value='999' WHERE key='request_limit'")
        with self.assertRaises(RunError):
            RunStore(self.store.path)

    def test_legacy_d02_send_uses_same_d03_accounting_and_cache(self):
        server, url = self.serve()
        for _ in range(2):
            output = io.StringIO()
            with patch("run_tasks.RunStore", return_value=self.store), patch.object(
                    client, "settings_from_environment", return_value={
                        "base_url": url, "model": "synthetic-model"}), patch.object(
                    client, "require_initial_snapshot", return_value="synthetic-initial-snapshot"), patch.object(
                    sys, "argv", ["judge_client.py", "--sample-id", "sample_020", "--send"]), redirect_stdout(output):
                self.assertEqual(client.main(), 0)
            self.assertEqual(self.task(json.loads(output.getvalue()))["status"], "succeeded")
        self.assertEqual(server.requests.qsize(), 1)
        self.assertEqual(self.store.status()["budget"]["reserved"], 1)

    def test_recover_complete_failure_preserves_retry_deadline_without_network(self):
        server, url = self.serve([(429, {"error": {"code": "rate_limit_exceeded"}}, {"Retry-After": "120"})])
        batch = self.create(url)

        def finish_and_interrupt(*args, **kwargs):
            client.execute_request(*args, **kwargs)
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            self.worker(executor=finish_and_interrupt).run(batch)
        with patch.object(client, "build_opener") as opener:
            recovered = self.worker().recover(batch)
        opener.assert_not_called()
        task = self.task(recovered)
        self.assertEqual(task["status"], "pending")
        self.assertEqual(task["attempts"][0]["status"], "failed")
        self.assertGreaterEqual(task["next_attempt_at"] - self.clock.now, 120)
        self.assertEqual(server.requests.qsize(), 1)

    def test_retry_after_http_date_and_invalid_values(self):
        from datetime import datetime, timedelta, timezone
        from email.utils import format_datetime
        future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=180))
        self.assertTrue(178 <= client.retry_after_seconds({"Retry-After": future}) <= 180)
        for value in ("", "invalid", "-10", "NaN", "Infinity", "999999999"):
            self.assertIsNone(client.retry_after_seconds({"Retry-After": value}))

    def test_invalid_batch_scope_counts_and_duplicate_samples_rejected(self):
        for kwargs in ({"max_attempts": 0}, {"max_attempts": 4}, {"timeout": float("nan")}):
            with self.assertRaises(ValueError):
                self.create("https://unused.example/v1", **kwargs)
        with self.assertRaises(ValueError):
            self.create("https://unused.example/v1", [self.prepared, self.prepared])
        holdout = copy.deepcopy(self.prepared)
        holdout["metadata"]["split"] = "holdout"
        with self.assertRaises(ValueError):
            self.create("https://unused.example/v1", [holdout])


if __name__ == "__main__":
    unittest.main()
