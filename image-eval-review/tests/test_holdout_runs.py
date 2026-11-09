"""G01 isolated accounting and local response replay; no external API calls."""
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import holdout_runs as ho
import judge_client as client
from run_store import RunError, identity
from run_tasks import Worker
from test_judge_client import synthetic_prepare, response

REAL_GATES = ho.gates


class Replay(io.BytesIO):
    status = 200
    headers = {}


class HoldoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ho.HoldoutStore(Path(self.temp.name) / "runs.sqlite3")
        # Use committed rules and synthetic pixels, never this machine's private labels/billing.
        seal = ho.pv.verify_registry()
        ids = ho.pv.read_json(ho.DATA / "split.json")["partitions"]["holdout"]["sample_ids"]
        self.frozen = {"sample_ids": ids, "snapshots": {sid: "synthetic-" + sid for sid in ids},
            "reference": {"reference_version": "synthetic_reference", "seal_sha256": "synthetic_seal"},
            "profile": copy.deepcopy(seal["comparison_profile"]), "base_url": "https://example.invalid/v1",
            "registry_sha256": ho.text_digest(ho.pv.REGISTRY), "runner_sha256": ho.text_digest(Path(ho.__file__))}
        self.profile = self.frozen["profile"]
        self.calls = []
        self.prepared = {}
        real_build = ho.pv.build_versioned_request

        def build(sid, model, tokens, *, prompt_version):
            key = (sid, prompt_version)
            if key not in self.prepared:
                with patch.object(client, "prepare_image", side_effect=synthetic_prepare):
                    self.prepared[key] = real_build(sid, model, tokens, prompt_version=prompt_version)
            return copy.deepcopy(self.prepared[key])

        for p in (patch.object(ho, "gates", side_effect=lambda: copy.deepcopy(self.frozen)),
                  patch.object(ho.pv, "build_versioned_request", side_effect=build),
                  patch.object(ho, "billing_check", return_value={"source": "synthetic", "unit": "CNY", "available_balance": 10}),
                  patch.dict(os.environ, {"VLM_API_KEY": "fake-g01-secret"})):
            p.start()
            self.addCleanup(p.stop)

    def executor(self, prepared, url, **kwargs):
        self.calls.append((prepared["metadata"]["sample_id"], prepared["metadata"]["prompt_version"]))
        payload = json.dumps(response(prepared)).encode()
        with patch.object(client, "build_opener") as opener:
            opener.return_value.open.return_value = Replay(payload)
            return client.execute_request(prepared, url, **kwargs)

    def worker(self, **kwargs):
        return ho.HoldoutWorker(self.store, executor=kwargs.get("executor", self.executor))

    def test_preparation_separates_versions_and_is_idempotent_no_requests(self):
        plan = self.store.prepare_pair()
        self.assertEqual(plan, self.store.prepare_pair())
        self.assertEqual(len(set(plan["batches"].values())), 2)
        self.assertEqual(len(self.store.status()["batches"]), 2)
        self.assertEqual(self.store.status()["budget"]["reserved"], 0)
        for version, bid in plan["batches"].items():
            tasks = self.store.tasks(bid)
            self.assertEqual([t["sample_id"] for t in tasks], self.frozen["sample_ids"])
            self.assertEqual(len(tasks), 30)
            meta = json.loads(tasks[0]["spec_json"])["metadata"]
            self.assertEqual(meta["split"], "holdout")
            self.assertEqual(meta["prompt_version"], version)

    def test_gate_failure_creates_no_tasks(self):
        with patch.object(ho, "gates", side_effect=RunError("freeze mismatch")):
            with self.assertRaises(RunError):
                self.store.prepare_pair()
        self.assertEqual(self.store.status()["batches"], [])

    def test_actual_gate_rejects_wrong_human_freeze(self):
        with patch.object(ho, "load_registered_reference", return_value={"reference": self.frozen["reference"]}):
            with self.assertRaisesRegex(RunError, "人工参考"):
                REAL_GATES()

    def test_actual_gate_rejects_missing_snapshot_or_changed_model(self):
        seal = ho.pv.verify_registry()
        ref = {"seal_sha256": seal["human_reference_seal_sha256"], "reference_version": seal["human_reference_version"]}
        rows = [{"split": "holdout", "sample_id": sid, "snapshot_sha256": digest}
                for sid, digest in self.frozen["snapshots"].items()]
        with patch.object(ho, "load_registered_reference", return_value={"reference": ref, "initial_rows": rows[:-1]}):
            with self.assertRaisesRegex(RunError, "30"):
                REAL_GATES()
        with patch.object(ho, "load_registered_reference", return_value={"reference": ref, "initial_rows": rows}), \
                patch.object(client, "settings_from_environment", return_value={"model": "changed", "base_url": "https://example.invalid/v1"}):
            with self.assertRaisesRegex(RunError, "接口或模型"):
                REAL_GATES()

    def test_billing_failure_precedes_batch_mutation(self):
        with patch.object(ho, "billing_check", side_effect=RunError("billing")):
            with self.assertRaises(RunError): self.store.prepare_pair()
        self.assertEqual(self.store.status()["batches"], [])

    def test_insufficient_shared_budget_prevents_pair(self):
        original = self.store.status
        def status(*args):
            result = original(*args)
            result["budget"]["reserved"] = 41
            return result
        with patch.object(self.store, "status", side_effect=status):
            with self.assertRaisesRegex(RunError, "60"):
                self.store.prepare_pair()
        self.assertEqual(original()["batches"], [])

    def test_parity_rejects_parameter_or_image_change(self):
        self.store.prepare_pair()
        sid = self.frozen["sample_ids"][0]
        a, b = [copy.deepcopy(self.prepared[(sid, v)]) for v in ho.VERSIONS]
        b["body"]["max_output_tokens"] += 1
        with self.assertRaises(RunError): ho.parity(a, b)

    def test_old_worker_does_not_accept_v2_and_new_worker_rebuilds_version(self):
        plan = self.store.prepare_pair()
        bid = plan["batches"]["judge_v2"]
        task = self.store.tasks(bid)[0]
        with self.assertRaises(RunError): Worker(self.store).inputs(task, self.store.batch(bid))
        request = self.worker().inputs(task, self.store.batch(bid))
        self.assertEqual(request["metadata"]["prompt_version"], "judge_v2")
        old = self.prepared[(task["sample_id"], "judge_v1")]
        self.assertNotEqual(identity(old, self.frozen["base_url"], 180), task["cache_key"])

    def test_frozen_binding_change_rejects_before_send(self):
        plan = self.store.prepare_pair()
        self.frozen["registry_sha256"] = "changed"
        result = self.worker().run(plan["batches"]["judge_v1"])
        self.assertEqual(result["batches"][0]["stop_code"], "input_or_initial_snapshot_check_failed")
        self.assertEqual(self.calls, [])

    def test_swapped_batch_versions_rejected(self):
        plan = self.store.prepare_pair()
        plan["batches"] = dict(zip(ho.VERSIONS, reversed(list(plan["batches"].values()))))
        client.write_json(self.store.plan_path, plan)
        with self.assertRaises(RunError): self.store.binding()

    def test_corrupt_input_stops_without_request(self):
        plan = self.store.prepare_pair()
        bid = plan["batches"]["judge_v1"]
        Path(self.store.tasks(bid)[0]["input_path"]).write_bytes(b"{}")
        result = self.worker().run(bid)
        self.assertEqual(result["batches"][0]["stop_code"], "input_or_initial_snapshot_check_failed")
        self.assertEqual(self.calls, [])

    def test_two_versions_execute_independently_and_resume_costs_zero(self):
        plan = self.store.prepare_pair()
        for version in ho.VERSIONS:
            result = self.worker().run(plan["batches"][version])
            self.assertEqual(result["batches"][0]["status"], "completed")
        self.assertEqual(len(self.calls), 60)
        self.assertEqual(self.store.status()["budget"]["reserved"], 60)
        self.assertEqual(set(self.calls), {(sid, v) for sid in self.frozen["sample_ids"] for v in ho.VERSIONS})
        result = ho.verify_results(self.store)
        self.assertTrue(result["both_versions_terminal"])
        for version in ho.VERSIONS:
            self.assertEqual(result["versions"][version]["counts"], {"succeeded": 30})
            self.worker().run(plan["batches"][version])
        self.assertEqual(len(self.calls), 60)
        self.assertNotIn(b"fake-g01-secret", self.store.path.read_bytes())

    def test_524_is_final_failure_and_remains_missing_without_reopening(self):
        plan = self.store.prepare_pair()
        bid = plan["batches"]["judge_v1"]
        def fail_first(prepared, url, **kwargs):
            if prepared["metadata"]["sample_id"] == self.frozen["sample_ids"][0]:
                with patch.object(client, "build_opener") as opener:
                    opener.return_value.open.side_effect = HTTPError(url, 524, "timeout", {}, io.BytesIO(b"timeout"))
                    return client.execute_request(prepared, url, **kwargs)
            return self.executor(prepared, url, **kwargs)
        result = self.worker(executor=fail_first).run(bid)
        self.assertEqual(result["batches"][0]["tasks"][0]["status"], "failed")
        attempts = result["batches"][0]["tasks"][0]["attempts"]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["error_code"], "http_524")
        before = self.store.status()["budget"]["reserved"]
        self.worker().run(bid)
        self.assertEqual(self.store.status()["budget"]["reserved"], before)
        verified = ho.verify_results(self.store)
        self.assertFalse(verified["both_versions_terminal"])
        self.assertGreater(verified["engineering_missing_items"]["judge_v1"], 0)

    def test_account_error_stops_pair_before_second_version(self):
        plan = self.store.prepare_pair()
        def denied(prepared, url, **kwargs):
            with patch.object(client, "build_opener") as opener:
                opener.return_value.open.side_effect = HTTPError(url, 401, "denied", {}, io.BytesIO(b"denied"))
                return client.execute_request(prepared, url, **kwargs)
        cls = ho.HoldoutWorker
        with patch.object(ho, "HoldoutWorker", side_effect=lambda store: cls(store, executor=denied)):
            verified = ho.run_pair(self.store)
        self.assertFalse(verified["both_versions_terminal"])
        self.assertEqual(verified["versions"]["judge_v1"]["status"], "stopped")
        self.assertEqual(verified["versions"]["judge_v2"]["counts"], {"pending": 30})
        self.assertEqual(self.store.status()["budget"]["reserved"], 1)
        self.assertEqual(self.store.batch(plan["batches"]["judge_v1"])["stop_code"], "http_401")

    def test_completed_response_recovery_and_shared_limit(self):
        plan = self.store.prepare_pair()
        bid = plan["batches"]["judge_v2"]
        task, batch = self.store.tasks(bid)[0], self.store.batch(bid)
        request = self.worker().inputs(task, batch)
        aid, number = self.store.reserve(task, batch)
        sending = copy.deepcopy(request)
        sending["metadata"]["initial_snapshot_sha256"] = self.frozen["snapshots"][task["sample_id"]]
        self.executor(sending, batch["base_url"], send=True, output_root=self.store.evidence,
                      timeout=180, run_id=aid, attempt_number=number)
        self.worker().recover(bid)
        self.assertEqual(self.store.tasks(bid)[0]["status"], "succeeded")
        self.assertEqual(self.store.status()["budget"]["reserved"], 1)
        with self.store.connect(immediate=True) as db:
            db.execute("UPDATE attempts SET budget_reserved=100")
        result = self.worker().run(plan["batches"]["judge_v1"])
        self.assertEqual(result["batches"][0]["stop_code"], "request_limit_reached")
        self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main()
