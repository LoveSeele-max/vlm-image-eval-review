"""F01 wrapper tests never send requests or use the author's API credentials."""
import copy
import json
import os
import queue
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import development_v1 as dev
import launch_dev_v1 as gui


class DevelopmentV1Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ids = ("sample_008", "sample_020")
        self.reference = {"seal_sha256": "a" * 64, "reference_version": "synthetic_reference"}
        self.profile = {"base_url": "https://example.invalid/v1", "timeout": 180.0,
                        "max_attempts": 3, "model": "synthetic-model", "max_output_tokens": 8192}
        self.requests = [{"metadata": {"sample_id": sid}, "body": {"model": "synthetic-model"}}
                         for sid in self.ids]
        self.tasks = [{"sample_id": sid, "status": "pending"} for sid in self.ids]
        self.batch = {"id": "existing", "repeat": False, "tasks": self.tasks,
                      "base_url": self.profile["base_url"], "timeout": 180.0, "max_attempts": 3}
        self.store = Mock()
        self.store.path = self.root / "runs.sqlite3"
        self.store.status.return_value = {"batches": [self.batch], "budget": {"reserved": 3}}
        self.store.batch.return_value = self.batch
        self.store.tasks.return_value = self.tasks
        self.store.create.return_value = "existing"

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, batch_id=None):
        with patch.object(dev, "development_reference", return_value=(self.ids, {}, self.reference)), \
                patch.object(dev, "baseline_profile", return_value=self.profile), \
                patch.object(dev, "snapshot_guard", return_value=Mock()), \
                patch.object(dev.client, "build_request", side_effect=self.requests), \
                patch.object(dev, "worker_lock", return_value=nullcontext()), \
                patch.object(dev, "inspect_cache", side_effect=[{"reusable": False}, {"reusable": True}]), \
                patch.object(dev, "Worker") as worker:
            worker.return_value.inputs.side_effect = self.requests
            return dev.prepare_batch(self.store, batch_id)

    def test_reuses_batch_without_requests_and_preserves_reference_binding(self):
        identifier, plan, _ = self.prepare()
        self.assertEqual(identifier, "existing")
        self.assertEqual(plan["expected_new_requests_if_all_pending"], 1)
        self.store.create.assert_not_called()
        binding = self.root / "development_v1/existing/binding.json"
        self.assertEqual(json.loads(binding.read_text(encoding="utf-8"))["reference"], self.reference)
        self.prepare()
        self.assertEqual(json.loads(binding.read_text(encoding="utf-8"))["reference"], self.reference)

    def test_creates_only_when_no_full_development_batch_exists(self):
        self.store.status.return_value["batches"] = []
        self.prepare()
        self.store.create.assert_called_once_with(self.requests, self.profile["base_url"], timeout=180.0,
                                                  max_attempts=3)

    def test_rejects_repeat_wrong_samples_or_changed_profile(self):
        for change in ({"repeat": True}, {"timeout": 50}, {"base_url": "https://other.invalid/v1"}):
            self.store.batch.return_value = {**self.batch, **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.prepare("explicit")
        self.store.batch.return_value = self.batch
        self.store.tasks.return_value = self.tasks[:1]
        with self.assertRaises(ValueError):
            self.prepare("explicit")

    def test_changed_frozen_binding_not_silently_replaced(self):
        self.prepare()
        self.reference = {"seal_sha256": "b" * 64}
        with self.assertRaisesRegex(ValueError, "拒绝覆盖"):
            self.prepare()

    def test_snapshot_guard_rejects_holdout_changed_reference_or_snapshot(self):
        guard = dev.snapshot_guard({"sample_008": "snapshot"}, "a" * 64)
        with patch.object(dev, "load_registered_reference", return_value={"reference": self.reference}), \
                patch.object(dev.client, "require_initial_snapshot", return_value="snapshot"):
            self.assertEqual(guard("sample_008"), "snapshot")
            with self.assertRaises(ValueError):
                guard("sample_001")
        with patch.object(dev, "load_registered_reference", return_value={"reference": {"seal_sha256": "b" * 64}}):
            with self.assertRaisesRegex(ValueError, "版本已改变"):
                guard("sample_008")
        with patch.object(dev, "load_registered_reference", return_value={"reference": self.reference}), \
                patch.object(dev.client, "require_initial_snapshot", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "快照"):
                guard("sample_008")

    def test_billing_record_requires_balance_in_confirmed_units(self):
        path = self.root / "billing.json"
        with patch.object(dev, "BILLING", path):
            for change in ({"unit": "credits"}, {"available_balance": 0}, {"available_balance": True}, {"source": "guess"}):
                record = {"source": "user_report", "unit": "CNY", "available_balance": 10, "checked_at": "synthetic"}
                path.write_text(json.dumps({**record, **change}), encoding="utf-8")
                with self.subTest(change=change), self.assertRaises(ValueError):
                    dev.billing_check()
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(dev.billing_check()["available_balance"], 10)

    def test_missing_billing_stops_before_worker(self):
        with patch.object(dev, "billing_check", side_effect=FileNotFoundError()), \
                patch.object(dev, "prepare_batch") as prepare, patch.object(dev, "Worker") as worker:
            with self.assertRaises(FileNotFoundError):
                dev.run_batch(self.store)
            prepare.assert_not_called()
            worker.assert_not_called()

    def test_run_uses_reference_guard_and_receipt_redaction(self):
        guard = Mock()
        callback = Mock()
        result = {"batches": [{"id": "existing"}]}
        with patch.object(dev, "billing_check", return_value={"unit": "CNY"}), \
                patch.object(dev, "prepare_batch", return_value=("existing", {"reference": self.reference}, guard)), \
                patch.object(dev, "Worker") as worker, patch.object(dev.client, "write_json") as write, \
                patch.dict(os.environ, {"VLM_API_KEY": "synthetic-f01-key"}):
            worker.return_value.run.return_value = result
            self.assertEqual(dev.run_batch(self.store, on_prepared=callback), result)
            worker.assert_called_once_with(self.store, snapshot_checker=guard)
            callback.assert_called_once_with("existing")
            self.assertEqual(write.call_args.args[2], "synthetic-f01-key")

    def test_cache_probe_never_executes_or_mutates_cache(self):
        connection = MagicMock()
        connection.execute.return_value.fetchone.return_value = {"id": "attempt", "status": "succeeded"}
        self.store.connect.return_value = nullcontext(connection)
        with patch.object(dev, "identity", return_value="cache-key"), patch.object(dev, "Worker") as worker:
            worker.return_value.evidence.return_value = ({"engineering_status": "valid_result"}, {}, {})
            result = dev.inspect_cache(self.store, self.requests[0], self.batch)
            self.assertTrue(result["reusable"])
            worker.return_value.run.assert_not_called()
            worker.return_value.executor.assert_not_called()
            self.assertEqual(connection.execute.call_count, 1)

    def test_window_clears_key_and_reports_full_batch_count(self):
        window = gui.DevelopmentV1Window.__new__(gui.DevelopmentV1Window)
        window.store, window.batch_id, window.results = self.store, "existing", queue.Queue()
        result = {"batches": [{"id": "existing", "status": "completed", "stop_code": None,
                              "tasks": [{"status": "succeeded", "cached_from": "cached"} for _ in range(10)]}]}
        def execute(*args, **kwargs):
            self.assertEqual(os.environ["VLM_API_KEY"], "synthetic-f01-key")
            return result
        with patch.dict(os.environ, {}, clear=True), patch.object(gui.dev, "run_batch", side_effect=execute):
            window.run("synthetic-f01-key")
            self.assertNotIn("VLM_API_KEY", os.environ)
        self.assertIn("10/10", window.results.get_nowait())

    def test_window_failure_never_displays_secret(self):
        window = gui.DevelopmentV1Window.__new__(gui.DevelopmentV1Window)
        window.store, window.batch_id, window.results = self.store, "existing", queue.Queue()
        with patch.dict(os.environ, {}, clear=True), patch.object(gui.dev, "run_batch", side_effect=RuntimeError("synthetic-secret")):
            window.run("synthetic-secret")
            self.assertNotIn("VLM_API_KEY", os.environ)
        self.assertNotIn("synthetic-secret", window.results.get_nowait())


if __name__ == "__main__":
    unittest.main()
