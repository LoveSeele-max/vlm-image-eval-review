import os
import queue
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import launch_development as launch


def batch(identifier, *, repeat=False):
    return {"id": identifier, "repeat": repeat,
            "tasks": [{"sample_id": sid} for sid in launch.SAMPLES]}


class DevelopmentLauncherTests(unittest.TestCase):
    def test_reuses_existing_batch_and_checks_all_inputs_and_snapshots(self):
        store = Mock()
        store.status.return_value = {"batches": [batch("original"), batch("repeat", repeat=True)]}
        store.batch.return_value = batch("original")
        store.tasks.return_value = batch("original")["tasks"]
        with patch.object(launch, "Worker") as worker, patch.object(
                launch.client, "require_initial_snapshot") as snapshot:
            self.assertEqual(launch.prepare_batch(store), "original")
        store.create.assert_not_called()
        self.assertEqual(snapshot.call_count, 3)
        self.assertEqual(worker.return_value.inputs.call_count, 3)
        worker.return_value.run.assert_not_called()

    def test_rejects_other_samples_and_repeat_before_any_execution(self):
        for invalid in [batch("repeat", repeat=True), {"repeat": False, "tasks": []}]:
            store = Mock()
            store.batch.return_value = invalid
            store.tasks.return_value = invalid["tasks"]
            with patch.object(launch, "Worker") as worker:
                with self.assertRaises(launch.RunError):
                    launch.prepare_batch(store, "explicit")
                worker.assert_not_called()

    def test_missing_confirmation_does_not_run(self):
        store = Mock()
        store.batch.return_value = batch("original")
        store.tasks.return_value = batch("original")["tasks"]
        with patch.object(launch, "Worker") as worker, patch.object(
                launch.client, "require_initial_snapshot", side_effect=ValueError("missing")):
            with self.assertRaises(ValueError):
                launch.prepare_batch(store, "original")
            worker.return_value.run.assert_not_called()

    def window(self):
        window = launch.DevelopmentWindow.__new__(launch.DevelopmentWindow)
        window.store = Mock()
        window.batch_id = "original"
        window.results = queue.Queue()
        return window

    def test_key_available_only_during_worker_and_redacted_receipt_write(self):
        window = self.window()
        result = {"batches": [{"status": "completed", "stop_code": None,
                              "tasks": [{"status": "succeeded"} for _ in launch.SAMPLES]}]}
        secret = "synthetic-d04-test-key"
        def execute(identifier):
            self.assertEqual(os.environ["VLM_API_KEY"], secret)
            return result
        with patch.dict(os.environ, {}, clear=True), patch.object(
                launch, "prepare_batch", return_value="original"), patch.object(
                launch, "Worker") as worker, patch.object(launch.client, "write_json") as write:
            worker.return_value.run.side_effect = execute
            window.run(secret)
            self.assertNotIn("VLM_API_KEY", os.environ)
            write.assert_called_once_with(launch.STATUS_PATH, result, secret)
        self.assertIn("3/3", window.results.get_nowait())

    def test_unexpected_error_clears_key_and_hides_exception_text(self):
        window = self.window()
        secret = "synthetic-d04-test-key"
        with patch.dict(os.environ, {}, clear=True), patch.object(
                launch, "prepare_batch", return_value="original"), patch.object(
                launch, "Worker") as worker:
            worker.return_value.run.side_effect = RuntimeError(secret)
            window.run(secret)
            self.assertNotIn("VLM_API_KEY", os.environ)
        self.assertNotIn(secret, window.results.get_nowait())


if __name__ == "__main__":
    unittest.main()
