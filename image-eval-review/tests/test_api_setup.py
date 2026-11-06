import json
import os
import queue
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api_connection import ConnectionTestError, normalize_base_url, test_connection
from configure_api import SettingsWindow

SECRET = "fake-key-for-local-tests"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.server.requests.put({
            "path": self.path,
            "authorization": self.headers.get("Authorization"),
            "body": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
        })
        self.send_response(self.server.response_code)
        if self.server.redirect:
            self.send_header("Location", self.server.redirect)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.server.response_body)

    def log_message(self, *args):
        pass


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.response_code = 200
        self.server.redirect = None
        self.server.requests = queue.Queue()
        self.payload = {
            "status": "completed",
            "model": "returned-test-model",
            "output": [
                {"type": "reasoning"},
                {"type": "message", "content": [{"type": "output_text", "text": "连接成功"}]},
            ],
            "usage": {
                "input_tokens": 25,
                "input_tokens_details": {"cached_tokens": 10},
                "output_tokens": 4,
                "total_tokens": 29,
            },
        }
        self.set_payload(self.payload)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        self.environment = patch.dict(os.environ, {
            "VLM_API_KEY": SECRET, "VLM_BASE_URL": self.base_url, "VLM_MODEL": "test-model",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def set_payload(self, payload):
        self.server.response_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def test_real_http_post_and_receipt(self):
        receipt = test_connection(self.base_url, "test-model")
        request = self.server.requests.get_nowait()
        self.assertEqual(request["path"], "/v1/responses")
        self.assertEqual(request["authorization"], "Bearer " + SECRET)
        self.assertEqual(request["body"]["input"], "只回复：连接成功")
        self.assertFalse(request["body"]["stream"])
        self.assertEqual(receipt["text"], "连接成功")
        self.assertEqual(receipt["returned_model"], "returned-test-model")
        self.assertEqual(receipt["usage"]["total_tokens"], 29)
        self.assertEqual(receipt["test_type"], "text_connection_only")
        self.assertNotIn(SECRET, json.dumps(receipt))

    def test_endpoint_normalization_without_guessing_v1(self):
        self.assertEqual(normalize_base_url(self.base_url + "/responses/"), self.base_url)
        self.assertEqual(normalize_base_url(" https://relay.example/custom/ "), "https://relay.example/custom")
        for bad in ("", "relay.example/v1", "https://user:pass@relay.example",
                    "https://relay.example/v1?key=abc", "https://relay.example/#key",
                    "https://relay.example:bad", "https://relay.example/v1/chat/completions"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalize_base_url(bad)

    def test_missing_or_invalid_key_sends_no_request(self):
        for key in ("", "key\ninjected"):
            os.environ["VLM_API_KEY"] = key
            with self.subTest(key=key), self.assertRaises(ValueError):
                test_connection(self.base_url, "test-model")
        self.assertTrue(self.server.requests.empty())

    def test_auth_failure_does_not_expose_provider_body_or_key(self):
        self.server.response_code = 401
        self.server.response_body = ("private error: " + SECRET).encode()
        with self.assertRaises(ConnectionTestError) as context:
            test_connection(self.base_url, "test-model")
        self.assertIn("401", str(context.exception))
        self.assertNotIn(SECRET, str(context.exception))
        self.assertEqual(self.server.requests.qsize(), 1)

    def test_redirect_is_not_followed(self):
        self.server.response_code = 307
        self.server.redirect = self.base_url + "/another-endpoint"
        with self.assertRaises(ConnectionTestError):
            test_connection(self.base_url, "test-model")
        self.assertEqual(self.server.requests.qsize(), 1)

    def test_invalid_and_incomplete_responses_fail(self):
        cases = (
            b"<html>login</html>",
            b"[]",
            b'{"status":"failed","error":{"message":"failed"}}',
            b'{"status":"incomplete","output":[]}',
            b'{"status":"completed","output":"wrong-type"}',
            b'{"status":"completed","output":[]}',
        )
        for body in cases:
            self.server.response_body = body
            with self.subTest(body=body), self.assertRaises(ConnectionTestError):
                test_connection(self.base_url, "test-model")

    def test_timeout_has_actionable_error(self):
        with patch("api_connection.build_opener") as opener:
            opener.return_value.open.side_effect = TimeoutError()
            with self.assertRaisesRegex(ConnectionTestError, "超时"):
                test_connection(self.base_url, "test-model")

    def test_receipt_redacts_echoed_key_and_ignores_unneeded_payload(self):
        self.payload["model"] = "echo:" + SECRET
        self.payload["output"][1]["content"][0]["text"] = SECRET
        self.payload["extra_private_field"] = SECRET
        self.set_payload(self.payload)
        receipt = test_connection(self.base_url, "test-model")
        self.assertNotIn(SECRET, json.dumps(receipt))
        self.assertEqual(receipt["text"], "[REDACTED]")
        self.assertNotIn("extra_private_field", receipt)

    def test_absent_usage_is_unknown_not_zero(self):
        del self.payload["usage"]
        self.set_payload(self.payload)
        receipt = test_connection(self.base_url, "test-model")
        self.assertIsNone(receipt["usage"]["total_tokens"])

    def make_window(self, folder):
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        app = SettingsWindow(root, Path(folder) / "config.local.json", Path(folder) / "receipt.json")
        self.addCleanup(lambda: root.destroy() if not app.closed else None)
        return root, app

    def finish_window_request(self, root, app):
        deadline = time.monotonic() + 5
        while app.busy and time.monotonic() < deadline:
            root.update()
            time.sleep(0.01)
        self.assertFalse(app.busy, "GUI request did not finish")

    def test_gui_save_test_and_restore_without_saving_key(self):
        with tempfile.TemporaryDirectory() as folder:
            root, app = self.make_window(folder)
            app.start_test()
            app.start_test()
            self.assertTrue(app.busy)
            self.assertTrue(app.test_button.instate(["disabled"]))
            self.finish_window_request(root, app)
            self.assertEqual(app.state.get(), "文字连接成功")
            self.assertIn("连接成功", app.output.get("1.0", "end"))
            self.assertFalse(app.test_button.instate(["disabled"]))
            config_text = app.config_path.read_text(encoding="utf-8")
            self.assertEqual(set(json.loads(config_text)), {"base_url", "model"})
            self.assertNotIn(SECRET, config_text)
            self.assertNotIn(SECRET, app.receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(self.server.requests.qsize(), 1)
            settings, error = app.load_settings()
            self.assertEqual(settings["model"], "test-model")
            self.assertFalse(error)
            app.close()

    def test_gui_failure_replaces_previous_success_and_restores_controls(self):
        self.server.response_code = 401
        with tempfile.TemporaryDirectory() as folder:
            root, app = self.make_window(folder)
            app.receipt_path.write_text('{"success": true}', encoding="utf-8")
            app.start_test()
            self.finish_window_request(root, app)
            self.assertEqual(app.state.get(), "连接未成功")
            self.assertIn("401", app.output.get("1.0", "end"))
            receipt = app.receipt_path.read_text(encoding="utf-8")
            self.assertFalse(json.loads(receipt)["success"])
            self.assertNotIn(SECRET, receipt)
            self.assertFalse(app.test_button.instate(["disabled"]))
            app.close()


if __name__ == "__main__":
    unittest.main()
