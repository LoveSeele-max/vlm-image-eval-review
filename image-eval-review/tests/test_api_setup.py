import json
import base64
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
from urllib.error import HTTPError
from urllib.request import Request, build_opener

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api_connection import CLIENT_USER_AGENT, ConnectionTestError, normalize_base_url, test_connection
from configure_api import SettingsWindow
from capability_probe import TEST_TYPE, create_challenge, test_capabilities

SECRET = "fake-key-for-local-tests"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.server.requests.put({
            "path": self.path,
            "authorization": self.headers.get("Authorization"),
            "user_agent": self.headers.get("User-Agent"),
            "accept": self.headers.get("Accept"),
            "body": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
        })
        if self.server.reject_library_agent and self.headers.get("User-Agent", "").startswith("Python-urllib"):
            self.send_response(403)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error_code":1010,"error_name":"browser_signature_banned"}')
            return
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
        self.server.reject_library_agent = False
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
        self.assertEqual(request["user_agent"], CLIENT_USER_AGENT)
        self.assertEqual(request["accept"], "application/json")
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

    def test_descriptive_client_agent_passes_library_agent_filter(self):
        self.server.reject_library_agent = True
        old_request = Request(
            self.base_url + "/responses",
            data=b'{"model":"test-model","input":"test"}',
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(HTTPError) as context:
            build_opener().open(old_request, timeout=2)
        self.assertEqual(context.exception.code, 403)
        context.exception.close()
        receipt = test_connection(self.base_url, "test-model")
        self.assertTrue(receipt["success"])
        self.assertEqual(self.server.requests.qsize(), 2)

    def test_cloudflare_1010_is_not_misreported_as_model_permission(self):
        self.server.response_code = 403
        self.set_payload({
            "error_code": 1010, "error_name": "browser_signature_banned",
            "detail": "unnecessary raw content: " + SECRET,
        })
        with self.assertRaises(ConnectionTestError) as context:
            test_connection(self.base_url, "test-model")
        message = str(context.exception)
        self.assertIn("Cloudflare 1010", message)
        self.assertIn("尚不能据此判断", message)
        self.assertNotIn(SECRET, message)
        self.assertNotIn("unnecessary raw content", message)
        self.assertEqual(self.server.requests.qsize(), 1)

    def test_generic_403_preserves_redacted_provider_diagnostic(self):
        self.server.response_code = 403
        self.set_payload({"error": {
            "message": "Model access denied; token=" + SECRET + "; Bearer another-secret; sk-other-secret",
        }})
        with self.assertRaises(ConnectionTestError) as context:
            test_connection(self.base_url, "test-model")
        message = str(context.exception)
        self.assertIn("可能是网关拦截或权限限制", message)
        self.assertIn("Model access denied", message)
        self.assertNotIn(SECRET, message)
        self.assertNotIn("another-secret", message)
        self.assertNotIn("sk-other-secret", message)

    def test_html_gateway_response_is_not_dumped_into_error(self):
        self.server.response_code = 403
        self.server.response_body = ("<html>private gateway page " + SECRET + "</html>").encode()
        with self.assertRaises(ConnectionTestError) as context:
            test_connection(self.base_url, "test-model")
        message = str(context.exception)
        self.assertIn("403", message)
        self.assertNotIn("<html>", message)
        self.assertNotIn(SECRET, message)

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

    def capability_answer(self, challenge):
        return {"images": [
            {key: image[key] for key in ("image_id", "color", "shape", "count")}
            for image in challenge["images"]
        ]}

    def set_capability_response(self, answer):
        self.payload["output"][1]["content"][0]["text"] = (
            answer if isinstance(answer, str) else json.dumps(answer))
        self.set_payload(self.payload)

    def test_three_images_strict_schema_and_validated_receipt(self):
        challenge = create_challenge()
        answer = self.capability_answer(challenge)
        self.set_capability_response(answer)
        receipt = test_capabilities(self.base_url, "test-model", challenge)
        request = self.server.requests.get_nowait()
        body = request["body"]
        self.assertEqual(request["user_agent"], CLIENT_USER_AGENT)
        self.assertEqual(request["path"], "/v1/responses")
        self.assertEqual(request["authorization"], "Bearer " + SECRET)
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertEqual(body["text"]["format"]["type"], "json_schema")
        self.assertFalse(body["text"]["format"]["schema"]["additionalProperties"])
        content = body["input"][0]["content"]
        self.assertEqual(len(content), 4)
        self.assertEqual(content[0]["type"], "input_text")
        self.assertNotIn(json.dumps(answer), content[0]["text"])
        for image, block in zip(challenge["images"], content[1:]):
            self.assertEqual(block["type"], "input_image")
            self.assertEqual(block["detail"], "low")
            data_url = block["image_url"]
            self.assertTrue(data_url.startswith("data:image/png;base64,"))
            self.assertEqual(data_url.split(",", 1)[1], image["png_base64"])
            self.assertTrue(base64.b64decode(data_url.split(",", 1)[1]).startswith(b"\x89PNG"))
        self.assertEqual(receipt["test_type"], TEST_TYPE)
        self.assertTrue(receipt["request_completed"])
        self.assertTrue(receipt["schema_valid"])
        self.assertTrue(receipt["visual_match"])
        self.assertTrue(receipt["success"])
        self.assertEqual(receipt["schema_enforcement"], "not_proven_by_smoke_test")
        self.assertEqual(receipt["usage"]["total_tokens"], 29)
        self.assertEqual(receipt["expected"], answer["images"])
        self.assertNotIn("png_base64", json.dumps(receipt))
        self.assertNotIn(SECRET, json.dumps(receipt))
        self.assertTrue(self.server.requests.empty())

    def test_valid_json_wrong_color_count_order_and_duplicate_id_fail_visual_check(self):
        challenge = create_challenge()
        for failure in ("color", "count", "order", "duplicate_id"):
            answer = self.capability_answer(challenge)
            if failure == "color":
                answer["images"][0]["color"] = (
                    "red" if answer["images"][0]["color"] != "red" else "blue")
            elif failure == "count":
                answer["images"][0]["count"] = answer["images"][0]["count"] % 3 + 1
            elif failure == "order":
                answer["images"].reverse()
            else:
                answer["images"][1]["image_id"] = 1
            self.set_capability_response(answer)
            with self.subTest(failure=failure):
                receipt = test_capabilities(self.base_url, "test-model", challenge)
                self.assertTrue(receipt["schema_valid"])
                self.assertFalse(receipt["visual_match"])
                self.assertFalse(receipt["success"])
                self.assertTrue(any(not item["match"] for item in receipt["checks"]))

    def test_malformed_json_preserves_usage_without_claiming_vision_failure(self):
        challenge = create_challenge()
        self.set_capability_response("```json\n{}\n```")
        receipt = test_capabilities(self.base_url, "test-model", challenge)
        self.assertTrue(receipt["request_completed"])
        self.assertFalse(receipt["schema_valid"])
        self.assertIsNone(receipt["visual_match"])
        self.assertFalse(receipt["success"])
        self.assertEqual(receipt["usage"]["total_tokens"], 29)
        self.assertEqual(self.server.requests.qsize(), 1)

    def test_refusal_is_not_treated_as_an_answer(self):
        self.payload["output"][1]["content"] = [{"type": "refusal", "refusal": SECRET}]
        self.set_payload(self.payload)
        with self.assertRaisesRegex(ConnectionTestError, "拒绝") as context:
            test_capabilities(self.base_url, "test-model", create_challenge())
        self.assertNotIn(SECRET, str(context.exception))
        self.assertEqual(self.server.requests.qsize(), 1)

    def test_gui_capability_previews_receipt_and_no_duplicate_requests(self):
        challenge = create_challenge()
        self.set_capability_response(self.capability_answer(challenge))
        with tempfile.TemporaryDirectory() as folder:
            root, app = self.make_window(folder)
            with patch("configure_api.create_challenge", return_value=challenge):
                app.start_capability_test()
                app.start_capability_test()
                app.start_test()
            self.assertEqual(len(app.previews), 3)
            self.assertTrue(all(image.width() == 128 for image in app.previews))
            self.assertTrue(app.test_button.instate(["disabled"]))
            self.assertTrue(app.capability_button.instate(["disabled"]))
            self.finish_window_request(root, app)
            self.assertEqual(app.state.get(), "三图识别与 JSON 格式检查通过")
            self.assertIn("JSON 本地校验：通过", app.output.get("1.0", "end"))
            self.assertIn("图 3：通过", app.output.get("1.0", "end"))
            self.assertFalse(app.capability_button.instate(["disabled"]))
            receipt = app.capability_receipt_path.read_text(encoding="utf-8")
            self.assertNotIn(SECRET, receipt)
            self.assertTrue(json.loads(receipt)["success"])
            self.assertFalse(app.receipt_path.exists())
            self.assertEqual(self.server.requests.qsize(), 1)
            app.close()

    def test_gui_capability_format_and_http_failure_preserve_text_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root, app = self.make_window(folder)
            app.start_test()
            self.finish_window_request(root, app)
            text_receipt = app.receipt_path.read_bytes()
            self.set_capability_response('{"images":[]}')
            app.start_capability_test()
            self.finish_window_request(root, app)
            self.assertEqual(app.state.get(), "请求已完成，但能力检查未通过")
            self.assertIn("JSON 本地校验：未通过", app.output.get("1.0", "end"))
            self.assertIn("三图识别核对：未执行", app.output.get("1.0", "end"))
            self.assertFalse(json.loads(app.capability_receipt_path.read_text(encoding="utf-8"))["success"])
            self.server.response_code = 400
            self.set_payload({"error": {"message": "unsupported json_schema " + SECRET}})
            app.start_capability_test()
            self.finish_window_request(root, app)
            self.assertEqual(app.state.get(), "多图＋JSON 测试未完成")
            self.assertIn("unsupported json_schema", app.output.get("1.0", "end"))
            failure_text = app.capability_receipt_path.read_text(encoding="utf-8")
            self.assertNotIn(SECRET, failure_text)
            failure = json.loads(failure_text)
            self.assertFalse(failure["request_completed"])
            self.assertIsNone(failure["schema_valid"])
            self.assertIsNone(failure["visual_match"])
            self.assertEqual(app.receipt_path.read_bytes(), text_receipt)
            self.assertFalse(app.capability_button.instate(["disabled"]))
            self.assertEqual(self.server.requests.qsize(), 3)
            app.close()


if __name__ == "__main__":
    unittest.main()
