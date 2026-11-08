import base64
import copy
import hashlib
import io
import json
import os
import queue
import sqlite3
import sys
import tempfile
import threading
import unittest
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import judge_client as client
from annotation_store import AnnotationStore
from build_manifest import DATA, read_jsonl


SECRET = "fake-d02-secret"


def synthetic_prepare(path, role, policy, expected_sha256):
    output = io.BytesIO()
    color = {"REF-MAIN": "red", "REF-SUPPLEMENT": "blue", "CANDIDATE": "green"}[role]
    Image.new("RGB", (8, 8), color).save(output, format="PNG")
    raw = output.getvalue()
    return raw, {"role": role, "source_sha256": expected_sha256,
                 "output_sha256": hashlib.sha256(raw).hexdigest(), "output_size": [8, 8],
                 "policy_id": policy["policy_id"], "pillow_version": "test-fixture"}


def build(sid="sample_020"):
    with patch.object(client, "prepare_image", side_effect=synthetic_prepare):
        return client.build_request(sid, "synthetic-model")


def valid_answer(prepared):
    return {
        "sample_id": prepared["metadata"]["sample_id"],
        "labels": [{"criterion_id": key, "status": "pass", "candidate_region": "synthetic region",
                    "candidate_observation": "synthetic observation", "reason": "synthetic reason"}
                   for key in prepared["expected"]],
    }


def response(prepared, text=None):
    return {
        "id": "synthetic-response", "model": "returned-synthetic-model", "status": "completed",
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text",
                          "text": text if text is not None else json.dumps(valid_answer(prepared))}]},
        ],
        "usage": {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300},
    }


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.server.requests.put({
            "path": self.path, "authorization": self.headers.get("Authorization"),
            "body": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
        })
        self.send_response(self.server.code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(self.server.declared_length or len(self.server.raw)))
        if self.server.code == 302:
            self.send_header("Location", self.server.redirect)
        self.end_headers()
        self.wfile.write(self.server.raw)

    def log_message(self, *args):
        pass


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.prepared = build()
        self.environment = patch.dict(os.environ, {"VLM_API_KEY": SECRET})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def serve(self, payload=None, raw=None, code=200, declared_length=None):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.raw = raw if raw is not None else json.dumps(
            payload if payload is not None else response(self.prepared)).encode()
        server.code, server.requests = code, queue.Queue()
        server.declared_length = declared_length
        server.redirect = f"http://127.0.0.1:{server.server_port}/redirected"
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server, f"http://127.0.0.1:{server.server_port}/v1"

    def run_request(self, url, send=True, **kwargs):
        return client.execute_request(self.prepared, url, send=send,
                                      output_root=self.folder / "runs", **kwargs)

    def test_build_all_40_exact_criteria_and_anonymous_three_image_order(self):
        total = 0
        for item in read_jsonl(DATA / "manifest.jsonl"):
            prepared = build(item["sample_id"])
            total += len(prepared["expected"])
            content = prepared["body"]["input"][0]["content"]
            self.assertEqual([row["role"] for row in prepared["metadata"]["images"]],
                             ["REF-MAIN", "REF-SUPPLEMENT", "CANDIDATE"])
            self.assertEqual([row["detail"] for row in content[1:]], ["high"] * 3)
            for image, receipt in zip(content[1:], prepared["metadata"]["images"]):
                raw = base64.b64decode(image["image_url"].split(",", 1)[1])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), receipt["output_sha256"])
            body = json.dumps(prepared["body"])
            for forbidden in ("source_path", ".jpg", ".png", "lora_seed", "candidate_status"):
                self.assertNotIn(forbidden, body)
            self.assertTrue(prepared["body"]["text"]["format"]["strict"])
            self.assertEqual(prepared["body"]["max_output_tokens"], 8192)
        self.assertEqual(total, 720)

    def test_dry_run_needs_no_key_and_makes_no_request(self):
        with patch.dict(os.environ, {"VLM_API_KEY": ""}), patch.object(client, "build_opener") as opener:
            folder, result = self.run_request("https://unused.example/v1", send=False)
        opener.assert_not_called()
        self.assertFalse(result["request_started"])
        self.assertFalse((folder / "raw_response.bin").exists())
        self.assertEqual(json.loads((folder / "request.json").read_bytes()), self.prepared["body"])

    def test_http_round_trip_evidence_usage_and_aggregation(self):
        server, url = self.serve()
        folder, result = self.run_request(url)
        request = server.requests.get_nowait()
        self.assertEqual(request["path"], "/v1/responses")
        self.assertEqual(request["authorization"], "Bearer " + SECRET)
        self.assertEqual(request["body"], self.prepared["body"])
        self.assertFalse(request["body"]["store"])
        self.assertEqual(result["engineering_status"], "valid_result")
        self.assertEqual(result["candidate_status"], "candidate_pass")
        self.assertEqual(result["returned_model"], "returned-synthetic-model")
        self.assertEqual(result["usage"]["total_tokens"], 300)
        self.assertEqual((folder / "raw_response.bin").read_bytes(), server.raw)
        for name in ("request.json", "request_meta.json", "provider.json", "answer.txt",
                     "parsed.json", "validation.json", "run.json"):
            self.assertTrue((folder / name).is_file(), name)

    def test_json_duplicate_keys_constants_overflow_unicode_and_code_fences_rejected(self):
        bad = ['{"sample_id":"a","sample_id":"b","labels":[]}', '{"n":NaN}', '{"n":Infinity}',
               '{"n":1e999}', '{"s":"\\ud800"}', '```json\n{}\n```', '{} trailing']
        for text in bad:
            with self.subTest(text=text):
                *_, report = client.validate_response(json.dumps(response(self.prepared, text)), self.prepared)
                self.assertEqual(report["engineering_status"], "invalid_result")
                self.assertEqual(report["candidate_status"], "needs_review")
        duplicate_outer = '{"status":"completed","status":"completed"}'
        *_, report = client.validate_response(duplicate_outer, self.prepared)
        self.assertEqual(report["validation_errors"][0]["code"], "duplicate_json_key")

    def test_missing_duplicate_unknown_extra_wrong_id_enum_and_evidence_rejected(self):
        for kind in ("missing", "duplicate", "unknown", "extra", "wrong_id", "bad_status",
                     "not_applicable", "empty", "invalid_type", "invalid_envelope"):
            answer = valid_answer(self.prepared)
            if kind == "missing":
                answer["labels"].pop()
            elif kind == "duplicate":
                answer["labels"].append(copy.deepcopy(answer["labels"][0]))
            elif kind == "unknown":
                answer["labels"][0]["criterion_id"] = "wooden_table"
            elif kind == "extra":
                answer["labels"][0]["required"] = False
            elif kind == "wrong_id":
                answer["sample_id"] = "sample_029"
            elif kind in ("bad_status", "not_applicable"):
                answer["labels"][0]["status"] = "invalid" if kind == "bad_status" else kind
            elif kind == "empty":
                answer["labels"][0]["reason"] = " "
            elif kind == "invalid_type":
                answer["labels"][0]["candidate_region"] = []
            else:
                answer["candidate_status"] = "candidate_pass"
            with self.subTest(kind=kind):
                *_, report = client.validate_response(
                    json.dumps(response(self.prepared, json.dumps(answer))), self.prepared)
                self.assertEqual(report["engineering_status"], "invalid_result")
                self.assertTrue(report["validation_errors"])

    def test_refusal_incomplete_error_missing_status_ambiguous_and_invalid_blocks(self):
        mutations = []
        for status in ("incomplete", "failed", None):
            payload = response(self.prepared)
            payload["status"] = status
            mutations.append(payload)
        payload = response(self.prepared)
        payload["error"] = {"code": "synthetic"}
        mutations.append(payload)
        for content in ([{"type": "refusal", "refusal": "synthetic refusal"}], [],
                        [{"type": "output_text", "text": "{}"}] * 2,
                        [{"type": "unexpected", "text": "{}"}]):
            payload = response(self.prepared)
            payload["output"][1]["content"] = content
            mutations.append(payload)
        payload = response(self.prepared)
        payload["output"][1]["status"] = "incomplete"
        mutations.append(payload)
        for payload in mutations:
            *_, report = client.validate_response(json.dumps(payload), self.prepared)
            self.assertEqual(report["engineering_status"], "task_not_succeeded")
            self.assertEqual(report["candidate_status"], "needs_review")

    def test_required_fail_unknown_and_auxiliary_aggregation_uses_frozen_rules(self):
        for kind, expected in (("required_fail", "candidate_fail"),
                               ("required_unknown", "needs_review"),
                               ("auxiliary_fail", "candidate_pass")):
            answer = valid_answer(self.prepared)
            required = kind != "auxiliary_fail"
            key = next(key for key, row in self.prepared["expected"].items()
                       if row["required"] == required)
            row = next(row for row in answer["labels"] if row["criterion_id"] == key)
            row["status"] = "unjudgeable" if kind == "required_unknown" else "fail"
            *_, report = client.validate_response(
                json.dumps(response(self.prepared, json.dumps(answer))), self.prepared)
            self.assertEqual(report["candidate_status"], expected)
            self.assertEqual(report["engineering_status"], "valid_result")

    def test_invalid_result_saved_instead_of_silently_repaired(self):
        answer = valid_answer(self.prepared)
        answer["labels"].pop()
        server, url = self.serve(response(self.prepared, json.dumps(answer)))
        folder, result = self.run_request(url)
        self.assertEqual(result["engineering_status"], "invalid_result")
        self.assertEqual(json.loads((folder / "parsed.json").read_bytes()), answer)
        self.assertEqual((folder / "raw_response.bin").read_bytes(), server.raw)
        self.assertIn("missing_criterion_id", (folder / "validation.json").read_text())

    def test_http_error_retains_body_and_no_visual_fail(self):
        server, url = self.serve(raw=b'{"error":{"message":"synthetic quota limit"}}', code=429)
        folder, result = self.run_request(url)
        self.assertEqual(result["error_code"], "http_429")
        self.assertEqual(result["candidate_status"], "needs_review")
        self.assertEqual((folder / "raw_response.bin").read_bytes(), server.raw)
        self.assertEqual(json.loads((folder / "provider.json").read_bytes()),
                         {"error": {"message": "synthetic quota limit"}})
        self.assertEqual(server.requests.qsize(), 1)

    def test_redirect_not_followed(self):
        server, url = self.serve(raw=b"redirect", code=302)
        _, result = self.run_request(url)
        self.assertEqual(result["error_code"], "http_302")
        self.assertEqual(server.requests.qsize(), 1)

    def test_timeout_preserves_failure_receipt_and_does_not_retry(self):
        with patch.object(client, "build_opener") as opener:
            opener.return_value.open.side_effect = URLError("synthetic timeout")
            folder, result = self.run_request("https://unused.example/v1")
        self.assertEqual(opener.return_value.open.call_count, 1)
        self.assertEqual(result["error_code"], "transport_error")
        self.assertIsNone(json.loads((folder / "parsed.json").read_bytes()))
        self.assertFalse((folder / "raw_response.bin").exists())

    def test_invalid_or_missing_credentials_send_nothing(self):
        for key in ("", "bad key"):
            with patch.dict(os.environ, {"VLM_API_KEY": key}), patch.object(client, "build_opener") as opener:
                with self.assertRaises(ValueError):
                    self.run_request("https://unused.example/v1")
            opener.assert_not_called()

    def test_raw_and_parsed_credentials_are_redacted_even_json_escaped_echo(self):
        answer = valid_answer(self.prepared)
        answer["labels"][0]["reason"] = SECRET + " Bearer other-sensitive-token"
        payload = response(self.prepared, json.dumps(answer))
        payload["diagnostic"] = SECRET
        raw = json.dumps(payload).encode()
        server, url = self.serve(raw=raw)
        folder, result = self.run_request(url)
        self.assertTrue(result["response_redacted"])
        for file in folder.iterdir():
            text = file.read_text(encoding="utf-8")
            self.assertNotIn(SECRET, text)
            self.assertNotIn("other-sensitive-token", text)
        escaped = ('{"diagnostic":"' + "".join(f"\\u{ord(c):04x}" for c in SECRET) + '"}').encode()
        clean = client.scrub_raw(escaped, SECRET)
        self.assertEqual(json.loads(clean)["diagnostic"], "[REDACTED]")

    def test_response_size_limit_retains_only_prefix_and_flags_failure(self):
        _, url = self.serve(raw=b"x" * 100)
        with patch.object(client, "MAX_RESPONSE_BYTES", 64):
            folder, result = self.run_request(url)
        self.assertEqual(result["error_code"], "response_too_large")
        self.assertTrue(result["response_truncated"])
        self.assertEqual((folder / "raw_response.bin").stat().st_size, 64)
        self.assertIsNone(result["raw_response_sha256"])

    def test_short_http_body_prefix_is_kept_and_never_accepted(self):
        server, url = self.serve(raw=b'{"status":', declared_length=200)
        folder, result = self.run_request(url)
        self.assertEqual(result["error_code"], "incomplete_http_body")
        self.assertEqual((folder / "raw_response.bin").read_bytes(), server.raw)

    def test_initial_snapshot_gate_read_only_and_holdout_blocked(self):
        db_path = self.folder / "labels.sqlite3"
        with self.assertRaises(ValueError):
            client.require_initial_snapshot("sample_020", db_path)
        self.assertFalse(db_path.exists())
        store = AnnotationStore(db_path)
        with self.assertRaises(ValueError):
            client.require_initial_snapshot("sample_020", db_path)
        for label in valid_answer(self.prepared)["labels"]:
            store.save("sample_020", label, "synthetic annotator", 0)
        snapshot_sha = store.confirm_initial("sample_020", "synthetic annotator")
        before = db_path.read_bytes()
        self.assertEqual(client.require_initial_snapshot("sample_020", db_path), snapshot_sha)
        self.assertEqual(db_path.read_bytes(), before)
        with self.assertRaises(ValueError):
            client.require_initial_snapshot("sample_001", db_path)
        with closing(sqlite3.connect(db_path)) as db:
            with db:
                db.execute("UPDATE initial_snapshots SET sha256='invalid'")
        with self.assertRaises(ValueError):
            client.require_initial_snapshot("sample_020", db_path)

    def test_run_directories_unique_and_no_previous_evidence_overwritten(self):
        first, _ = self.run_request("https://unused.example/v1", send=False)
        before = (first / "run.json").read_bytes()
        second, _ = self.run_request("https://unused.example/v1", send=False)
        self.assertNotEqual(first, second)
        self.assertEqual((first / "run.json").read_bytes(), before)

    def test_invalid_configuration_sample_and_output_limit_fail_before_network(self):
        for sid, limit in (("sample_unknown", 8192), ("sample_020", 0),
                           ("sample_020", True)):
            with self.assertRaises(ValueError):
                client.build_request(sid, "test-model", limit)
        with self.assertRaises(ValueError):
            self.run_request("https://user:password@unused.example/v1")
        with self.assertRaises(ValueError):
            self.run_request("https://unused.example/v1", timeout=float("nan"))

    def test_settings_reads_only_endpoint_model_and_environment_overrides(self):
        config = self.folder / "config.local.json"
        config.write_text(json.dumps({"base_url": "https://unused.example/custom",
                                      "model": "configured-model", "api_key": "ignored-secret"}))
        with patch.object(client, "CONFIG", config), patch.dict(
                os.environ, {"VLM_BASE_URL": "", "VLM_MODEL": "", "VLM_API_KEY": ""}):
            self.assertEqual(client.settings_from_environment(), {
                "base_url": "https://unused.example/custom", "model": "configured-model"})
            with patch.dict(os.environ, {"VLM_MODEL": "override-model"}):
                self.assertEqual(client.settings_from_environment()["model"], "override-model")
            config.write_text('{"base_url": [], "model": "test"}')
            with self.assertRaises(ValueError):
                client.settings_from_environment()

    def test_cli_sending_requires_initial_snapshot_before_building_or_network(self):
        args = ["judge_client.py", "--sample-id", "sample_020", "--send"]
        with patch.object(sys, "argv", args), patch.object(
                client, "settings_from_environment", return_value={
                    "base_url": "https://unused.example/v1", "model": "test"}), patch.object(
                client, "require_initial_snapshot", side_effect=ValueError("not confirmed")), patch.object(
                client, "build_request") as builder, patch.object(client, "build_opener") as opener:
            with self.assertRaises(ValueError):
                client.main()
        builder.assert_not_called()
        opener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
