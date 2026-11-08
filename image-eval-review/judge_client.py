"""D02: one Responses request, strict parsing and private evidence files.

Default CLI mode builds a request without sending it. The --send entry point
uses D03's durable scheduler; holdout execution remains locked.
"""

import argparse
import base64
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
import uuid
from contextlib import closing
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import HTTPException, IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from api_connection import CLIENT_USER_AGENT, NoRedirect, validate_settings
from build_manifest import DATA, ROOT, read_jsonl, text_digest
from evaluate_results import evaluate_result, expected_for_sample
from freeze_rules import load_frozen_context, render_prompt
from prepare_image import prepare_image


PRIVATE = ROOT / "image-eval-review/private"
CONFIG = ROOT / "image-eval-review/config.local.json"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
DEFAULT_OUTPUT_TOKENS = 8192


class ResultError(ValueError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def strict_json(value):
    """Reject duplicate keys, nonfinite numbers and invalid Unicode."""
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ResultError("duplicate_json_key")
            result[key] = item
        return result

    def number(text):
        result = float(text)
        if not math.isfinite(result):
            raise ResultError("nonfinite_json_number")
        return result

    def constant(_):
        raise ResultError("nonfinite_json_number")

    try:
        parsed = json.loads(value, object_pairs_hook=pairs, parse_float=number,
                            parse_constant=constant)
        # A JSON escape must not smuggle an unpaired surrogate into UTF-8 files.
        json.dumps(parsed, ensure_ascii=False).encode("utf-8")
        return parsed
    except ResultError:
        raise
    except (ValueError, UnicodeError, RecursionError, OverflowError):
        raise ResultError("invalid_json") from None


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def scrub_text(text, secret):
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return re.sub(r"(?i)\bBearer\s+[^\s\"\\]+", "Bearer [REDACTED]", text)


def scrub_raw(raw, secret):
    """Keep response bytes unchanged unless credentials must be removed.

    Decode each JSON string token for escaped key echoes, without collapsing
    duplicate keys or otherwise repairing the response.
    """
    text = raw.decode("utf-8", errors="replace")

    def replace(match):
        try:
            value = json.loads(match[0])
            clean = scrub_text(value, secret)
            return json.dumps(clean, ensure_ascii=True) if clean != value else match[0]
        except (ValueError, UnicodeError):
            return match[0]

    cleaned = re.sub(r'"(?:[^"\\]|\\.)*"', replace, text)
    cleaned = scrub_text(cleaned, secret)
    return cleaned.encode("utf-8") if cleaned != text else raw


def write_json(path, value, secret=""):
    # Serialize first so even credential echoes in nested diagnostics are removed.
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(scrub_raw(encoded(value) + b"\n", secret))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def retry_after_seconds(headers):
    value = headers.get("Retry-After", "")
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return seconds if math.isfinite(seconds) and 0 <= seconds <= 86400 else None


def read_response(response):
    """Retain a bounded prefix even if an HTTP body terminates unexpectedly."""
    chunks, size = [], 0
    error = None
    try:
        while size <= MAX_RESPONSE_BYTES:
            block = response.read(min(65536, MAX_RESPONSE_BYTES + 1 - size))
            if not block:
                break
            chunks.append(block)
            size += len(block)
    except IncompleteRead as exc:
        chunks.append(exc.partial)
        error = "incomplete_http_body"
    except (TimeoutError, URLError, OSError, HTTPException):
        error = "transport_error"
    raw = b"".join(chunks)
    length = response.headers.get("Content-Length", "")
    if length.isdigit() and len(raw) < int(length) and len(raw) <= MAX_RESPONSE_BYTES:
        error = error or "incomplete_http_body"
    return raw, error


def settings_from_environment():
    config = {}
    if CONFIG.exists():
        value = strict_json(CONFIG.read_bytes())
        if not isinstance(value, dict):
            raise ValueError("本机接口配置须为对象。")
        config = {key: value.get(key, "") for key in ("base_url", "model")}
    base_url = os.environ.get("VLM_BASE_URL") or config.get("base_url", "")
    model = os.environ.get("VLM_MODEL") or config.get("model", "")
    if not isinstance(base_url, str) or not isinstance(model, str):
        raise ValueError("本机接口地址与模型名须为文字。")
    return validate_settings(base_url, model)


def build_request(sample_id, model, max_output_tokens=DEFAULT_OUTPUT_TOKENS):
    if not isinstance(model, str) or not model.strip():
        raise ValueError("模型名不能为空。")
    if type(max_output_tokens) is not int or not 1024 <= max_output_tokens <= 32768:
        raise ValueError("输出上限应为 1024–32768 的整数。")
    policy, reference, evaluation, samples = load_frozen_context()
    expected = expected_for_sample(sample_id, reference, evaluation, samples)
    manifest = {row["sample_id"]: row for row in read_jsonl(DATA / "manifest.jsonl")}
    sources = {row["sample_id"]: row for row in read_jsonl(DATA / "sample_sources.jsonl")}
    assets = {row["reference_id"]: row for row in json.loads(
        (DATA / "reference_assets.json").read_text(encoding="utf-8"))}
    image_policy = json.loads((DATA / "image_input_policy.json").read_text(encoding="utf-8"))
    prompt = render_prompt(sample_id)
    schema = json.loads((DATA / "response_schema.json").read_text(encoding="utf-8"))
    content = [{"type": "input_text", "text": prompt}]
    receipts = []
    for role in image_policy["image_order"]:
        row = assets[role] if role != "CANDIDATE" else sources[sample_id]
        source_hash = row["sha256"] if role != "CANDIDATE" else manifest[sample_id]["candidate_sha256"]
        png, receipt = prepare_image(Path(row["source_path"]), role, image_policy, source_hash)
        content.append({"type": "input_image", "image_url": "data:image/png;base64," +
                        base64.b64encode(png).decode("ascii"),
                        "detail": image_policy["image_detail_requested"]})
        receipts.append(receipt)
    body = {
        "model": model.strip(), "stream": False, "store": False,
        "max_output_tokens": max_output_tokens,
        "input": [{"role": "user", "content": content}],
        "text": {"format": {"type": "json_schema", "name": "image_evaluation",
                            "strict": True, "schema": schema}},
    }
    return {
        "body": body, "expected": expected, "policy": policy,
        "metadata": {
            "sample_id": sample_id, "split": manifest[sample_id]["split"],
            "rule_version": policy["rule_version"], "prompt_version": "judge_v1",
            "ruleset_sha256": text_digest(DATA / "ruleset.json"),
            "manifest_sha256": text_digest(DATA / "manifest.jsonl"),
            "prompt_sha256": digest(prompt.encode("utf-8")),
            "schema_sha256": text_digest(DATA / "response_schema.json"),
            "image_policy_sha256": text_digest(DATA / "image_input_policy.json"),
            "client_source_sha256": text_digest(Path(__file__)),
            "images": receipts, "request_body_sha256": digest(encoded(body)),
        },
    }


def validate_response(raw, prepared):
    """The original envelope/answer survive even when semantic validation fails."""
    sid, expected, policy = (prepared["metadata"]["sample_id"],
                             prepared["expected"], prepared["policy"])
    provider = answer = parsed = None
    try:
        provider = strict_json(raw)
        if not isinstance(provider, dict):
            raise ResultError("invalid_response_envelope")
        if provider.get("error"):
            raise ResultError("provider_error")
        if provider.get("status") != "completed":
            raise ResultError("response_not_completed")
        output = provider.get("output")
        if not isinstance(output, list):
            raise ResultError("invalid_output")
        messages = []
        for item in output:
            if not isinstance(item, dict):
                raise ResultError("invalid_output_item")
            if item.get("type") == "reasoning":
                continue
            if item.get("type") != "message" or item.get("role") != "assistant":
                raise ResultError("unexpected_output_item")
            if item.get("status") != "completed":
                raise ResultError("message_not_completed")
            blocks = item.get("content")
            if not isinstance(blocks, list):
                raise ResultError("invalid_content")
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "refusal":
                    raise ResultError("refusal")
                if (not isinstance(block, dict) or block.get("type") != "output_text"
                        or not isinstance(block.get("text"), str)):
                    raise ResultError("invalid_output_text")
                messages.append(block["text"])
        if len(messages) != 1 or not messages[0].strip():
            raise ResultError("ambiguous_or_missing_answer")
        answer = messages[0]
        parsed = strict_json(answer)
        report = evaluate_result(sid, parsed, expected, policy)
        return provider, answer, parsed, report
    except ResultError as exc:
        report = evaluate_result(sid, None, expected, policy, task_status="failed")
        report["validation_errors"] = [{"code": exc.code}]
        # A completed envelope with invalid answer JSON is a result format error.
        if answer is not None:
            report["engineering_status"] = "invalid_result"
        return provider, answer, parsed, report


def require_initial_snapshot(sample_id, database=PRIVATE / "annotations.sqlite3"):
    """Read-only CLI gate; no label values are passed to the model."""
    policy, reference, evaluation, samples = load_frozen_context()
    manifest = {row["sample_id"]: row for row in read_jsonl(DATA / "manifest.jsonl")}
    if sample_id not in manifest or manifest[sample_id]["split"] != "development":
        raise ValueError("D02 发送入口只开放开发集，保留集运行留待后续阶段。")
    try:
        with closing(sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)) as db:
            row = db.execute(
                "SELECT payload,sha256 FROM initial_snapshots WHERE sample_id=?",
                (sample_id,)).fetchone()
    except sqlite3.Error:
        raise ValueError("未找到可用的人工初始快照，请先在标注页确认。") from None
    if not row:
        raise ValueError("请先由作者在标注页确认本图初始记录，再调用模型。")
    payload = strict_json(row[0])
    expected = expected_for_sample(sample_id, reference, evaluation, samples)
    if (not isinstance(payload, dict) or digest(row[0].encode("utf-8")) != row[1]
            or payload.get("sample_id") != sample_id
            or payload.get("rule_version") != policy["rule_version"]
            or payload.get("ruleset_sha256") != text_digest(DATA / "ruleset.json")
            or payload.get("candidate_sha256") != manifest[sample_id]["candidate_sha256"]):
        raise ValueError("人工初始快照与当前规则或图片不匹配。")
    records = payload.get("records", {})
    if not isinstance(records, dict) or any(
            not isinstance(record, dict) or not isinstance(record.get("label"), dict)
            or record["label"].get("criterion_id") != key for key, record in records.items()):
        raise ValueError("人工初始快照的条目记录格式非法。")
    result = {"sample_id": sample_id, "labels": [record["label"] for record in records.values()]}
    if evaluate_result(sample_id, result, expected, policy)["engineering_status"] != "valid_result":
        raise ValueError("人工初始快照不完整或不合法。")
    return row[1]


def execute_request(prepared, base_url, *, send=False, output_root=PRIVATE / "judge_runs",
                    timeout=180, run_id=None, attempt_number=1):
    """Single attempt only. D03 can compose this with its own task scheduler."""
    settings = validate_settings(base_url, prepared["body"]["model"])
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 600:
        raise ValueError("超时须大于 0 且不超过 600 秒。")
    secret = os.environ.get("VLM_API_KEY", "").strip() if send else ""
    if send and (not secret or any(char.isspace() for char in secret)):
        raise ValueError("VLM_API_KEY 未设置或含空白字符，未发送请求。")
    if run_id is not None and (not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", run_id)):
        raise ValueError("尝试编号非法。")
    if type(attempt_number) is not int or not 1 <= attempt_number <= 1000:
        raise ValueError("尝试次数非法。")
    # An immutable directory per attempt; no overwriting another run's evidence.
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
    folder = Path(output_root) / run_id
    folder.mkdir(parents=True, exist_ok=False)
    body = encoded(prepared["body"])
    write_json(folder / "request.json", prepared["body"], secret)
    write_json(folder / "request_meta.json", prepared["metadata"], secret)
    receipt = {
        "run_id": run_id, **prepared["metadata"],
        "endpoint": settings["base_url"] + "/responses", "timeout_seconds": timeout,
        "requested_model": settings["model"], "max_output_tokens": prepared["body"]["max_output_tokens"],
        "stream": False, "store": False, "attempt": attempt_number,
        "mode": "single_request" if send else "dry_run",
        "request_started": False, "http_status": None, "error_code": None,
        "provider_status": None, "returned_model": None, "usage": None,
        "elapsed_seconds": 0, "completed_at": None,
        "retry_after_seconds": None,
        "schema_enforcement": "not_proven_by_local_validation",
    }
    write_json(folder / "run.json", receipt, secret)
    if not send:
        return folder, receipt
    request = Request(receipt["endpoint"], data=body, method="POST", headers={
        "Authorization": "Bearer " + secret, "Content-Type": "application/json",
        "Accept": "application/json", "User-Agent": CLIENT_USER_AGENT,
    })
    raw = None
    provider = answer = parsed = None
    started = time.monotonic()
    receipt["request_started"] = True
    # If killed during I/O, D03 can later distinguish an unfinished attempt.
    write_json(folder / "run.json", receipt, secret)
    try:
        try:
            with build_opener(NoRedirect()).open(request, timeout=timeout) as response:
                receipt["http_status"] = response.status
                receipt["retry_after_seconds"] = retry_after_seconds(response.headers)
                raw, receipt["error_code"] = read_response(response)
        except HTTPError as exc:
            receipt["http_status"] = exc.code
            receipt["retry_after_seconds"] = retry_after_seconds(exc.headers)
            try:
                raw, _ = read_response(exc)
            finally:
                exc.close()
            receipt["error_code"] = f"http_{receipt['http_status']}"
    except (TimeoutError, URLError, OSError, HTTPException):
        receipt["error_code"] = "transport_error"
    if raw is not None:
        too_large = len(raw) > MAX_RESPONSE_BYTES
        kept = scrub_raw(raw[:MAX_RESPONSE_BYTES], secret)
        (folder / "raw_response.bin").write_bytes(kept)
        receipt.update({
            "raw_response_sha256": digest(raw) if not too_large else None,
            "received_prefix_sha256": digest(raw),
            "saved_response_sha256": digest(kept),
            "response_redacted": kept != raw[:MAX_RESPONSE_BYTES],
            "response_truncated": too_large, "saved_response_bytes": len(kept),
        })
        if too_large:
            receipt["error_code"] = receipt["error_code"] or "response_too_large"
    if not receipt["error_code"]:
        provider, answer, parsed, report = validate_response(raw, prepared)
    if receipt["error_code"]:
        if raw is not None and len(raw) <= MAX_RESPONSE_BYTES:
            try:
                provider = strict_json(raw)
            except ResultError:
                pass
        report = evaluate_result(
            prepared["metadata"]["sample_id"], None, prepared["expected"],
            prepared["policy"], task_status="failed")
        report["validation_errors"] = [{"code": receipt["error_code"]}]
    if isinstance(provider, dict):
        receipt["provider_status"] = provider.get("status")
        receipt["returned_model"] = provider.get("model")
        receipt["usage"] = provider.get("usage")
    if report["engineering_status"] != "valid_result" and not receipt["error_code"]:
        receipt["error_code"] = report["validation_errors"][0]["code"]
    receipt["engineering_status"] = report["engineering_status"]
    receipt["candidate_status"] = report["candidate_status"]
    receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
    receipt["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(folder / "provider.json", provider, secret)
    (folder / "answer.txt").write_text(scrub_text(answer or "", secret), encoding="utf-8")
    write_json(folder / "parsed.json", parsed, secret)
    write_json(folder / "validation.json", report, secret)
    write_json(folder / "run.json", receipt, secret)
    return folder, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--send", action="store_true", help="真实发送一次；默认仅生成私有请求文件")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    settings = settings_from_environment()
    initial_sha = require_initial_snapshot(args.sample_id) if args.send else None
    prepared = build_request(args.sample_id, settings["model"], args.max_output_tokens)
    if initial_sha:
        prepared["metadata"]["initial_snapshot_sha256"] = initial_sha
    if args.send:
        from run_tasks import send_one
        result = send_one(prepared, settings["base_url"], args.timeout)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 1 if any(batch["status"] == "stopped" or any(
            task["status"] == "failed" for task in batch["tasks"]) for batch in result["batches"]) else 0
    folder, receipt = execute_request(
        prepared, settings["base_url"], send=args.send, timeout=args.timeout)
    # Never print prompts, model answers, URLs or credentials to the console.
    print(json.dumps({
        "run_id": receipt["run_id"], "sample_id": args.sample_id, "mode": receipt["mode"],
        "request_started": receipt["request_started"],
        "engineering_status": receipt.get("engineering_status"),
        "error_code": receipt["error_code"], "evidence_directory": str(folder),
    }, ensure_ascii=True, indent=2))
    return 1 if receipt["error_code"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("运行已中断，记录已保留。请使用 run_tasks.py 的 recover 或 resume。", file=sys.stderr)
        sys.exit(130)
    except ValueError as exc:
        # Our validators emit fixed messages, never arbitrary provider bodies.
        message = scrub_text(str(exc), os.environ.get("VLM_API_KEY", "").strip())
        print("运行前检查失败：" + message, file=sys.stderr)
        sys.exit(2)
    except (OSError, KeyError, TypeError):
        print("运行前检查失败：请核对配置、密钥、素材、规则及初始快照；未将异常详情写入终端。", file=sys.stderr)
        sys.exit(2)
