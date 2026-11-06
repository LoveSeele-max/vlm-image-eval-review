"""A single text probe for a Responses-compatible endpoint."""

import json
import os
import re
import time
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_ERROR_BYTES = 16 * 1024
CLIENT_USER_AGENT = "vlm-image-eval-review/0.1"


class ConnectionTestError(Exception):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the Authorization header to a redirected host.
        return None


def normalize_base_url(value):
    value = value.strip().rstrip("/")
    parts = urlsplit(value)
    if (
        parts.scheme not in ("https", "http")
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise ValueError("请填写完整的 http/https 接口地址，不含密钥、查询参数或账号密码。")
    try:
        parts.port
    except ValueError as exc:
        raise ValueError("接口地址中的端口不正确。") from exc
    path = parts.path.rstrip("/")
    if path.endswith("/responses"):
        path = path[:-len("/responses")]
    if path.endswith("/chat/completions"):
        raise ValueError("当前测试使用 Responses 接口，请填写服务商提供的 Base URL。")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def validate_settings(base_url, model):
    base_url = normalize_base_url(base_url)
    model = model.strip()
    if not model:
        raise ValueError("请填写完整模型名。")
    return {"base_url": base_url, "model": model}


def redact(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, dict):
        return {key: redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, secret) for item in value]
    return value


def integer_or_none(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def explain_http_error(error, api_key):
    code = error.code
    try:
        raw = error.read(MAX_ERROR_BYTES)
    except (OSError, ValueError):
        raw = b""
    finally:
        error.close()
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    if code == 403 and (
        payload.get("error_code") in (1010, "1010")
        or payload.get("error_name") == "browser_signature_banned"
    ):
        return (
            "HTTP 403：Cloudflare 1010 拦截了客户端标识，"
            "尚不能据此判断密钥或模型权限。请确认使用最新版工具；"
            "若仍被拦截，请联系中转服务商检查网关规则。"
        )
    messages = {
        400: "请求参数未被接受，请核对模型名和接口协议。",
        401: "密钥校验未通过，请检查密钥和服务商配置。",
        403: "请求被拒绝，可能是网关拦截或权限限制。",
        404: "未找到接口或模型，请检查地址和服务商配置。",
        429: "请求受到限制，请检查调用频率和账户额度。",
    }
    message = f"HTTP {code}：" + messages.get(code, "服务商拒绝或未能完成请求。")
    provider_error = payload.get("error")
    detail = provider_error.get("message") if isinstance(provider_error, dict) else None
    if not isinstance(detail, str):
        detail = payload.get("detail")
    if isinstance(detail, str):
        # Only extract a bounded, redacted diagnostic field; never log the raw body.
        detail = redact(detail, api_key)
        detail = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", detail)
        detail = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", detail)
        detail = " ".join(detail.split())[:500]
        if detail:
            message += "\n服务端提示：" + detail
    return message


def test_connection(base_url, model, timeout=120):
    settings = validate_settings(base_url, model)
    api_key = os.environ.get("VLM_API_KEY", "").strip()
    if not api_key:
        raise ValueError("请填写 API Key。")
    if any(char.isspace() for char in api_key):
        raise ValueError("API Key 中包含空格或换行，请检查粘贴内容。")
    body = {
        "model": settings["model"],
        "input": "只回复：连接成功",
        "max_output_tokens": 1024,
        "stream": False,
    }
    request = Request(
        settings["base_url"] + "/responses",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": CLIENT_USER_AGENT,
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with build_opener(NoRedirect()).open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        raise ConnectionTestError(explain_http_error(exc, api_key)) from None
    except (URLError, TimeoutError, OSError, ValueError, UnicodeError):
        raise ConnectionTestError("连接失败或超时，请检查接口地址、网络和代理设置。") from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ConnectionTestError("接口响应过大，未保存原始内容。")
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeError):
        raise ConnectionTestError("接口未返回有效 JSON，请检查是否填成了网站首页地址。") from None
    if not isinstance(payload, dict):
        raise ConnectionTestError("接口返回格式不符合 Responses 协议。")
    if payload.get("error"):
        raise ConnectionTestError("接口返回了错误，请查看服务商控制台。")
    if payload.get("status") not in (None, "completed"):
        raise ConnectionTestError("请求尚未完整完成；请检查服务商限制或输出上限。")
    output = payload.get("output", [])
    if not isinstance(output, list):
        raise ConnectionTestError("接口的 output 字段格式不正确。")
    texts = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "output_text"
                and isinstance(block.get("text"), str)
            ):
                texts.append(block["text"])
    text = "\n".join(texts).strip()
    if not text:
        raise ConnectionTestError("接口没有返回文字结果，请检查中转是否完整支持 Responses。")
    usage = payload.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("input_tokens_details")
    details = details if isinstance(details, dict) else {}
    # Keep a small receipt, never the raw payload or request headers.
    receipt = {
        "test_type": "text_connection_only",
        "success": True,
        "tested_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "requested_model": settings["model"],
        "returned_model": payload.get("model") if isinstance(payload.get("model"), str) else None,
        "provider_status": payload.get("status"),
        "text": text[:2000],
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "usage": {
            "input_tokens": integer_or_none(usage.get("input_tokens")),
            "cached_tokens": integer_or_none(details.get("cached_tokens")),
            "output_tokens": integer_or_none(usage.get("output_tokens")),
            "total_tokens": integer_or_none(usage.get("total_tokens")),
        },
    }
    return redact(receipt, api_key)
