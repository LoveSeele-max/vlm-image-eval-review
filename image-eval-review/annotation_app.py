"""D01 localhost-only anonymous annotation page. No model API integration."""

import argparse
import json
import secrets
import sqlite3
import sys
import threading
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from annotation_store import (
    AnnotationError, AnnotationStore, ConflictError, DEFAULT_DB, DEFAULT_MARKDOWN,
)
from build_manifest import DATA, read_jsonl
from prepare_image import prepare_image


WEB = Path(__file__).resolve().parent / "web"


class AnnotationService:
    def __init__(self, database=DEFAULT_DB, auto_import=True):
        self.store = AnnotationStore(database)
        self.token = secrets.token_urlsafe(32)
        self.image_policy = json.loads(
            (DATA / "image_input_policy.json").read_text(encoding="utf-8"))
        self.sources = {row["sample_id"]: row for row in read_jsonl(DATA / "sample_sources.jsonl")}
        self.references = {row["reference_id"]: row for row in json.loads(
            (DATA / "reference_assets.json").read_text(encoding="utf-8"))}
        self.import_receipt = None
        if auto_import and DEFAULT_MARKDOWN.exists():
            with self.store.connect() as db:
                has_labels = bool(db.execute("SELECT 1 FROM labels LIMIT 1").fetchone())
            if not has_labels:
                supplement = DEFAULT_MARKDOWN.parent / "annotation_supplement_20261107.json"
                self.import_receipt = self.store.import_markdown(
                    DEFAULT_MARKDOWN, supplement_path=supplement if supplement.exists() else None)

    @lru_cache(maxsize=44)
    def image(self, identifier):
        if identifier in self.references:
            row = self.references[identifier]
            return prepare_image(
                Path(row["source_path"]), identifier, self.image_policy, row["sha256"])[0]
        if identifier in self.store.manifest:
            source = self.sources[identifier]
            row = self.store.manifest[identifier]
            return prepare_image(
                Path(source["source_path"]), "CANDIDATE", self.image_policy,
                row["candidate_sha256"])[0]
        raise AnnotationError("图片编号未登记。")


def make_server(service, port=8501):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Avoid request contents/source paths in terminal output.
            pass

        def safe_origin(self):
            valid = {f"127.0.0.1:{self.server.server_port}",
                     f"localhost:{self.server.server_port}"}
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            return host in valid and (origin is None or origin in {f"http://{h}" for h in valid})

        def send(self, payload, content_type="application/json; charset=utf-8", status=200):
            if isinstance(payload, (dict, list)):
                payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", (
                "default-src 'self'; img-src 'self' data:; style-src 'self'; "
                "script-src 'self'; frame-ancestors 'none'; base-uri 'none'"))
            self.end_headers()
            self.wfile.write(payload)

        def error(self, text, status=400):
            self.send({"error": text}, status=status)

        def do_GET(self):
            if not self.safe_origin():
                return self.error("仅允许本机页面访问。", 403)
            parsed = urlsplit(self.path)
            try:
                if parsed.path == "/":
                    page = (WEB / "annotation.html").read_bytes().replace(
                        b"__APP_TOKEN__", service.token.encode("ascii"))
                    return self.send(page, "text/html; charset=utf-8")
                static = {"/app.js": ("app.js", "text/javascript; charset=utf-8"),
                          "/app.css": ("app.css", "text/css; charset=utf-8")}
                if parsed.path in static:
                    name, kind = static[parsed.path]
                    return self.send((WEB / name).read_bytes(), kind)
                if parsed.path == "/api/health":
                    return self.send({"app": "vlm-image-annotation", "version": 1})
                if parsed.path == "/api/overview":
                    return self.send(service.store.overview())
                if parsed.path in ("/api/sample", "/api/export"):
                    sid = parse_qs(parsed.query).get("sample_id", [""])[0]
                    result = (service.store.sample(sid) if parsed.path == "/api/sample"
                              else service.store.export(sid))
                    return self.send(result)
                if parsed.path.startswith("/api/image/"):
                    key = parsed.path.removeprefix("/api/image/")
                    return self.send(service.image(key), "image/png")
                return self.error("页面不存在。", 404)
            except AnnotationError as exc:
                return self.error(str(exc))
            except (OSError, ValueError, KeyError, sqlite3.Error):
                return self.error("本地资料读取失败，请检查文件与规则版本。", 500)

        def do_POST(self):
            if (not self.safe_origin()
                    or self.headers.get("X-Annotation-Token") != service.token):
                return self.error("页面会话已失效，请刷新后重试。", 403)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 100000:
                    return self.error("请求大小非法。", 413)
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    return self.error("请求格式应为 JSON。", 415)

                def unique_pairs(pairs):
                    result = {}
                    for key, value in pairs:
                        if key in result:
                            raise AnnotationError("JSON 存在重复字段。")
                        result[key] = value
                    return result

                body = json.loads(self.rfile.read(size), object_pairs_hook=unique_pairs)
                if not isinstance(body, dict):
                    return self.error("请求格式应为对象。")
                route = urlsplit(self.path).path
                if route == "/api/save":
                    if set(body) != {"sample_id", "label", "annotator",
                                     "expected_revision", "change_reason"}:
                        return self.error("保存字段不完整。")
                    revision = service.store.save(
                        body["sample_id"], body["label"], body["annotator"],
                        body["expected_revision"], body["change_reason"])
                    return self.send({"revision": revision})
                if route == "/api/confirm":
                    if set(body) != {"sample_id", "annotator"}:
                        return self.error("确认字段不完整。")
                    digest = service.store.confirm_initial(body["sample_id"], body["annotator"])
                    return self.send({"sha256": digest})
                return self.error("接口不存在。", 404)
            except ConflictError as exc:
                return self.error(str(exc), 409)
            except AnnotationError as exc:
                return self.error(str(exc))
            except (ValueError, TypeError, KeyError):
                return self.error("请求包含非法数据。")
            except (OSError, sqlite3.Error):
                return self.error("本地保存失败，请保留当前填写后重试。", 500)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--no-auto-import", action="store_true")
    parser.add_argument("--open", action="store_true", help="启动后打开本机浏览器")
    args = parser.parse_args()
    service = AnnotationService(args.db, auto_import=not args.no_auto_import)
    server = make_server(service, args.port)
    address = f"http://127.0.0.1:{server.server_port}"
    print(f"匿名标注页已启动：{address}", flush=True)
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError):
        print("启动失败：请检查端口占用、图片位置和正式规则完整性。", file=sys.stderr)
        sys.exit(1)
