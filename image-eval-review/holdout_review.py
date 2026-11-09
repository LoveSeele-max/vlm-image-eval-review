"""G03 gated three-way holdout review, separate author-confirmed adjudications."""
import argparse
import copy
import json
import secrets
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import holdout_metrics as metrics
import judge_client as client
from adjudication_store import AdjudicationStore, DEFAULT_DB, require
from annotation_store import ConflictError, now
from build_manifest import DATA, read_jsonl
from freeze_human_reference import load_registered_reference
from freeze_rules import load_frozen_context
from review_app import ReviewService as ImageService, public_label, WEB


FILTERS = ("binary", "version_change", "ai_uncertain", "human_uncertain", "engineering", "all")
STATES = ("pending", "all", "resolved", "unresolved")


def flags(row):
    h = row["human"]["status"]
    states = [r["status"] if r else None for r in (row["v1"], row["v2"])]
    return {"binary": h in ("pass", "fail") and any(s in ("pass", "fail") and s != h for s in states),
            "version_change": None not in states and states[0] != states[1],
            "ai_uncertain": h in ("pass", "fail") and "unjudgeable" in states,
            "human_uncertain": h == "unjudgeable", "engineering": None in states, "all": True}


def load_review_data():
    latest = client.strict_json((client.PRIVATE/"g02_latest.json").read_bytes())
    folder = Path(latest["output"]).resolve()
    require(folder.is_relative_to((client.PRIVATE/"holdout_metrics").resolve()), "G02 登记路径非法。")
    metrics.verify_export(folder, refresh_source=False)
    items, _, source = metrics.load_sources()
    report = client.strict_json((folder/"metrics.json").read_bytes())
    exported = [client.strict_json(line) for line in (folder/"items.jsonl").read_bytes().splitlines()]
    require(items == exported and report["source"] == source, "G02 与当前冻结来源不同，拒绝解锁。")
    frozen = load_registered_reference()
    policy, reference, evaluation, samples = load_frozen_context()
    spec = {r["sample_id"]: r for r in samples}
    manifest = {r["sample_id"]: r for r in read_jsonl(DATA/"manifest.jsonl")}
    human = {(r["sample_id"], r["record"]["label"]["criterion_id"]): r["record"]
             for r in frozen["initial_rows"] if r["split"] == "holdout"}
    results = client.strict_json((Path(source["g01_directory"])/"results.json").read_bytes())
    ai = {(r["prompt_version"],r["sample_id"]): r for r in results}
    rows, summaries = [], []
    ids = sorted({r["sample_id"] for r in items})
    for sid in ids:
        require(manifest[sid]["split"] == "holdout", "只开放固定保留集。")
        columns = {v: {r["criterion_id"]: r for r in (ai[v,sid]["labels"] or [])} for v in metrics.VERSIONS}
        summaries.append({"sample_id": sid, "prompt": manifest[sid]["prompt"],
            "engineering": {v: {"status": ai[v,sid]["engineering_status"], "error_code": ai[v,sid]["stop_code"]}
                            for v in metrics.VERSIONS}})
        for dimension, rules in (("subject_features",reference["criteria"]),
                ("structural_integrity",evaluation["structure_criteria"]), ("prompt_adherence",spec[sid]["prompt_criteria"])):
            for rule in rules:
                cid = rule["id"]
                author = human[sid,cid]
                origin = author["origin"]
                provenance = ("作者明确指定判断，文字由助手复制或整理" if origin.get("decision_source") == "explicit_user_chat_instruction"
                    else "作者判断，含助手辅助证据文字" if origin.get("kind") == "author_status_codex_evidence" else "作者页面填写")
                row = {"sample_id":sid,"criterion_id":cid,"dimension":dimension,"required":rule["required"],
                    "name":rule.get("name",rule.get("description",cid)),"human":public_label(author["label"]),
                    "human_provenance":provenance,"v1":public_label(columns["judge_v1"][cid]) if cid in columns["judge_v1"] else None,
                    "v2":public_label(columns["judge_v2"][cid]) if cid in columns["judge_v2"] else None,
                    **{k:rule.get(k,"") for k in ("pass_when","fail_when","unjudgeable_when","boundary")}}
                row["flags"] = flags(row)
                rows.append(row)
    require(len(rows)==540 and len(human)==540 and len(ids)==30, "复核范围不是完整冻结保留集。")
    g01 = Path(source["g01_directory"])
    watched = [folder/"metrics.json",folder/"checksums.json",folder/"items.jsonl",
        g01/"completion_receipt.json",g01/"experiment.json",g01/"results.json",g01/"verification.json",
        DATA/"human_reference_registration.json",DATA/"prompt_versions.json",metrics.MetricRuns().path]
    registered = client.strict_json((DATA/"human_reference_registration.json").read_bytes())
    reference_folder = (client.PRIVATE/registered["relative_private_path"]).resolve()
    require(reference_folder.is_relative_to(client.PRIVATE.resolve()),"人工参考路径非法。")
    watched.extend(p for p in reference_folder.rglob("*") if p.is_file())
    watched.extend(DATA/name for name in metrics.runs.pv.FIXED_DATA_NAMES)
    watched.extend(Path(__file__).with_name(name) for name in metrics.runs.pv.ENGINE_NAMES)
    watched.extend((DATA/"sample_sources.jsonl",metrics.runs.pv.PROMPTS/"judge_v1.txt",metrics.runs.pv.PROMPTS/"judge_v2.txt"))
    require(client.digest((g01/"results.json").read_bytes())==source["g01_files_sha256"]["results.json"],
            "构建复核页时 G01 结果发生变化。")
    fingerprints = {str(p):client.digest(p.read_bytes()) for p in watched}
    binding = {"reference":source["reference"],"batches":source["batches"],
               "g01_inputs_sha256":source["g01_files_sha256"],"g02_checksums_sha256":client.digest((folder/"checksums.json").read_bytes())}
    return {"meta":{"reference_version":source["reference"]["reference_version"],"rule_version":policy["rule_version"],
                     "loaded_at":now(),"holdout_unlocked":True,"primary_metrics_unchanged":True},
            "rows":rows,"samples":summaries,"binding":binding,"watched":fingerprints}


class HoldoutReviewService:
    def __init__(self, review_database=DEFAULT_DB, *, dataset=None):
        self.dataset = dataset or load_review_data()
        self.store = AdjudicationStore(self.dataset["binding"],self.dataset["rows"],review_database)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.images = ImageService.__new__(ImageService)
        self.images.sources = {r["sample_id"]:r for r in read_jsonl(DATA/"sample_sources.jsonl")}
        self.images.manifest = {r["sample_id"]:r for r in read_jsonl(DATA/"manifest.jsonl")}
        self.images.references = {r["reference_id"]:r for r in client.strict_json((DATA/"reference_assets.json").read_bytes())}
        self.images.image_policy = client.strict_json((DATA/"image_input_policy.json").read_bytes())
        self.images.allowed = {r["sample_id"] for r in self.dataset["samples"]}

    def guard(self):
        require(all(Path(path).is_file() and client.digest(Path(path).read_bytes()) == sha
                    for path,sha in self.dataset["watched"].items()), "冻结来源或结果已变化，请重新核验后再复核。")

    def review(self, filter_name="binary", state="pending", sample="all", dimension="all", query="", page=0, page_size=5):
        require(filter_name in FILTERS and state in STATES and dimension in ("all",*metrics.DIMENSIONS)
                and isinstance(query,str) and len(query)<=200 and type(page) is int and page>=0
                and type(page_size) is int and page_size in (5,10), "筛选参数非法。")
        require(sample=="all" or sample in self.images.allowed, "仅提供固定保留集样本。")
        with self.lock:
            self.guard()
            current,history = self.store.snapshot()
            rows = []
            for original in self.dataset["rows"]:
                row = copy.deepcopy(original)
                key = (row["sample_id"],row["criterion_id"])
                row["review"] = current.get(key)
                row["review_state"] = ("pending" if key not in current else "unresolved"
                    if current[key]["decision"]=="unresolved" else "resolved")
                row["history"] = [r for r in history if (r["sample_id"],r["criterion_id"])==key]
                rows.append(row)
            summary = {name:sum(bool(r["flags"][name]) for r in rows) for name in FILTERS}
            summary.update({"resolved":sum(r["review_state"]=="resolved" for r in rows),
                            "unresolved":sum(r["review_state"]=="unresolved" for r in rows),
                            "binary_pending":sum(r["flags"]["binary"] and r["review_state"]!="resolved" for r in rows),"samples":len(self.images.allowed)})
            selected = [r for r in rows if (sample=="all" or r["sample_id"]==sample)
                and (dimension=="all" or r["dimension"]==dimension)
                and (not query or query.casefold() in (r["sample_id"]+" "+r["criterion_id"]+" "+r["name"]).casefold())]
            filter_counts = {name:sum(bool(r["flags"][name]) for r in selected) for name in FILTERS}
            selected = [r for r in selected if r["flags"][filter_name] and (state=="all" or
                        (r["review_state"]!="resolved" if state=="pending" else r["review_state"]==state))]
            selected.sort(key=lambda r:(r["sample_id"],r["criterion_id"]))
            pages = max(1,(len(selected)+page_size-1)//page_size)
            page = min(page,pages-1)
            return {"meta":self.dataset["meta"],"summary":summary,"csrf_token":self.token,
                    "samples":self.dataset["samples"],"filter_counts":filter_counts,"matched_items":len(selected),
                    "page":page,"pages":pages,"page_size":page_size,"rows":selected[page*page_size:(page+1)*page_size]}

    def save(self,payload):
        expected = {"sample_id","criterion_id","decision","final_label","review_reason","annotator","expected_revision","confirmed"}
        require(isinstance(payload,dict) and set(payload)==expected,"提交字段不完整或含未知内容。")
        with self.lock:
            self.guard()
            return self.store.save(payload["sample_id"],payload["criterion_id"],**{k:v for k,v in payload.items() if k not in ("sample_id","criterion_id")})

    def reload(self):
        with self.lock:
            dataset = load_review_data()
            require(dataset["binding"]==self.dataset["binding"],"复核来源已改变，不能沿用旧结论。")
            self.dataset = dataset
        return {"loaded_at":dataset["meta"]["loaded_at"]}

    def image(self,identifier):
        self.guard()
        return self.images.image(identifier)


def make_server(service,port=8503):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass

        def safe_origin(self):
            valid={f"127.0.0.1:{self.server.server_port}",f"localhost:{self.server.server_port}"}
            return self.headers.get("Host","") in valid and (self.headers.get("Origin") is None or
                self.headers.get("Origin") in {"http://"+h for h in valid})

        def send(self,value,status=200,kind="application/json; charset=utf-8",download=False):
            if isinstance(value,(dict,list)): value=json.dumps(value,ensure_ascii=False).encode()
            self.send_response(status)
            for key,val in (("Content-Type",kind),("Content-Length",str(len(value))),("Cache-Control","no-store"),
                ("X-Content-Type-Options","nosniff"),("Content-Security-Policy","default-src 'self'; img-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'; base-uri 'none'")):
                self.send_header(key,val)
            if download:self.send_header("Content-Disposition",'attachment; filename="holdout-reviews.json"')
            self.end_headers();self.wfile.write(value)

        def do_GET(self):
            if not self.safe_origin():return self.send({"error":"仅允许本机页面访问。"},403)
            parsed=urlsplit(self.path)
            try:
                static={"/":("holdout.html","text/html; charset=utf-8"),"/holdout.js":("holdout.js","text/javascript; charset=utf-8"),
                        "/review.css":("review.css","text/css; charset=utf-8"),"/holdout.css":("holdout.css","text/css; charset=utf-8")}
                if parsed.path in static:
                    name,kind=static[parsed.path];return self.send((WEB/name).read_bytes(),kind=kind)
                if parsed.path=="/api/health":return self.send({"app":"vlm-holdout-review","version":1})
                if parsed.path=="/api/review":
                    query=parse_qs(parsed.query,keep_blank_values=True)
                    require(set(query)<={"filter","state","sample","dimension","q","page","page_size"} and
                            all(len(v)==1 for v in query.values()),"筛选参数非法。")
                    get=lambda k,d:query.get(k,[d])[0]
                    return self.send(service.review(filter_name=get("filter","binary"),state=get("state","pending"),
                        sample=get("sample","all"),dimension=get("dimension","all"),query=get("q",""),
                        page=int(get("page","0")),page_size=int(get("page_size","5"))))
                if parsed.path.startswith("/api/image/"):return self.send(service.image(parsed.path.removeprefix("/api/image/")),kind="image/png")
                if parsed.path=="/api/export":
                    service.guard();return self.send(service.store.export(),download=True)
                return self.send({"error":"页面不存在。"},404)
            except ValueError as exc:return self.send({"error":str(exc)},400)
            except (OSError,KeyError,TypeError,sqlite3.Error):return self.send({"error":"本机复核数据读取失败，请重新核验。"},500)

        def do_POST(self):
            if not self.safe_origin() or not secrets.compare_digest(self.headers.get("X-Review-Token",""),service.token):
                return self.send({"error":"请从本机复核页操作或刷新页面。"},403)
            if self.path not in ("/api/save","/api/reload"):return self.send({"error":"此页不提供模型调用或规则修改接口。"},405)
            try:
                length=int(self.headers.get("Content-Length","0"))
                if not 0<=length<=65536:return self.send({"error":"提交内容过大。"},413)
                raw=self.rfile.read(length)
                if self.path=="/api/reload":return self.send(service.reload())
                require(self.headers.get_content_type()=="application/json","请以 JSON 提交。")
                return self.send({"review":service.save(client.strict_json(raw))})
            except ConflictError as exc:return self.send({"error":str(exc)},409)
            except ValueError as exc:return self.send({"error":str(exc)},400)
            except (OSError,KeyError,TypeError,sqlite3.Error):return self.send({"error":"复核保存未完成，请保留当前表单并重新核对。"},500)
    server=ThreadingHTTPServer(("127.0.0.1",port),Handler)
    server.daemon_threads=True
    return server


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",type=int,default=8503)
    args=parser.parse_args()
    service=HoldoutReviewService()
    server=make_server(service,args.port)
    print(f"保留集三方复核页：http://127.0.0.1:{server.server_port}",flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=="__main__":
    try:main()
    except (OSError,ValueError,KeyError,TypeError,sqlite3.Error):
        raise SystemExit("复核页启动失败，请核对 G01/G02、冻结参考与本机证据；没有发送模型请求。") from None
