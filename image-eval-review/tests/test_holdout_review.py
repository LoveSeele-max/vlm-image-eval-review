"""G03 fixtures: isolated author confirmation/history, HTTP guards, source gate, no API."""
import copy
import json
import sqlite3
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import adjudication_store as a
import holdout_review as h
from test_review_app import label


def dataset():
    rows=[]
    for i in range(8):
        cid="synthetic_"+str(i)
        row={"sample_id":"sample_001" if i<4 else "sample_002","criterion_id":cid,
            "name":"合成条目 "+str(i),"dimension":"subject_features","required":i%2==0,
            "human":h.public_label(label(cid,"pass")),"v1":h.public_label(label(cid,"fail")),
            "v2":h.public_label(label(cid,"unjudgeable" if i==0 else "fail")),
            "human_provenance":"synthetic author","pass_when":"synthetic pass","fail_when":"synthetic fail",
            "unjudgeable_when":"synthetic unknown","boundary":"synthetic boundary"}
        row["flags"]=h.flags(row);rows.append(row)
    return {"meta":{"reference_version":"synthetic_reference","loaded_at":"synthetic_time"},
        "rows":rows,"samples":[{"sample_id":sid,"prompt":"synthetic <script>window.bad=1</script>",
            "engineering":{v:{"status":"succeeded","error_code":None} for v in h.metrics.VERSIONS}}
            for sid in ("sample_001","sample_002")],"binding":{"experiment":"synthetic_g03"},"watched":{}}


def payload(row,**changes):
    value={"sample_id":row["sample_id"],"criterion_id":row["criterion_id"],"decision":"keep","final_label":None,
        "review_reason":"Synthetic author checked this visible region against the fixed rule.","annotator":"synthetic author",
        "expected_revision":0,"confirmed":True}
    value.update(changes)
    return value


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.folder=Path(self.tmp.name);self.data=dataset()
        self.service=h.HoldoutReviewService(self.folder/"reviews.sqlite3",dataset=self.data)

    def save(self,**changes):return self.service.save(payload(self.data["rows"][0],**changes))


class ReviewTests(FixtureCase):
    def test_default_first_five_pending_and_filter_counts(self):
        got=self.service.review()
        self.assertEqual(len(got["rows"]),5)
        self.assertEqual(got["matched_items"],8)
        self.assertEqual(got["summary"]["binary_pending"],8)
        self.assertEqual(got["filter_counts"]["version_change"],1)
        self.assertEqual(len(self.service.review(page=1)["rows"]),3)
        self.assertEqual(self.service.review(sample="sample_002")["matched_items"],4)

    def test_no_confirmation_or_reason_never_writes(self):
        for change in ({"confirmed":False},{"confirmed":"true"},{"review_reason":" "},{"annotator":""},
                       {"expected_revision":True},{"decision":"automatic"}):
            with self.assertRaises(ValueError):self.save(**change)
        self.assertEqual(self.service.store.snapshot(),({},[]))

    def test_keep_records_source_label_and_confirmation_without_mutating_source(self):
        before=copy.deepcopy(self.data)
        saved=self.save()
        self.assertEqual(saved["decision"],"keep")
        self.assertEqual(saved["frozen_label"],saved["final_label"])
        self.assertTrue(saved["confirmed_by_author"])
        self.assertEqual(self.data,before)
        self.assertEqual(self.service.review()["matched_items"],7)
        self.assertEqual(self.service.review(state="resolved")["matched_items"],1)

    def test_revision_retains_original_and_contiguous_history(self):
        self.save()
        revised=label("synthetic_0","fail")
        saved=self.save(decision="revise",final_label=revised,expected_revision=1,review_reason="Synthetic corrected visual evidence.")
        self.assertEqual(saved["revision"],2)
        self.assertEqual(saved["frozen_label"]["status"],"pass")
        self.assertEqual(saved["final_label"]["status"],"fail")
        export=self.service.store.export()
        self.assertEqual([r["revision"] for r in export["history"]],[1,2])
        self.assertEqual(export["resolved_items"],1)
        self.assertEqual(self.service.store.snapshot()[0]["sample_001","synthetic_0"],saved)

    def test_unresolved_has_no_final_label_and_stays_pending(self):
        self.save(decision="unresolved")
        self.assertEqual(self.service.review()["matched_items"],8)
        self.assertEqual(self.service.review(state="unresolved")["matched_items"],1)
        self.assertEqual(self.service.store.export()["resolved_items"],0)
        self.assertIsNone(self.service.store.export()["current_reviews"][0]["final_label"])

    def test_duplicate_request_stale_revision_and_concurrent_writer(self):
        saved=self.save()
        with self.assertRaises(a.ConflictError):self.save()
        self.assertEqual(self.save(expected_revision=1),saved)
        def change(_):
            try:return self.save(expected_revision=1,review_reason="Synthetic concurrent change.")["revision"]
            except a.ConflictError:return "conflict"
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(change,range(2)))
        self.assertCountEqual(results,[2,"conflict"])
        self.assertEqual(len(self.service.store.export()["history"]),2)

    def test_unknown_development_fields_or_bad_final_label_rejected(self):
        with self.assertRaises(ValueError):self.service.save({**payload(self.data["rows"][0]),"sample_id":"sample_008"})
        with self.assertRaises(ValueError):self.service.save({**payload(self.data["rows"][0]),"model":"send"})
        for final in (label("wrong","fail"),label("synthetic_0","not_applicable"),
                      {**label("synthetic_0","fail"),"candidate_observation":""},
                      {"criterion_id":"synthetic_0",**self.data["rows"][0]["human"]}):
            with self.assertRaises(ValueError):self.save(decision="revise",final_label=final)
        with self.assertRaises(ValueError):self.save(decision="keep",final_label=label("synthetic_0"))

    def test_binding_or_history_tamper_rejected(self):
        self.save()
        with self.assertRaises(ValueError):a.AdjudicationStore({"experiment":"changed"},self.data["rows"],self.service.store.path)
        with self.service.store.connect() as db:db.execute("UPDATE review_history SET payload='{}'")
        with self.assertRaises(ValueError):self.service.store.snapshot()

    def test_history_delete_cannot_hide_current_review(self):
        self.save()
        with self.service.store.connect() as db:db.execute("DELETE FROM review_history")
        with self.assertRaises(ValueError):self.service.store.snapshot()

    def test_protected_database_names_and_filter_guard(self):
        for name in ("annotations.sqlite3","RUNS.sqlite3"):
            with self.assertRaises(ValueError):a.AdjudicationStore({},self.data["rows"],self.folder/name)
        for kwargs in ({"sample":"sample_008"},{"filter_name":"bad"},{"state":"automatic"},{"page":-1},{"page_size":1000}):
            with self.assertRaises(ValueError):self.service.review(**kwargs)

    def test_changed_source_blocks_save_and_image(self):
        p=self.folder/"source.json";p.write_text("original",encoding="utf-8")
        self.data["watched"][str(p)]=h.client.digest(p.read_bytes())
        p.write_text("changed",encoding="utf-8")
        with self.assertRaises(ValueError):self.save()
        with self.assertRaises(ValueError):self.service.image("sample_001")
        self.assertEqual(self.service.store.snapshot(),({},[]))

    def test_unlock_failure_never_initializes_review_store(self):
        with patch.object(h,"load_review_data",side_effect=ValueError("freeze gate")),patch.object(h,"AdjudicationStore") as store:
            with self.assertRaises(ValueError):h.HoldoutReviewService(self.folder/"new.sqlite3")
            store.assert_not_called()

    def test_healthy_gate_requires_G02_before_loading_holdout(self):
        latest={"output":str(h.client.PRIVATE/"holdout_metrics/test")}
        with patch.object(Path,"read_bytes",return_value=json.dumps(latest).encode()), \
             patch.object(h.metrics,"verify_export",side_effect=ValueError("G02 corrupt")), \
             patch.object(h.metrics,"load_sources") as source:
            with self.assertRaises(ValueError):h.load_review_data()
            source.assert_not_called()

    def test_flags_distinguish_uncertainty_engineering_and_binary_disagreement(self):
        row=copy.deepcopy(self.data["rows"][0]);row["v1"]=None;row["v2"]["status"]="unjudgeable"
        got=h.flags(row)
        self.assertFalse(got["binary"]);self.assertFalse(got["version_change"])
        self.assertTrue(got["engineering"]);self.assertTrue(got["ai_uncertain"])


class HttpTests(FixtureCase):
    def setUp(self):
        super().setUp()
        self.server=h.make_server(self.service,0)
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        self.addCleanup(self.server.server_close);self.addCleanup(self.server.shutdown)
        self.url=f"http://127.0.0.1:{self.server.server_port}"

    def call(self,path,method="GET",body=None,headers=None):
        data=json.dumps(body).encode() if body is not None else None
        request=urllib.request.Request(self.url+path,data=data,method=method,headers=headers or {})
        try:
            with urllib.request.urlopen(request,timeout=5) as r:return r.status,r.headers,r.read()
        except urllib.error.HTTPError as e:
            with e:return e.code,e.headers,e.read()

    def test_http_confirmed_save_export_csrf_origin_and_no_send_routes(self):
        good={"Content-Type":"application/json","X-Review-Token":self.service.token,"Origin":self.url}
        code,_,raw=self.call("/api/review")
        self.assertEqual(code,200);self.assertEqual(json.loads(raw)["csrf_token"],self.service.token)
        self.assertEqual(self.call("/api/save","POST",payload(self.data["rows"][0]))[0],403)
        self.assertEqual(self.call("/api/save","POST",payload(self.data["rows"][0]),{**good,"Origin":"https://outside.invalid"})[0],403)
        self.assertEqual(self.call("/api/save","POST",payload(self.data["rows"][0],confirmed=False),good)[0],400)
        with patch.object(h.client,"execute_request") as execute:
            self.assertEqual(self.call("/api/run","POST",{},good)[0],405)
            self.assertEqual(self.call("/api/save","POST",payload(self.data["rows"][0]),good)[0],200)
            execute.assert_not_called()
        self.assertEqual(self.call("/api/save","POST",payload(self.data["rows"][0]),good)[0],409)
        code,headers,raw=self.call("/api/export")
        self.assertEqual(code,200);self.assertIn("attachment",headers["Content-Disposition"])
        self.assertEqual(len(json.loads(raw)["history"]),1)

    def test_http_assets_literal_markup_and_bad_queries(self):
        for path in ("/","/holdout.js","/holdout.css","/review.css","/api/health"):
            self.assertEqual(self.call(path)[0],200)
        self.assertEqual(self.call("/api/review?sample=sample_008")[0],400)
        self.assertEqual(self.call("/api/review?filter=all&filter=binary")[0],400)
        self.assertEqual(self.call("/api/image/sample_008")[0],400)
        self.assertEqual(self.call("/private/adjudications.sqlite3")[0],404)
        self.assertNotIn("innerHTML",(h.WEB/"holdout.js").read_text(encoding="utf-8"))


if __name__=="__main__":unittest.main()
