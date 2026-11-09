"""Open or start the gated G03 holdout review page without a console window."""
import json
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
from annotation_store import PRIVATE

ADDRESS="http://127.0.0.1:8503"


def running():
    try:
        with urllib.request.urlopen(ADDRESS+"/api/health",timeout=.8) as response:
            return json.load(response).get("app")=="vlm-holdout-review"
    except (OSError,ValueError):return False


def launch():
    if not running():
        PRIVATE.mkdir(parents=True,exist_ok=True)
        script=Path(__file__).with_name("holdout_review.py")
        with (PRIVATE/"holdout_review_server.log").open("ab") as log:
            process=subprocess.Popen([sys.executable,str(script)],cwd=script.parent.parent,
                stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=="win32" else 0)
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            if running():break
            if process.poll() is not None:raise RuntimeError("保留集复核页启动失败，请检查本机 private/holdout_review_server.log。")
            time.sleep(.5)
        else:raise RuntimeError("来源仍在核验，请稍后重新打开；不会新增模型调用。")
    webbrowser.open(ADDRESS)
    print("保留集三方复核页："+ADDRESS)


if __name__=="__main__":
    try:launch()
    except (OSError,RuntimeError) as exc:raise SystemExit(str(exc)) from None
