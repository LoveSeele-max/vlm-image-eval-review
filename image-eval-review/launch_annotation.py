"""Open the existing local annotation app, or start it without a console window."""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

from annotation_store import PRIVATE


ADDRESS = "http://127.0.0.1:8501"


def running():
    try:
        with urllib.request.urlopen(ADDRESS + "/api/health", timeout=0.5) as response:
            return json.loads(response.read()).get("app") == "vlm-image-annotation"
    except (OSError, ValueError, urllib.error.URLError):
        return False


def launch():
    if not running():
        PRIVATE.mkdir(parents=True, exist_ok=True)
        script = Path(__file__).resolve().with_name("annotation_app.py")
        with (PRIVATE / "annotation_server.log").open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, str(script)], cwd=script.parent.parent,
                stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if running():
                break
            if process.poll() is not None:
                raise RuntimeError("标注页启动失败，请查看 image-eval-review/private/annotation_server.log。")
            time.sleep(0.25)
        else:
            raise RuntimeError("标注页尚未就绪，请查看本机启动日志后重试。")
    webbrowser.open(ADDRESS)
    print("标注页已打开：http://127.0.0.1:8501")


if __name__ == "__main__":
    try:
        launch()
    except (OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
