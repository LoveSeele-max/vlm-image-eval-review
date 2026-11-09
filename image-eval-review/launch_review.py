"""Reuse or start the read-only F02 review server without a console window."""
import json
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

from annotation_store import PRIVATE

ADDRESS = "http://127.0.0.1:8502"


def running():
    try:
        with urllib.request.urlopen(ADDRESS + "/api/health", timeout=0.6) as response:
            return json.load(response).get("app") == "vlm-development-review"
    except (OSError, ValueError):
        return False


def launch():
    if not running():
        PRIVATE.mkdir(parents=True, exist_ok=True)
        script = Path(__file__).with_name("review_app.py")
        with (PRIVATE / "review_server.log").open("ab") as log:
            process = subprocess.Popen([sys.executable, str(script)], cwd=script.parent.parent,
                stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if running():
                break
            if process.poll() is not None:
                raise RuntimeError("分歧页启动失败，请检查本机 private/review_server.log。")
            time.sleep(0.25)
        else:
            raise RuntimeError("证据核验尚未就绪，请检查本机 private/review_server.log。")
    webbrowser.open(ADDRESS)
    print("开发集人机对照页：http://127.0.0.1:8502")


if __name__ == "__main__":
    try:
        launch()
    except (OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
