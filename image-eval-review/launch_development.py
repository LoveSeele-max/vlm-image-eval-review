"""Password-only launcher for the three D04 development samples."""

import argparse
import os
import queue
import sqlite3
import threading
import tkinter as tk
from tkinter import ttk

import judge_client as client
from run_store import RunError, RunStore
from run_tasks import Worker


SAMPLES = ("sample_020", "sample_029", "sample_042")
STATUS_PATH = client.PRIVATE / "d04_run_status.json"


def prepare_batch(store, batch_id=None):
    """Reuse the existing experiment, including its retry history and limits."""
    if batch_id is None:
        for batch in reversed(store.status()["batches"]):
            if not batch["repeat"] and tuple(t["sample_id"] for t in batch["tasks"]) == SAMPLES:
                batch_id = batch["id"]
                break
    if batch_id is None:
        settings = client.settings_from_environment()
        requests = [client.build_request(sid, settings["model"]) for sid in SAMPLES]
        batch_id = store.create(requests, settings["base_url"])
    batch = store.batch(batch_id)
    tasks = store.tasks(batch_id)
    if batch["repeat"] or tuple(t["sample_id"] for t in tasks) != SAMPLES:
        raise RunError("请选择包含三条指定开发样本的普通批次。")
    worker = Worker(store)
    for task in tasks:
        client.require_initial_snapshot(task["sample_id"])
        worker.inputs(task, batch)
    return batch_id


class DevelopmentWindow:
    def __init__(self, root, batch_id=None):
        self.root = root
        self.batch_id = batch_id
        self.store = RunStore()
        self.results = queue.Queue()
        self.busy = False
        root.title("D04 · 三条开发样本联调")
        root.geometry("650x340")
        root.minsize(600, 320)
        root.protocol("WM_DELETE_WINDOW", self.close)
        frame = ttk.Frame(root, padding=24)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text="运行 sample_020、sample_029、sample_042",
                  font=("Microsoft YaHei UI", 13, "bold")).grid(row=0, sticky="w", pady=(0, 12))
        ttk.Label(frame, text="填写 API Key 后点击开始。密钥仅用于本窗口进程，关闭后清除。").grid(
            row=1, sticky="w", pady=(0, 12))
        self.key = tk.StringVar()
        self.entry = ttk.Entry(frame, textvariable=self.key, show="*")
        self.entry.grid(row=2, sticky="ew", pady=(0, 12))
        self.button = ttk.Button(frame, text="开始运行", command=self.start)
        self.button.grid(row=3, sticky="w", pady=(0, 16))
        self.message = tk.StringVar(value="等待输入密钥；尚未发起请求。")
        ttk.Label(frame, textvariable=self.message, wraplength=580, justify="left").grid(
            row=4, sticky="w")
        root.bind("<Return>", lambda event: self.start())
        self.entry.focus_set()
        root.after(300, self.poll)

    def start(self):
        if self.busy:
            return
        secret = self.key.get().strip()
        if not secret or not secret.isascii() or any(c.isspace() for c in secret):
            self.message.set("请填写有效的 API Key。")
            return
        self.key.set("")
        self.busy = True
        self.entry.configure(state="disabled")
        self.button.configure(state="disabled")
        self.message.set("正在核验快照及输入，然后串行调用三条样本。请保留窗口。")
        threading.Thread(target=self.run, args=(secret,), daemon=True).start()

    def run(self, secret):
        try:
            batch_id = prepare_batch(self.store, self.batch_id)
            self.batch_id = batch_id
            os.environ["VLM_API_KEY"] = secret
            result = Worker(self.store).run(batch_id)
            client.write_json(STATUS_PATH, result, secret)
            batch = result["batches"][0]
            count = sum(t["status"] == "succeeded" for t in batch["tasks"])
            self.results.put(f"本批已通过工程校验 {count}/3 条。\n"
                             f"批次状态：{batch['status']}；停止原因：{batch['stop_code'] or '无'}。\n"
                             "运行记录已保存，可以返回聊天继续核验。")
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
            self.results.put("运行未完成。请返回聊天检查快照、输入版本或运行库；已有记录保留。")
        except Exception:
            self.results.put("运行异常。已有证据保留，请返回聊天检查并恢复。")
        finally:
            os.environ.pop("VLM_API_KEY", None)

    def poll(self):
        try:
            message = self.results.get_nowait()
        except queue.Empty:
            if self.busy and self.batch_id:
                try:
                    batch = self.store.status(self.batch_id)["batches"][0]
                    lines = [f"{t['sample_id']}：{t['status']}（{len(t['attempts'])} 次尝试）"
                             for t in batch["tasks"]]
                    self.message.set("正在运行，请保留窗口。\n" + "\n".join(lines))
                except (OSError, ValueError, sqlite3.Error):
                    pass
        else:
            self.busy = False
            self.message.set(message)
            self.entry.configure(state="normal")
            self.button.configure(state="normal")
        self.root.after(500, self.poll)

    def close(self):
        if self.busy:
            self.message.set("请求正在执行。请等待结束后关闭，以便完整保存证据。")
            return
        self.key.set("")
        os.environ.pop("VLM_API_KEY", None)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id")
    args = parser.parse_args()
    root = tk.Tk()
    DevelopmentWindow(root, args.batch_id)
    root.mainloop()


if __name__ == "__main__":
    main()
