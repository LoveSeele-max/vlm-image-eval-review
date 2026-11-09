"""Password-only local launcher for the full F01 development V1 batch."""
import argparse
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk

import development_v1 as dev
from run_store import RunStore


class DevelopmentV1Window:
    def __init__(self, root, batch_id=None):
        self.root, self.batch_id = root, batch_id
        self.store = RunStore()
        self.results = queue.Queue()
        self.busy = False
        root.title("开发集 V1 · 10 张图片")
        root.geometry("730x560")
        root.minsize(680, 510)
        root.protocol("WM_DELETE_WINDOW", self.close)
        frame = ttk.Frame(root, padding=22)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text="开发集 V1 批量运行", font=("Microsoft YaHei UI", 14, "bold")).grid(row=0, sticky="w")
        ttk.Label(frame, text="共 10 张，优先复用已验证成功结果。仅显示工程进度。\n密钥仅本窗口进程使用，运行结束清除。",
                  justify="left").grid(row=1, sticky="w", pady=12)
        self.key = tk.StringVar()
        ttk.Label(frame, text="API Key").grid(row=2, sticky="w")
        self.entry = ttk.Entry(frame, textvariable=self.key, show="*")
        self.entry.grid(row=3, sticky="ew", pady=(4, 12))
        self.button = ttk.Button(frame, text="开始开发集 V1", command=self.start)
        self.button.grid(row=4, sticky="w")
        self.message = tk.StringVar(value="准备就绪。填写密钥后开始，已有批次及重试历史会保留。")
        ttk.Label(frame, textvariable=self.message, wraplength=675, justify="left").grid(row=5, sticky="w", pady=14)
        root.bind("<Return>", lambda event: self.start())
        self.entry.focus_set()
        root.after(500, self.poll)

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
        self.message.set("正在核验正式人工参考、输入、配置和缓存，然后串行运行。")
        threading.Thread(target=self.run, args=(secret,), daemon=True).start()

    def run(self, secret):
        try:
            os.environ["VLM_API_KEY"] = secret
            result = dev.run_batch(self.store, self.batch_id,
                                   on_prepared=lambda identifier: setattr(self, "batch_id", identifier))
            batch = result["batches"][0]
            self.batch_id = batch["id"]
            succeeded = sum(t["status"] == "succeeded" for t in batch["tasks"])
            cached = sum(bool(t["cached_from"]) for t in batch["tasks"])
            self.results.put(f"本批工程校验通过 {succeeded}/{len(batch['tasks'])}，其中缓存复用 {cached} 条。\n"
                             f"状态：{batch['status']}；停止原因：{batch['stop_code'] or '无'}。\n"
                             "记录已保存，请返回聊天继续核验。")
        except (FileNotFoundError, dev.RunError):
            # Budget information contains no secrets. Other runtime details stay in private evidence.
            if not dev.BILLING.exists():
                self.results.put("还缺当前余额及单位的记录，请返回聊天完成账单核对。尚未发送请求。")
            else:
                self.results.put("核验未通过，已有记录保留。请返回聊天核对版本、配置和批次状态。")
        except Exception:
            self.results.put("运行未完成，已有证据保留。请返回聊天检查并恢复。")
        finally:
            os.environ.pop("VLM_API_KEY", None)

    def poll(self):
        try:
            message = self.results.get_nowait()
        except queue.Empty:
            if self.busy and self.batch_id:
                try:
                    batch = self.store.status(self.batch_id)["batches"][0]
                    self.message.set("运行中，请保留窗口。\n" + "\n".join(
                        f"{t['sample_id']}：{t['status']}（尝试 {len(t['attempts'])} 次）" for t in batch["tasks"]))
                except Exception:
                    pass
        else:
            self.busy = False
            self.message.set(message)
            self.entry.configure(state="normal")
            self.button.configure(state="normal")
        self.root.after(500, self.poll)

    def close(self):
        if self.busy:
            self.message.set("请求正在执行，请等待结束再关闭，以便保存完整证据。")
            return
        self.key.set("")
        os.environ.pop("VLM_API_KEY", None)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id")
    args = parser.parse_args()
    root = tk.Tk()
    DevelopmentV1Window(root, args.batch_id)
    root.mainloop()


if __name__ == "__main__":
    main()
