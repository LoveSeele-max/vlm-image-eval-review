"""Local API settings window; all dependencies are in the Python standard library."""

import json
import os
import queue
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

from api_connection import ConnectionTestError, redact, test_connection, validate_settings

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "image-eval-review" / "config.local.json"
RECEIPT_PATH = ROOT / "private" / "api_connection_last_test.json"


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class SettingsWindow:
    def __init__(self, root, config_path=CONFIG_PATH, receipt_path=RECEIPT_PATH):
        self.root = root
        self.config_path = config_path
        self.receipt_path = receipt_path
        self.results = queue.Queue()
        self.busy = False
        self.closed = False
        self.last_receipt = None
        root.title("生图评测 · 接口配置")
        root.geometry("740x570")
        root.minsize(640, 520)
        root.configure(background="#f4f5f7")
        root.protocol("WM_DELETE_WINDOW", self.close)

        style = ttk.Style(root)
        style.configure(".", font=("Microsoft YaHei UI", 10))
        style.configure("Surface.TFrame", background="#ffffff")
        style.configure("Surface.TLabel", background="#ffffff")
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 15, "bold"), background="#ffffff")
        style.configure("State.TLabel", background="#f4f5f7", foreground="#3d4652")
        style.configure("TButton", padding=(12, 7))

        settings, load_error = self.load_settings()
        self.base_url = tk.StringVar(value=os.environ.get("VLM_BASE_URL") or settings.get("base_url", ""))
        self.model = tk.StringVar(value=os.environ.get("VLM_MODEL") or settings.get("model", "gpt-6.1-terra"))
        self.api_key = tk.StringVar(value=os.environ.get("VLM_API_KEY", ""))
        self.show_key = tk.BooleanVar(value=False)
        self.state = tk.StringVar(value=load_error or "尚未测试")
        self.build()
        self.root.bind("<Return>", lambda event: self.start_test())
        self.root.after(100, self.poll)

    def load_settings(self):
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Expected object")
            return {key: value for key, value in data.items()
                    if key in ("base_url", "model") and isinstance(value, str)}, ""
        except FileNotFoundError:
            return {}, ""
        except (OSError, ValueError):
            return {}, "本机配置无法读取，请重新填写。"

    def build(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        form = ttk.Frame(self.root, style="Surface.TFrame", padding=(24, 20))
        form.grid(row=0, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="接口配置", style="Title.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 18))
        ttk.Label(form, text="接口地址", style="Surface.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 16), pady=8)
        self.url_entry = ttk.Entry(form, textvariable=self.base_url)
        self.url_entry.grid(row=1, column=1, columnspan=2, sticky="ew", pady=8)
        ttk.Label(form, text="模型名", style="Surface.TLabel").grid(
            row=2, column=0, sticky="w", pady=8)
        self.model_entry = ttk.Entry(form, textvariable=self.model)
        self.model_entry.grid(row=2, column=1, columnspan=2, sticky="ew", pady=8)
        ttk.Label(form, text="API Key", style="Surface.TLabel").grid(
            row=3, column=0, sticky="w", pady=8)
        self.key_entry = ttk.Entry(form, textvariable=self.api_key, show="*")
        self.key_entry.grid(row=3, column=1, sticky="ew", pady=8)
        self.key_toggle = ttk.Checkbutton(
            form, text="显示", variable=self.show_key,
            command=lambda: self.key_entry.configure(show="" if self.show_key.get() else "*"))
        self.key_toggle.grid(row=3, column=2, padx=(10, 0))
        ttk.Label(form, text="Responses", style="Surface.TLabel").grid(
            row=4, column=1, sticky="w", pady=(4, 0))
        ttk.Label(form, text="密钥：仅本次会话", style="Surface.TLabel").grid(
            row=4, column=2, sticky="e", pady=(4, 0))

        result = ttk.Frame(self.root, padding=(24, 16, 24, 8))
        result.grid(row=1, column=0, sticky="nsew")
        result.columnconfigure(0, weight=1)
        result.rowconfigure(2, weight=1)
        ttk.Label(result, textvariable=self.state, style="State.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8))
        self.progress = ttk.Progressbar(result, mode="determinate", value=0)
        self.progress.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.output = scrolledtext.ScrolledText(
            result, height=9, wrap="word", font=("Microsoft YaHei UI", 10),
            background="#ffffff", foreground="#20252c", borderwidth=1, relief="solid",
            padx=12, pady=10, state="disabled")
        self.output.grid(row=2, column=0, sticky="nsew")

        actions = ttk.Frame(self.root, padding=(24, 8, 24, 20))
        actions.grid(row=2, column=0, sticky="ew")
        actions.columnconfigure(0, weight=1)
        self.save_button = ttk.Button(actions, text="保存配置", command=self.save)
        self.save_button.grid(row=0, column=0, sticky="w")
        self.test_button = ttk.Button(actions, text="保存并测试连接", command=self.start_test)
        self.test_button.grid(row=0, column=1, padx=(12, 0))
        ttk.Button(actions, text="关闭", command=self.close).grid(row=0, column=2, padx=(12, 0))
        self.url_entry.focus_set()

    def set_output(self, text):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")

    def settings(self):
        return validate_settings(self.base_url.get(), self.model.get())

    def save(self):
        if self.busy:
            return False
        try:
            settings = self.settings()
            save_json(self.config_path, settings)
        except ValueError as exc:
            self.state.set("配置未保存")
            self.set_output(redact(str(exc), self.api_key.get().strip()))
            return False
        except OSError:
            self.state.set("配置未保存")
            self.set_output("无法保存配置，请检查项目目录写入权限。")
            return False
        self.base_url.set(settings["base_url"])
        self.model.set(settings["model"])
        self.state.set("地址和模型名已保存；密钥未写入文件")
        return True

    def start_test(self):
        if self.busy:
            return
        if not self.api_key.get().strip():
            self.state.set("请填写 API Key")
            self.key_entry.focus_set()
            return
        if not self.save():
            return
        settings = self.settings()
        os.environ["VLM_API_KEY"] = self.api_key.get().strip()
        self.busy = True
        self.last_receipt = None
        self.state.set("正在测试文字连接…")
        self.set_output("")
        self.set_busy(True)
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        threading.Thread(target=self.worker, args=(settings,), daemon=True).start()

    def set_busy(self, busy):
        for widget in (self.url_entry, self.model_entry, self.key_entry,
                       self.key_toggle, self.save_button, self.test_button):
            widget.configure(state="disabled" if busy else "normal")

    def worker(self, settings):
        try:
            receipt = test_connection(settings["base_url"], settings["model"])
            self.results.put(("ok", receipt))
        except (ConnectionTestError, ValueError) as exc:
            self.results.put(("error", str(exc)))
        except Exception:
            self.results.put(("error", "测试未完成，请检查配置和网络后重试。"))

    def poll(self):
        if self.closed:
            return
        try:
            kind, value = self.results.get_nowait()
        except queue.Empty:
            pass
        else:
            self.busy = False
            self.set_busy(False)
            self.progress.stop()
            self.progress.configure(mode="determinate", value=0)
            if kind == "ok":
                self.last_receipt = value
                self.state.set("文字连接成功")
                self.render_receipt(value)
            else:
                self.state.set("连接未成功")
                error = redact(value, os.environ.get("VLM_API_KEY", ""))
                failure = {
                    "test_type": "text_connection_only",
                    "success": False,
                    "tested_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "requested_model": self.model.get(),
                    "error": error,
                }
                try:
                    save_json(self.receipt_path, redact(failure, os.environ.get("VLM_API_KEY", "")))
                except OSError:
                    error += "\n测试记录保存失败，请检查目录写入权限。"
                self.set_output(error)
        self.root.after(100, self.poll)

    def render_receipt(self, receipt):
        usage = receipt["usage"]
        show = lambda value: "未返回" if value is None else str(value)
        lines = [
            "回复：" + receipt["text"],
            "返回模型：" + show(receipt["returned_model"]),
            "接口状态：" + show(receipt["provider_status"]),
            "耗时：" + str(receipt["elapsed_seconds"]) + " 秒",
            "",
            "输入 token：" + show(usage["input_tokens"]),
            "其中缓存 token：" + show(usage["cached_tokens"]),
            "输出 token：" + show(usage["output_tokens"]),
            "合计 token：" + show(usage["total_tokens"]),
        ]
        try:
            save_json(self.receipt_path, receipt)
        except OSError:
            lines.extend(["", "测试成功；记录保存失败，请检查目录写入权限。"])
        self.set_output("\n".join(lines))

    def close(self):
        if self.busy and not messagebox.askyesno(
            "关闭窗口", "请求仍在进行。关闭后不再保存结果，已发出的请求仍可能计费。", parent=self.root):
            return
        self.closed = True
        self.api_key.set("")
        os.environ.pop("VLM_API_KEY", None)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    SettingsWindow(root)
    root.mainloop()
