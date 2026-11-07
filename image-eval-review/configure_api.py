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
from capability_probe import TEST_TYPE, create_challenge, test_capabilities

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
        self.capability_receipt_path = receipt_path.with_name("api_capability_last_test.json")
        self.active_test = "text_connection_only"
        self.challenge = None
        self.previews = []
        self.results = queue.Queue()
        self.busy = False
        self.closed = False
        self.last_receipt = None
        root.title("生图评测 · 接口配置")
        root.geometry("820x760")
        root.minsize(760, 700)
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
        self.poll_id = self.root.after(100, self.poll)

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
        result.rowconfigure(3, weight=1)
        ttk.Label(result, textvariable=self.state, style="State.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8))
        self.progress = ttk.Progressbar(result, mode="determinate", value=0)
        self.progress.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.preview_frame = ttk.Frame(result)
        self.preview_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        for index in range(3):
            self.preview_frame.columnconfigure(index, weight=1)
        self.preview_labels = [
            ttk.Label(self.preview_frame, text=f"测试图 {index + 1}：点击多图测试后生成",
                      compound="top", anchor="center")
            for index in range(3)
        ]
        for index, label in enumerate(self.preview_labels):
            label.grid(row=0, column=index, sticky="ew")
        self.output = scrolledtext.ScrolledText(
            result, height=9, wrap="word", font=("Microsoft YaHei UI", 10),
            background="#ffffff", foreground="#20252c", borderwidth=1, relief="solid",
            padx=12, pady=10, state="disabled")
        self.output.grid(row=3, column=0, sticky="nsew")

        actions = ttk.Frame(self.root, padding=(24, 8, 24, 20))
        actions.grid(row=2, column=0, sticky="ew")
        actions.columnconfigure(0, weight=1)
        self.save_button = ttk.Button(actions, text="保存配置", command=self.save)
        self.save_button.grid(row=0, column=0, sticky="w")
        self.test_button = ttk.Button(actions, text="保存并测试连接", command=self.start_test)
        self.test_button.grid(row=0, column=1, padx=(12, 0))
        self.capability_button = ttk.Button(
            actions, text="多图＋JSON 测试", command=self.start_capability_test)
        self.capability_button.grid(row=0, column=2, padx=(12, 0))
        ttk.Button(actions, text="关闭", command=self.close).grid(row=0, column=3, padx=(12, 0))
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

    def start_capability_test(self):
        self.start_test(TEST_TYPE)

    def start_test(self, test_type="text_connection_only"):
        if self.busy:
            return
        if not self.api_key.get().strip():
            self.state.set("请填写 API Key")
            self.key_entry.focus_set()
            return
        if not self.save():
            return
        if test_type == TEST_TYPE:
            self.challenge = create_challenge()
            self.previews = [
                tk.PhotoImage(master=self.root, data=image["png_base64"]).subsample(2, 2)
                for image in self.challenge["images"]
            ]
            for index, (label, image) in enumerate(zip(self.preview_labels, self.previews), 1):
                label.configure(image=image, text=f"测试图 {index}")
        else:
            for index, label in enumerate(self.preview_labels, 1):
                label.configure(image="", text=f"测试图 {index}：点击多图测试后生成")
            self.previews = []
        settings = self.settings()
        os.environ["VLM_API_KEY"] = self.api_key.get().strip()
        self.active_test = test_type
        self.busy = True
        self.last_receipt = None
        self.state.set("正在测试多图＋JSON…" if test_type == TEST_TYPE else "正在测试文字连接…")
        self.set_output(
            "一次请求发送上方 3 张临时测试图，并要求 JSON Schema 输出。\n"
            "等待接口返回后，将分别检查格式和识图结果。可能产生接口费用。"
            if test_type == TEST_TYPE else "")
        self.set_busy(True)
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        threading.Thread(target=self.worker, args=(settings, test_type, self.challenge), daemon=True).start()

    def set_busy(self, busy):
        for widget in (self.url_entry, self.model_entry, self.key_entry,
                       self.key_toggle, self.save_button, self.test_button, self.capability_button):
            widget.configure(state="disabled" if busy else "normal")

    def worker(self, settings, test_type, challenge):
        try:
            if test_type == TEST_TYPE:
                receipt = test_capabilities(settings["base_url"], settings["model"], challenge)
            else:
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
                if value["test_type"] == TEST_TYPE:
                    self.state.set(
                        "三图识别与 JSON 格式检查通过" if value["success"]
                        else "请求已完成，但能力检查未通过")
                else:
                    self.state.set("文字连接成功")
                self.render_receipt(value)
            else:
                self.state.set("多图＋JSON 测试未完成" if self.active_test == TEST_TYPE else "连接未成功")
                error = redact(value, os.environ.get("VLM_API_KEY", ""))
                failure = {
                    "test_type": self.active_test,
                    "success": False,
                    "tested_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "requested_model": self.model.get(),
                    "error": error,
                }
                if self.active_test == TEST_TYPE:
                    failure.update({
                        "request_completed": False, "schema_valid": None, "visual_match": None,
                        "fixture_version": self.challenge["fixture_version"],
                        "expected": [
                            {key: image[key] for key in ("image_id", "color", "shape", "count")}
                            for image in self.challenge["images"]
                        ],
                        "image_sha256": [image["sha256"] for image in self.challenge["images"]],
                    })
                    error += "\n本次未完成组合测试，无法单独确认多图或 JSON Schema 能力。"
                self.last_receipt = failure
                try:
                    save_json(self.current_receipt_path(), redact(failure, os.environ.get("VLM_API_KEY", "")))
                except OSError:
                    error += "\n测试记录保存失败，请检查目录写入权限。"
                self.set_output(error)
        self.poll_id = self.root.after(100, self.poll)

    def current_receipt_path(self):
        return self.capability_receipt_path if self.active_test == TEST_TYPE else self.receipt_path

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
        if receipt["test_type"] == TEST_TYPE:
            prefix = [
                "组合请求：已完成",
                "JSON 本地校验：" + ("通过" if receipt["schema_valid"] else "未通过"),
                "三图识别核对：" + (
                    "通过" if receipt["visual_match"] else
                    "未通过" if receipt["visual_match"] is False else "未执行"),
            ]
            if receipt["schema_error"]:
                prefix.append("格式问题：" + receipt["schema_error"])
            for item in receipt.get("checks", []):
                prefix.append(
                    f"图 {item['position']}：{'通过' if item['match'] else '不符'}；"
                    f"预期 {self.describe(item['expected'])}；"
                    f"返回 {self.describe(item['actual'])}")
            prefix.extend([
                "",
                "这是基础能力检查；单次通过不能证明服务端强制执行了 Schema，也不代表正式评测准确率。",
                "",
            ])
            lines = prefix + lines
        try:
            save_json(self.current_receipt_path(), receipt)
        except OSError:
            lines.extend(["", "记录保存失败，请检查目录写入权限。"])
        self.set_output("\n".join(lines))

    @staticmethod
    def describe(item):
        colors = {"red": "红色", "green": "绿色", "blue": "蓝色",
                  "orange": "橙色", "purple": "紫色", "cyan": "青色"}
        shapes = {"circle": "圆形", "square": "正方形", "triangle": "三角形"}
        return (f"编号 {item['image_id']} / {item['count']} 个"
                f"{colors[item['color']]}{shapes[item['shape']]}")

    def close(self):
        if self.busy and not messagebox.askyesno(
            "关闭窗口", "请求仍在进行。关闭后不再保存结果，已发出的请求仍可能计费。", parent=self.root):
            return
        self.closed = True
        self.root.after_cancel(self.poll_id)
        self.api_key.set("")
        os.environ.pop("VLM_API_KEY", None)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    SettingsWindow(root)
    root.mainloop()
