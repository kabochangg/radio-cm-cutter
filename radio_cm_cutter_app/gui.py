from __future__ import annotations

import json
import os
import queue
import threading
import traceback
import webbrowser
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, W, Button, Entry, Frame, Label, StringVar, Tk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

from radio_cm_cutter.api import default_output_dir, process_folder_api

APP_NAME = "radio-cm-cutter"
SETTINGS_FILE = "gui_settings.json"


class RadioCmCutterGui:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Radio CM Cutter (MVP)")
        self.root.geometry("900x620")

        self.settings_path = self._settings_path()
        self.settings = self._load_settings()

        self.input_var = StringVar(value=self.settings.get("input_folder", ""))
        self.model_var = StringVar(value=self.settings.get("model_path", "model/model.pkl"))
        self.output_var = StringVar(value=self.settings.get("output_folder", str(default_output_dir())))
        self.ffmpeg_var = StringVar(value=self.settings.get("ffmpeg_path", ""))

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.running = False
        self.last_output_dir: Path | None = None

        self._build()
        self._poll_log_queue()

    def _settings_path(self) -> Path:
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else (Path.home() / ".config")
        target_dir = base / APP_NAME
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / SETTINGS_FILE

    def _load_settings(self) -> dict:
        if not self.settings_path.exists():
            return {}
        try:
            return json.loads(self.settings_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_settings(self) -> None:
        data = {
            "input_folder": self.input_var.get().strip(),
            "model_path": self.model_var.get().strip(),
            "output_folder": self.output_var.get().strip(),
            "ffmpeg_path": self.ffmpeg_var.get().strip(),
        }
        self.settings_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build(self) -> None:
        self._row("入力フォルダ", self.input_var, self._pick_input)
        self._row("モデル (.pkl)", self.model_var, self._pick_model)
        self._row("出力フォルダ", self.output_var, self._pick_output)
        self._row("FFmpegパス(任意)", self.ffmpeg_var, self._pick_ffmpeg)

        action = Frame(self.root)
        action.pack(fill="x", padx=12, pady=8)

        self.run_btn = Button(action, text="実行", width=14, command=self._start)
        self.run_btn.pack(side=LEFT)

        self.open_report_btn = Button(action, text="ml_report.html を開く", state="disabled", command=self._open_report)
        self.open_report_btn.pack(side=LEFT, padx=8)

        self.log_text = ScrolledText(self.root, wrap="word", height=22)
        self.log_text.pack(fill=BOTH, expand=True, padx=12, pady=8)

    def _row(self, label: str, var: StringVar, picker) -> None:
        row = Frame(self.root)
        row.pack(fill="x", padx=12, pady=6)
        Label(row, text=label, width=18, anchor=W).pack(side=LEFT)
        Entry(row, textvariable=var).pack(side=LEFT, fill="x", expand=True, padx=8)
        Button(row, text="参照...", width=10, command=picker).pack(side=RIGHT)

    def _pick_input(self) -> None:
        p = filedialog.askdirectory(title="入力フォルダを選択")
        if p:
            self.input_var.set(p)

    def _pick_model(self) -> None:
        p = filedialog.askopenfilename(title="model.pkl を選択", filetypes=[("Pickle", "*.pkl"), ("All", "*.*")])
        if p:
            self.model_var.set(p)

    def _pick_output(self) -> None:
        p = filedialog.askdirectory(title="出力フォルダを選択")
        if p:
            self.output_var.set(p)

    def _pick_ffmpeg(self) -> None:
        p = filedialog.askopenfilename(title="ffmpeg.exe または ffmpeg/bin フォルダを選択", filetypes=[("Executable", "*.exe"), ("All", "*.*")])
        if p:
            self.ffmpeg_var.set(p)

    def _append_log(self, message: str) -> None:
        self.log_text.insert(END, message + "\n")
        self.log_text.see(END)

    def _enqueue_log(self, message: str) -> None:
        self.log_queue.put(message)

    def _poll_log_queue(self) -> None:
        try:
            while True:
                self._append_log(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        self.root.after(150, self._poll_log_queue)

    def _set_running(self, running: bool) -> None:
        self.running = running
        self.run_btn.configure(state="disabled" if running else "normal")

    def _start(self) -> None:
        if self.running:
            return
        input_dir = self.input_var.get().strip()
        if not input_dir:
            messagebox.showwarning("入力不足", "入力フォルダを選択してください。")
            return

        output_dir = self.output_var.get().strip() or str(default_output_dir())
        self.output_var.set(output_dir)

        self.open_report_btn.configure(state="disabled")
        self._save_settings()
        self._set_running(True)
        self._append_log("=== 実行開始 ===")

        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def _worker(self) -> None:
        try:
            result = process_folder_api(
                folder=self.input_var.get().strip(),
                model_path=self.model_var.get().strip() or None,
                out_dir=self.output_var.get().strip() or None,
                ffmpeg_path=self.ffmpeg_var.get().strip() or None,
                recursive=True,
                log_callback=self._enqueue_log,
            )
            self.last_output_dir = result.output_dir
            report_path = result.output_dir / "ml_report.html"
            if report_path.exists():
                self.root.after(0, lambda: self.open_report_btn.configure(state="normal"))
            self._enqueue_log(f"完了: success={result.success}, failed={result.failed}, output={result.output_dir}")
        except Exception as exc:
            msg = self._friendly_error(exc)
            self._enqueue_log(msg)
            self._enqueue_log(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("実行エラー", msg))
        finally:
            self.root.after(0, lambda: self._set_running(False))

    def _friendly_error(self, exc: Exception) -> str:
        raw = str(exc)
        lower = raw.lower()
        if "required command not found" in lower or "ffmpeg" in lower or "ffprobe" in lower:
            return (
                "原因: ffmpeg / ffprobe が見つかりません。\n"
                "次の行動: 1) FFmpeg をインストールして PATH に追加 2) もしくは ffmpeg.exe を選択して再実行してください。"
            )
        if isinstance(exc, FileNotFoundError):
            return f"原因: ファイルまたはフォルダが見つかりません ({raw})\n次の行動: 入力フォルダ/モデルパス/設定ファイルの存在を確認してください。"
        if isinstance(exc, PermissionError):
            return f"原因: 権限エラーです ({raw})\n次の行動: 書き込み可能な出力フォルダを選択してください。"
        return f"原因: {raw}\n次の行動: ログを確認し、設定を見直して再実行してください。"

    def _open_report(self) -> None:
        if not self.last_output_dir:
            return
        report = self.last_output_dir / "ml_report.html"
        if not report.exists():
            messagebox.showinfo("情報", "ml_report.html が見つかりません。")
            return
        webbrowser.open(report.resolve().as_uri())


def main() -> None:
    root = Tk()
    RadioCmCutterGui(root)
    root.mainloop()
