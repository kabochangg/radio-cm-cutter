from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import traceback
import urllib.error
import urllib.request
import webbrowser
import zipfile
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, W, Button, Entry, Frame, Label, StringVar, Tk, Toplevel, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

from radio_cm_cutter.api import default_output_dir, process_folder_api
from radio_cm_cutter_app.diagnostics import collect_diagnostics, export_diagnostic_zip
from radio_cm_cutter_app.ffmpeg_support import FFMPEG_ARCHIVE_NAME, FFMPEG_DOWNLOAD_URL
from radio_cm_cutter_app.runtime_paths import config_dir, bundled_resource_path, ffmpeg_dir, logs_dir

APP_NAME = "radio-cm-cutter"
SETTINGS_FILE = "gui_settings.json"


class RadioCmCutterGui:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Radio CM Cutter (MVP)")
        self.root.geometry("900x620")

        self.settings_path = self._settings_path()
        self.settings = self._load_settings()
        self.config_path = bundled_resource_path("config.json")

        self.input_var = StringVar(value=self.settings.get("input_folder", ""))
        self.model_var = StringVar(value=self.settings.get("model_path", self._default_model_path()))
        self.output_var = StringVar(value=self.settings.get("output_folder", str(default_output_dir())))
        self.ffmpeg_var = StringVar(value=self.settings.get("ffmpeg_path", ""))

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.running = False
        self.last_output_dir: Path | None = None

        self.startup_logger = self._create_logger("startup")
        self.run_logger = self._create_logger("run")
        self.startup_logger.info("GUI started")

        self._build()
        self.root.after(150, self._check_ffmpeg_on_startup)
        self._poll_log_queue()

    def _settings_path(self) -> Path:
        return config_dir() / SETTINGS_FILE

    def _create_logger(self, kind: str) -> logging.Logger:
        log_path = logs_dir() / f"{kind}.log"
        logger = logging.getLogger(f"{APP_NAME}.{kind}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path for h in logger.handlers):
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(fh)
        return logger

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

    def _default_model_path(self) -> str:
        return str(bundled_resource_path("model", "model.pkl"))

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

        self.diag_btn = Button(action, text="診断", command=self._open_diagnostics_dialog)
        self.diag_btn.pack(side=LEFT, padx=8)

        self.open_logs_btn = Button(action, text="ログを開く", command=self._open_logs_folder)
        self.open_logs_btn.pack(side=LEFT, padx=8)

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
            self._save_settings()

    def _ffmpeg_search_candidates(self) -> list[Path]:
        selected = self.ffmpeg_var.get().strip()
        candidates: list[Path] = []
        if selected:
            sel = Path(selected).expanduser()
            candidates.extend([sel, sel.parent / "ffprobe.exe"] if sel.suffix.lower() == ".exe" else [sel / "ffmpeg.exe", sel / "ffprobe.exe", sel / "bin" / "ffmpeg.exe", sel / "bin" / "ffprobe.exe"])
        install_dir = ffmpeg_dir()
        candidates.extend(
            [
                install_dir / "ffmpeg.exe",
                install_dir / "ffprobe.exe",
                install_dir / "bin" / "ffmpeg.exe",
                install_dir / "bin" / "ffprobe.exe",
            ]
        )
        return candidates

    def _resolve_ffmpeg_state(self) -> tuple[bool, list[str], str]:
        searched: list[str] = []
        selected = self.ffmpeg_var.get().strip()
        if selected:
            p = Path(selected).expanduser()
            if p.is_file():
                probe = p.parent / "ffprobe.exe"
                searched.extend([str(p), str(probe)])
                if p.name.lower() == "ffmpeg.exe" and probe.exists():
                    return True, searched, str(p)
            elif p.is_dir():
                f1 = p / "ffmpeg.exe"
                p1 = p / "ffprobe.exe"
                f2 = p / "bin" / "ffmpeg.exe"
                p2 = p / "bin" / "ffprobe.exe"
                searched.extend([str(f1), str(p1), str(f2), str(p2)])
                if (f1.exists() and p1.exists()) or (f2.exists() and p2.exists()):
                    return True, searched, str(p)
            else:
                searched.append(str(p))

        which_ffmpeg = shutil.which("ffmpeg")
        which_ffprobe = shutil.which("ffprobe")
        searched.append(f"PATH: ffmpeg={which_ffmpeg or '未検出'}")
        searched.append(f"PATH: ffprobe={which_ffprobe or '未検出'}")
        if which_ffmpeg and which_ffprobe:
            return True, searched, which_ffmpeg

        for c in self._ffmpeg_search_candidates():
            s = str(c)
            if s not in searched:
                searched.append(s)

        return False, searched, ""

    def _check_ffmpeg_on_startup(self) -> None:
        ok, searched, _ = self._resolve_ffmpeg_state()
        selected = self.ffmpeg_var.get().strip()
        if ok:
            return
        if selected and not Path(selected).expanduser().exists():
            self._show_ffmpeg_missing_dialog(searched, reason="前回指定された ffmpeg パスが見つかりませんでした。")

    def _download_ffmpeg(self) -> tuple[bool, str]:
        target_dir = ffmpeg_dir()
        archive = target_dir / FFMPEG_ARCHIVE_NAME
        try:
            urllib.request.urlretrieve(FFMPEG_DOWNLOAD_URL, archive)
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(target_dir)
            for candidate in target_dir.rglob("ffmpeg.exe"):
                probe = candidate.parent / "ffprobe.exe"
                if probe.exists():
                    self.ffmpeg_var.set(str(candidate))
                    self._save_settings()
                    return True, f"ダウンロード完了: {candidate}"
            return False, "ダウンロードは完了しましたが ffmpeg.exe / ffprobe.exe を見つけられませんでした。"
        except urllib.error.URLError as exc:
            return False, f"ネットワークエラー: {exc.reason}"
        except zipfile.BadZipFile:
            return False, "ダウンロードしたファイルが壊れているため解凍できませんでした。"
        except Exception as exc:
            return False, str(exc)
        finally:
            if archive.exists():
                archive.unlink(missing_ok=True)

    def _open_readme_guide(self) -> None:
        readme = bundled_resource_path("README.md")
        if not readme.exists():
            messagebox.showwarning("README未検出", "導入手順ファイル README.md が見つかりませんでした。")
            return
        webbrowser.open(readme.resolve().as_uri())

    def _show_ffmpeg_missing_dialog(self, searched: list[str], reason: str | None = None) -> bool:
        dlg = Toplevel(self.root)
        dlg.title("FFmpeg が見つかりません")
        dlg.geometry("760x500")
        dlg.transient(self.root)
        dlg.grab_set()

        description = (
            "ffmpeg / ffprobe が見つからないため実行できません。\n"
            "自動で探した場所と、次にやることを確認してください。"
        )
        if reason:
            description = f"{reason}\n\n{description}"
        Label(dlg, text=description, justify="left", anchor="w").pack(fill="x", padx=12, pady=(12, 6))

        next_step = (
            "次にやること:\n"
            "1) [ffmpeg.exeを選択] で ffmpeg.exe（または ffmpeg/bin を含むフォルダ）を指定\n"
            "2) [導入手順を開く(README)] で導入手順を確認\n"
            "3) （任意）[ffmpegをダウンロード] で自動取得"
        )
        Label(dlg, text=next_step, justify="left", anchor="w").pack(fill="x", padx=12, pady=(0, 8))

        text = ScrolledText(dlg, wrap="word", height=16)
        text.pack(fill=BOTH, expand=True, padx=12, pady=8)
        text.insert("1.0", "自動で探した場所:\n- " + "\n- ".join(searched))
        text.configure(state="disabled")

        decision = {"ok": False}

        def pick() -> None:
            p = filedialog.askopenfilename(
                title="ffmpeg.exe を選択",
                filetypes=[("FFmpeg executable", "ffmpeg.exe"), ("Executable", "*.exe"), ("All", "*.*")],
            )
            if not p:
                return
            self.ffmpeg_var.set(p)
            self._save_settings()
            ok, _, _ = self._resolve_ffmpeg_state()
            if ok:
                decision["ok"] = True
                dlg.destroy()
                return
            messagebox.showwarning("ffprobe未検出", "ffprobe.exe も必要です。ffmpeg/bin 一式を指定してください。")

        def download() -> None:
            self._enqueue_log("FFmpeg ダウンロードを開始します...")
            ok, msg = self._download_ffmpeg()
            self._enqueue_log(msg)
            if ok:
                messagebox.showinfo("FFmpeg", msg)
                decision["ok"] = True
                dlg.destroy()
            else:
                messagebox.showerror("FFmpeg ダウンロード失敗", msg)

        btns = Frame(dlg)
        btns.pack(fill="x", padx=12, pady=(0, 12))
        Button(btns, text="ffmpeg.exeを選択", width=18, command=pick).pack(side=LEFT)
        Button(btns, text="導入手順を開く(README)", width=22, command=self._open_readme_guide).pack(side=LEFT, padx=8)
        Button(btns, text="ffmpegをダウンロード", width=18, command=download).pack(side=LEFT, padx=8)
        Button(btns, text="キャンセル", width=12, command=dlg.destroy).pack(side=RIGHT)

        self.root.wait_window(dlg)
        return decision["ok"]

    def _append_log(self, message: str) -> None:
        self.log_text.insert(END, message + "\n")
        self.log_text.see(END)

    def _enqueue_log(self, message: str) -> None:
        self.run_logger.info(message)
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

        ok, searched, _ = self._resolve_ffmpeg_state()
        if not ok:
            selected = self.ffmpeg_var.get().strip()
            reason = "前回指定された ffmpeg パスが見つかりませんでした。" if selected else None
            if not self._show_ffmpeg_missing_dialog(searched, reason=reason):
                return

        self.open_report_btn.configure(state="disabled")
        self._save_settings()
        self._set_running(True)
        self._enqueue_log("=== 実行開始 ===")
        self.run_logger.info(
            "process-folder requested input=%s output=%s model=%s ffmpeg=%s",
            self.input_var.get().strip(),
            self.output_var.get().strip(),
            self.model_var.get().strip(),
            self.ffmpeg_var.get().strip(),
        )

        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def _worker(self) -> None:
        try:
            result = process_folder_api(
                folder=self.input_var.get().strip(),
                model_path=self.model_var.get().strip() or None,
                out_dir=self.output_var.get().strip() or None,
                config_path=str(self.config_path),
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
            tb = traceback.format_exc()
            self._enqueue_log(tb)
            self.run_logger.error(tb)
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

    def _open_logs_folder(self) -> None:
        target = logs_dir()
        try:
            if os.name == "nt":
                os.startfile(str(target))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", str(target)], check=False)
            else:
                subprocess.run(["xdg-open", str(target)], check=False)
        except Exception as exc:
            messagebox.showerror("エラー", f"ログフォルダを開けませんでした: {exc}")

    def _open_diagnostics_dialog(self) -> None:
        diag = collect_diagnostics(
            settings_path=self.settings_path,
            model_path=self.model_var.get().strip() or None,
            ffmpeg_path=self.ffmpeg_var.get().strip() or None,
        )

        dlg = Toplevel(self.root)
        dlg.title("診断")
        dlg.geometry("780x480")

        text = ScrolledText(dlg, wrap="word", height=20)
        text.pack(fill=BOTH, expand=True, padx=12, pady=12)
        lines = [f"{k}: {diag[k]}" for k in sorted(diag.keys())]
        text.insert("1.0", "\n".join(lines))
        text.configure(state="disabled")

        bar = Frame(dlg)
        bar.pack(fill="x", padx=12, pady=(0, 12))

        def do_export() -> None:
            out_zip = export_diagnostic_zip(diag=diag, settings_path=self.settings_path)
            self.startup_logger.info("diagnostics zip exported: %s", out_zip)
            messagebox.showinfo("診断", f"診断レポートZIPを出力しました:\n{out_zip}")

        Button(bar, text="診断レポートzipを出力", command=do_export).pack(side=LEFT)


def main() -> None:
    root = Tk()
    RadioCmCutterGui(root)
    root.mainloop()
