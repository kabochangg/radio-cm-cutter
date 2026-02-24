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
from tkinter import BOTH, END, LEFT, RIGHT, W, BooleanVar, Button, Checkbutton, Entry, Frame, Label, Radiobutton, StringVar, Tk, Toplevel, filedialog, messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from radio_cm_cutter.api import default_output_dir, evaluate_api, process_folder_api, train_api
from radio_cm_cutter_app.diagnostics import collect_diagnostics, export_diagnostic_zip
from radio_cm_cutter_app.support import (
    APP_NAME,
    FFMPEG_ARCHIVE_NAME,
    FFMPEG_DOWNLOAD_URL,
    bundled_resource_path,
    config_dir,
    ffmpeg_dir,
    logs_dir,
)
SETTINGS_FILE = "gui_settings.json"
MODE_DETECT_ONLY = "detect_only"
MODE_DETECT_AND_CUT = "detect_and_cut"


class RadioCmCutterGui:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Radio CM Cutter (MVP)")
        self.root.geometry("980x700")

        self.settings_path = self._settings_path()
        self.settings = self._load_settings()
        self.config_path = bundled_resource_path("config.json")

        self.input_var = StringVar(value=self.settings.get("input_folder", ""))
        self.model_var = StringVar(value=self.settings.get("model_path", self._default_model_path()))
        self.output_var = StringVar(value=self.settings.get("output_folder", str(default_output_dir())))
        self.ffmpeg_var = StringVar(value=self.settings.get("ffmpeg_path", ""))
        self.mode_var = StringVar(value=self.settings.get("run_mode", MODE_DETECT_ONLY))
        self.auto_open_report_var = BooleanVar(value=bool(self.settings.get("auto_open_report", False)))

        self.train_labels_var = StringVar(value=self.settings.get("train_labels", ""))
        self.train_out_var = StringVar(value=self.settings.get("train_out", "model/model.pkl"))
        self.eval_labels_var = StringVar(value=self.settings.get("eval_labels", ""))
        self.eval_model_var = StringVar(value=self.settings.get("eval_model", self._default_model_path()))
        self.metrics_var = StringVar(value="precision: -   recall: -   f1: -")

        self.log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
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
            "run_mode": self.mode_var.get().strip() or MODE_DETECT_ONLY,
            "auto_open_report": bool(self.auto_open_report_var.get()),
            "train_labels": self.train_labels_var.get().strip(),
            "train_out": self.train_out_var.get().strip(),
            "eval_labels": self.eval_labels_var.get().strip(),
            "eval_model": self.eval_model_var.get().strip(),
        }
        self.settings_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _default_model_path(self) -> str:
        return str(bundled_resource_path("model", "model.pkl"))

    def _build(self) -> None:
        note = Label(
            self.root,
            text=(
                "labels = 音声のどの区間がCM/番組かを記録したCSVです。\n"
                "学習(Train): labelsからモデルを作成します。評価(Evaluate): 別データで精度を確認します。\n"
                "推奨: train用とevaluate用でデータを分割し、同じデータを使い回さないでください。"
            ),
            justify="left",
            anchor="w",
        )
        note.pack(fill="x", padx=12, pady=(10, 4))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=BOTH, expand=True, padx=12, pady=6)

        self.tab_run = Frame(self.notebook)
        self.tab_train = Frame(self.notebook)
        self.tab_eval = Frame(self.notebook)
        self.notebook.add(self.tab_run, text="検出/カット")
        self.notebook.add(self.tab_train, text="学習 (Train)")
        self.notebook.add(self.tab_eval, text="評価 (Evaluate)")

        self._build_run_tab()
        self._build_train_tab()
        self._build_eval_tab()

    def _build_run_tab(self) -> None:
        self._row(self.tab_run, "入力フォルダ", self.input_var, self._pick_input)
        self._row(self.tab_run, "モデル (.pkl)", self.model_var, self._pick_model)
        self._row(self.tab_run, "出力フォルダ", self.output_var, self._pick_output)
        self._row(self.tab_run, "FFmpegパス(任意)", self.ffmpeg_var, self._pick_ffmpeg)

        mode_row = Frame(self.tab_run)
        mode_row.pack(fill="x", padx=12, pady=4)
        Label(mode_row, text="実行モード", width=18, anchor=W).pack(side=LEFT)
        Radiobutton(mode_row, text="検出だけ", variable=self.mode_var, value=MODE_DETECT_ONLY).pack(side=LEFT, padx=(0, 12))
        Radiobutton(mode_row, text="検出してカット", variable=self.mode_var, value=MODE_DETECT_AND_CUT).pack(side=LEFT)

        option_row = Frame(self.tab_run)
        option_row.pack(fill="x", padx=12, pady=(0, 4))
        Checkbutton(option_row, text="完了後にレポートを自動で開く", variable=self.auto_open_report_var, command=self._save_settings).pack(side=LEFT)

        action = Frame(self.tab_run)
        action.pack(fill="x", padx=12, pady=8)
        self.run_btn = Button(action, text="実行", width=14, command=self._start)
        self.run_btn.pack(side=LEFT)
        self.open_report_btn = Button(action, text="ml_report.html を開く", state="disabled", command=self._open_report)
        self.open_report_btn.pack(side=LEFT, padx=8)
        self.diag_btn = Button(action, text="診断", command=self._open_diagnostics_dialog)
        self.diag_btn.pack(side=LEFT, padx=8)
        self.open_logs_btn = Button(action, text="ログを開く", command=self._open_logs_folder)
        self.open_logs_btn.pack(side=LEFT, padx=8)

        self.run_log_text = ScrolledText(self.tab_run, wrap="word", height=16)
        self.run_log_text.pack(fill=BOTH, expand=True, padx=12, pady=8)

    def _build_train_tab(self) -> None:
        self._row(self.tab_train, "labels入力", self.train_labels_var, self._pick_train_labels_files)
        row = Frame(self.tab_train)
        row.pack(fill="x", padx=12, pady=(0, 6))
        Button(row, text="labelsフォルダ選択", command=self._pick_train_labels_folder).pack(side=LEFT)
        Label(row, text="(複数CSVは ; 区切り)").pack(side=LEFT, padx=8)
        self._row(self.tab_train, "model.pkl保存先", self.train_out_var, self._pick_train_out)

        action = Frame(self.tab_train)
        action.pack(fill="x", padx=12, pady=8)
        self.train_btn = Button(action, text="学習を実行", width=14, command=self._start_train)
        self.train_btn.pack(side=LEFT)

        self.train_log_text = ScrolledText(self.tab_train, wrap="word", height=20)
        self.train_log_text.pack(fill=BOTH, expand=True, padx=12, pady=8)

    def _build_eval_tab(self) -> None:
        self._row(self.tab_eval, "labels入力", self.eval_labels_var, self._pick_eval_labels_files)
        row = Frame(self.tab_eval)
        row.pack(fill="x", padx=12, pady=(0, 6))
        Button(row, text="labelsフォルダ選択", command=self._pick_eval_labels_folder).pack(side=LEFT)
        Label(row, text="(複数CSVは ; 区切り)").pack(side=LEFT, padx=8)

        self._row(self.tab_eval, "モデル (.pkl)", self.eval_model_var, self._pick_eval_model)

        action = Frame(self.tab_eval)
        action.pack(fill="x", padx=12, pady=8)
        self.eval_btn = Button(action, text="評価を実行", width=14, command=self._start_eval)
        self.eval_btn.pack(side=LEFT)
        Button(action, text="結果を保存", command=self._save_metrics).pack(side=LEFT, padx=8)

        Label(self.tab_eval, textvariable=self.metrics_var, font=("TkDefaultFont", 11, "bold")).pack(fill="x", padx=12, pady=4)
        self.eval_log_text = ScrolledText(self.tab_eval, wrap="word", height=18)
        self.eval_log_text.pack(fill=BOTH, expand=True, padx=12, pady=8)

    def _row(self, parent: Frame, label: str, var: StringVar, picker) -> None:
        row = Frame(parent)
        row.pack(fill="x", padx=12, pady=6)
        Label(row, text=label, width=18, anchor=W).pack(side=LEFT)
        Entry(row, textvariable=var).pack(side=LEFT, fill="x", expand=True, padx=8)
        Button(row, text="参照...", width=10, command=picker).pack(side=RIGHT)

    def _parse_labels(self, raw: str) -> list[str]:
        items = [x.strip() for x in raw.split(";") if x.strip()]
        out: list[str] = []
        for item in items:
            p = Path(item).expanduser()
            if p.is_dir():
                out.append(str((p / "*.csv")))
            else:
                out.append(str(p))
        return out

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

    def _pick_train_labels_files(self) -> None:
        paths = filedialog.askopenfilenames(title="labels CSVを選択", filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if paths:
            self.train_labels_var.set(";".join(paths))

    def _pick_train_labels_folder(self) -> None:
        p = filedialog.askdirectory(title="labelsフォルダを選択")
        if p:
            self.train_labels_var.set(p)

    def _pick_train_out(self) -> None:
        p = filedialog.asksaveasfilename(title="model.pkl の保存先", defaultextension=".pkl", filetypes=[("Pickle", "*.pkl")], initialfile="model.pkl")
        if p:
            self.train_out_var.set(p)

    def _pick_eval_labels_files(self) -> None:
        paths = filedialog.askopenfilenames(title="labels CSVを選択", filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if paths:
            self.eval_labels_var.set(";".join(paths))

    def _pick_eval_labels_folder(self) -> None:
        p = filedialog.askdirectory(title="labelsフォルダを選択")
        if p:
            self.eval_labels_var.set(p)

    def _pick_eval_model(self) -> None:
        p = filedialog.askopenfilename(title="model.pkl を選択", filetypes=[("Pickle", "*.pkl"), ("All", "*.*")])
        if p:
            self.eval_model_var.set(p)

    # ffmpeg and diagnostics methods are unchanged from previous behavior
    def _ffmpeg_search_candidates(self) -> list[Path]:
        selected = self.ffmpeg_var.get().strip()
        candidates: list[Path] = []
        if selected:
            sel = Path(selected).expanduser()
            candidates.extend([sel, sel.parent / "ffprobe.exe"] if sel.suffix.lower() == ".exe" else [sel / "ffmpeg.exe", sel / "ffprobe.exe", sel / "bin" / "ffmpeg.exe", sel / "bin" / "ffprobe.exe"])
        install_dir = ffmpeg_dir()
        candidates.extend([install_dir / "ffmpeg.exe", install_dir / "ffprobe.exe", install_dir / "bin" / "ffmpeg.exe", install_dir / "bin" / "ffprobe.exe"])
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
        description = "ffmpeg / ffprobe が見つからないため実行できません。\n自動で探した場所と、次にやることを確認してください。"
        if reason:
            description = f"{reason}\n\n{description}"
        Label(dlg, text=description, justify="left", anchor="w").pack(fill="x", padx=12, pady=(12, 6))
        Label(dlg, text="次にやること:\n1) ffmpeg.exe を指定\n2) READMEで導入手順を確認\n3) 任意で自動ダウンロード", justify="left", anchor="w").pack(fill="x", padx=12, pady=(0, 8))
        text = ScrolledText(dlg, wrap="word", height=16)
        text.pack(fill=BOTH, expand=True, padx=12, pady=8)
        text.insert("1.0", "自動で探した場所:\n- " + "\n- ".join(searched))
        text.configure(state="disabled")
        decision = {"ok": False}

        def pick() -> None:
            p = filedialog.askopenfilename(title="ffmpeg.exe を選択", filetypes=[("FFmpeg executable", "ffmpeg.exe"), ("Executable", "*.exe"), ("All", "*.*")])
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
            self._enqueue_log("run", "FFmpeg ダウンロードを開始します...")
            ok, msg = self._download_ffmpeg()
            self._enqueue_log("run", msg)
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

    def _append_log(self, target: str, message: str) -> None:
        box = self.run_log_text if target == "run" else self.train_log_text if target == "train" else self.eval_log_text
        box.insert(END, message + "\n")
        box.see(END)

    def _enqueue_log(self, target: str, message: str) -> None:
        self.run_logger.info("[%s] %s", target, message)
        self.log_queue.put((target, message))

    def _poll_log_queue(self) -> None:
        try:
            while True:
                target, msg = self.log_queue.get_nowait()
                self._append_log(target, msg)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_log_queue)

    def _set_running(self, running: bool) -> None:
        self.running = running
        st = "disabled" if running else "normal"
        self.run_btn.configure(state=st)
        self.train_btn.configure(state=st)
        self.eval_btn.configure(state=st)

    def _start(self) -> None:
        if self.running:
            return
        if not self.input_var.get().strip():
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
        self._enqueue_log("run", "=== 実行開始 ===")
        threading.Thread(target=self._worker_run, daemon=True).start()

    def _worker_run(self) -> None:
        try:
            mode = self.mode_var.get().strip() or MODE_DETECT_ONLY
            detect_result = process_folder_api(
                folder=self.input_var.get().strip(),
                model_path=self.model_var.get().strip() or None,
                out_dir=self.output_var.get().strip() or None,
                config_path=str(self.config_path),
                ffmpeg_path=self.ffmpeg_var.get().strip() or None,
                recursive=True,
                detect_only=True,
                log_callback=lambda m: self._enqueue_log("run", m),
            )
            self.last_output_dir = detect_result.output_dir
            self._maybe_enable_report_button(detect_result.output_dir)
            if mode == MODE_DETECT_ONLY:
                self._enqueue_log("run", f"完了(検出のみ): success={detect_result.success}, failed={detect_result.failed}, segments={detect_result.total_segments}, total_cm_sec={detect_result.total_cm_sec:.1f}")
                self._auto_open_report_if_enabled(detect_result.output_dir)
                return
            if not self._confirm_cut(detect_result.total_segments, detect_result.total_cm_sec):
                self._enqueue_log("run", "カットはキャンセルされました（検出結果のみ保存）。")
                self._auto_open_report_if_enabled(detect_result.output_dir)
                return
            cut_result = process_folder_api(
                folder=self.input_var.get().strip(),
                model_path=self.model_var.get().strip() or None,
                out_dir=self.output_var.get().strip() or None,
                config_path=str(self.config_path),
                ffmpeg_path=self.ffmpeg_var.get().strip() or None,
                recursive=True,
                detect_only=False,
                log_callback=lambda m: self._enqueue_log("run", m),
            )
            self.last_output_dir = cut_result.output_dir
            self._maybe_enable_report_button(cut_result.output_dir)
            self._enqueue_log("run", f"完了(検出+カット): success={cut_result.success}, failed={cut_result.failed}, output={cut_result.output_dir}")
            self._auto_open_report_if_enabled(cut_result.output_dir)
        except Exception as exc:
            self._handle_error(exc, "run")
        finally:
            self.root.after(0, lambda: self._set_running(False))

    def _start_train(self) -> None:
        if self.running:
            return
        labels = self._parse_labels(self.train_labels_var.get())
        if not labels:
            messagebox.showwarning("入力不足", "labelsを指定してください（フォルダ or CSV）。")
            return
        if not self.train_out_var.get().strip():
            messagebox.showwarning("入力不足", "model.pkl の保存先を指定してください。")
            return
        self._save_settings()
        self._set_running(True)
        self._enqueue_log("train", "=== 学習開始 ===")
        threading.Thread(target=self._worker_train, daemon=True).start()

    def _worker_train(self) -> None:
        try:
            out = train_api(
                labels=self._parse_labels(self.train_labels_var.get()),
                out_path=self.train_out_var.get().strip(),
                config_path=str(self.config_path),
                log_callback=lambda m: self._enqueue_log("train", m),
            )
            self._enqueue_log("train", f"学習完了: {out}")
            self.root.after(0, lambda: messagebox.showinfo("学習完了", f"model.pkl を生成しました:\n{out}"))
        except Exception as exc:
            self._handle_error(exc, "train")
        finally:
            self.root.after(0, lambda: self._set_running(False))

    def _start_eval(self) -> None:
        if self.running:
            return
        labels = self._parse_labels(self.eval_labels_var.get())
        if not labels:
            messagebox.showwarning("入力不足", "labelsを指定してください（フォルダ or CSV）。")
            return
        if not self.eval_model_var.get().strip():
            messagebox.showwarning("入力不足", "model.pkl を選択してください。")
            return
        self._save_settings()
        self._set_running(True)
        self._enqueue_log("eval", "=== 評価開始 ===")
        threading.Thread(target=self._worker_eval, daemon=True).start()

    def _worker_eval(self) -> None:
        try:
            result = evaluate_api(
                labels=self._parse_labels(self.eval_labels_var.get()),
                model_path=self.eval_model_var.get().strip(),
                config_path=str(self.config_path),
                log_callback=lambda m: self._enqueue_log("eval", m),
            )
            self.metrics_var.set(f"precision: {result.precision:.4f}   recall: {result.recall:.4f}   f1: {result.f1:.4f}   samples: {result.samples}")
            self._enqueue_log("eval", "評価完了")
        except Exception as exc:
            self._handle_error(exc, "eval")
        finally:
            self.root.after(0, lambda: self._set_running(False))

    def _save_metrics(self) -> None:
        raw = self.metrics_var.get()
        if "-" in raw:
            messagebox.showinfo("情報", "先に評価を実行してください。")
            return
        out = filedialog.asksaveasfilename(title="metrics保存", defaultextension=".json", filetypes=[("JSON", "*.json"), ("Text", "*.txt")], initialfile="metrics.json")
        if not out:
            return
        p = Path(out)
        if p.suffix.lower() == ".txt":
            p.write_text(raw + "\n", encoding="utf-8")
        else:
            parts = raw.replace("samples:", "samples=").split()
            metrics = {
                "precision": float(parts[1]),
                "recall": float(parts[3]),
                "f1": float(parts[5]),
                "samples": int(parts[7]),
            }
            p.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        messagebox.showinfo("保存完了", f"保存しました:\n{p}")

    def _handle_error(self, exc: Exception, target: str) -> None:
        msg = self._friendly_error(exc)
        self._enqueue_log(target, msg)
        tb = traceback.format_exc()
        self._enqueue_log(target, tb)
        self.run_logger.error(tb)
        self.root.after(0, lambda: messagebox.showerror("実行エラー", msg))

    def _confirm_cut(self, total_segments: int, total_cm_sec: float) -> bool:
        msg = f"カット前に検出結果を確認してください。\n\n検出箇所: {total_segments} 箇所\n合計CM秒数: {total_cm_sec:.1f} 秒\n\nこの内容でカット処理を実行しますか？"
        return bool(self._run_on_ui_thread(lambda: messagebox.askyesno("カット前確認", msg)))

    def _run_on_ui_thread(self, func):
        done = threading.Event()
        result: dict[str, object] = {}

        def _invoke() -> None:
            try:
                result["value"] = func()
            finally:
                done.set()

        self.root.after(0, _invoke)
        done.wait()
        return result.get("value")

    def _maybe_enable_report_button(self, output_dir: Path) -> None:
        report_path = output_dir / "ml_report.html"
        if report_path.exists():
            self.root.after(0, lambda: self.open_report_btn.configure(state="normal"))

    def _auto_open_report_if_enabled(self, output_dir: Path) -> None:
        if not self.auto_open_report_var.get():
            return
        self.last_output_dir = output_dir
        self.root.after(0, self._open_report)

    def _friendly_error(self, exc: Exception) -> str:
        raw = str(exc)
        lower = raw.lower()
        if "required command not found" in lower or "ffmpeg" in lower or "ffprobe" in lower:
            return "原因: ffmpeg / ffprobe が見つかりません。\n次の行動: FFmpeg を導入し、ffmpeg.exe を選択するかPATHを設定してください。"
        if isinstance(exc, FileNotFoundError):
            return f"原因: ファイルまたはフォルダが見つかりません ({raw})\n次の行動: labels/model/入力フォルダのパスを確認してください。"
        if isinstance(exc, PermissionError):
            return f"原因: 権限エラーです ({raw})\n次の行動: 書き込み可能な保存先を選択してください。"
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
        diag = collect_diagnostics(settings_path=self.settings_path, model_path=self.model_var.get().strip() or None, ffmpeg_path=self.ffmpeg_var.get().strip() or None)
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
