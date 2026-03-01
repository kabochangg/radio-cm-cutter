from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, X, Button, Canvas, Entry, Frame, Label, StringVar, Toplevel, filedialog, messagebox
from tkinter import ttk

from radio_cm_cutter.api import (
    cut_one_api,
    detect_one_api,
    labels_from_segments,
    save_labels_csv,
    train_evaluate_and_update_model_api,
)
from radio_cm_cutter_app.player import FFPlayPlayer
from radio_cm_cutter_app.state import load_config, save_config


class SegmentReviewDialog:
    CANVAS_W = 860
    CANVAS_H = 90

    def __init__(self, owner, audio_path: Path, model_path: Path | None, output_dir: Path, config_path: Path, ffmpeg_path: str | None) -> None:
        self.owner = owner
        self.audio_path = audio_path
        self.model_path = model_path
        self.output_dir = output_dir
        self.config_path = config_path
        self.ffmpeg_path = ffmpeg_path
        self.duration_sec = 0.0
        self.mode = ""
        self.segments: list[tuple[float, float]] = []
        self.selected_index: int | None = None
        self.drag_start_x: float | None = None
        self.drag_current_x: float | None = None
        self.player = FFPlayPlayer(audio_path)

        cfg = load_config(config_path)
        self.start_th_var = StringVar(value=str(cfg.get("ml_start_prob", 0.55)))
        self.end_th_var = StringVar(value=str(cfg.get("ml_end_prob", 0.50)))
        self.min_len_var = StringVar(value=str(cfg.get("min_cm_sec", 8.0)))
        self.merge_gap_var = StringVar(value=str(cfg.get("merge_gap_sec", 4.0)))
        self.pad_pre_var = StringVar(value=str(cfg.get("pad_pre_sec", cfg.get("pad_sec", 0.2))))
        self.pad_post_var = StringVar(value=str(cfg.get("pad_post_sec", cfg.get("pad_sec", 0.2))))

        self.start_var = StringVar(value="0.0")
        self.end_var = StringVar(value="0.0")

        self.win = Toplevel(owner.root)
        self.win.title("レビュー＆ラベリング（1ファイル）")
        self.win.geometry("980x780")
        self.win.transient(owner.root)
        self.win.grab_set()
        self.win.protocol("WM_DELETE_WINDOW", self._close)
        self._build()
        self._detect_segments()

    def _build(self) -> None:
        Label(self.win, text=f"入力MP3: {self.audio_path}", justify="left", anchor="w").pack(fill=X, padx=12, pady=(12, 6))
        params = Frame(self.win)
        params.pack(fill=X, padx=12, pady=(0, 6))
        for txt, var in [
            ("threshold(start)", self.start_th_var),
            ("threshold(end)", self.end_th_var),
            ("min_len", self.min_len_var),
            ("merge_gap", self.merge_gap_var),
            ("pad_pre", self.pad_pre_var),
            ("pad_post", self.pad_post_var),
        ]:
            Label(params, text=txt).pack(side=LEFT)
            Entry(params, width=6, textvariable=var).pack(side=LEFT, padx=(2, 8))

        action = Frame(self.win)
        action.pack(fill=X, padx=12, pady=(0, 8))
        Button(action, text="このファイルで再検出プレビュー", command=self._redetect_preview).pack(side=LEFT)
        Button(action, text="選択区間を再生", command=self._play_selected).pack(side=LEFT, padx=6)
        Button(action, text="停止", command=self.player.stop).pack(side=LEFT, padx=6)
        Button(action, text="確定（保存→train→evaluate→更新）", command=self._confirm).pack(side=RIGHT)
        Button(action, text="キャンセル", command=self._close).pack(side=RIGHT, padx=6)

        self.timeline = Canvas(self.win, width=self.CANVAS_W, height=self.CANVAS_H, bg="#f3f3f3", highlightthickness=1, highlightbackground="#bdbdbd")
        self.timeline.pack(fill=X, padx=12, pady=8)
        self.timeline.bind("<ButtonPress-1>", self._on_drag_start)
        self.timeline.bind("<B1-Motion>", self._on_drag_move)
        self.timeline.bind("<ButtonRelease-1>", self._on_drag_release)

        list_frame = Frame(self.win)
        list_frame.pack(fill=BOTH, expand=True, padx=12, pady=6)
        self.segment_table = ttk.Treeview(list_frame, columns=("start", "end", "dur"), show="headings", height=12)
        for c in ("start", "end", "dur"):
            self.segment_table.heading(c, text=c)
            self.segment_table.column(c, width=120, anchor="w")
        self.segment_table.pack(side=LEFT, fill=BOTH, expand=True)
        self.segment_table.bind("<<TreeviewSelect>>", self._on_select_segment)

        edit = Frame(self.win)
        edit.pack(fill=X, padx=12, pady=(4, 12))
        Label(edit, text="start(sec)").pack(side=LEFT)
        Entry(edit, width=10, textvariable=self.start_var).pack(side=LEFT, padx=(4, 10))
        Label(edit, text="end(sec)").pack(side=LEFT)
        Entry(edit, width=10, textvariable=self.end_var).pack(side=LEFT, padx=(4, 10))
        Button(edit, text="適用", command=self._apply_edit).pack(side=LEFT)
        Button(edit, text="ドラッグ範囲を追加", command=self._add_drag_selection).pack(side=LEFT, padx=8)

    def _updated_config(self) -> dict:
        cfg = load_config(self.config_path)
        cfg["ml_start_prob"] = float(self.start_th_var.get())
        cfg["ml_end_prob"] = float(self.end_th_var.get())
        cfg["min_cm_sec"] = float(self.min_len_var.get())
        cfg["merge_gap_sec"] = float(self.merge_gap_var.get())
        pre = float(self.pad_pre_var.get())
        post = float(self.pad_post_var.get())
        cfg["pad_pre_sec"] = pre
        cfg["pad_post_sec"] = post
        cfg["pad_sec"] = (pre + post) / 2.0
        save_config(self.config_path, cfg)
        return cfg

    def _redetect_preview(self) -> None:
        self._updated_config()
        self._detect_segments()

    def _detect_segments(self) -> None:
        segments, duration, mode = detect_one_api(str(self.audio_path), str(self.model_path) if self.model_path else None, str(self.config_path), self.ffmpeg_path)
        self.duration_sec = duration
        self.mode = mode
        self.segments = [(round(s, 1), round(e, 1)) for s, e in segments if e > s]
        self.selected_index = 0 if self.segments else None
        self._refresh_table()
        self._draw_timeline()

    def _confirm(self) -> None:
        self._updated_config()
        target_dir = filedialog.askdirectory(title="カットMP3の保存先", initialdir=str(self.output_dir), parent=self.win)
        if not target_dir:
            return
        base = self.audio_path.stem
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path(target_dir).expanduser().resolve() / f"review_{ts}"
        out_root.mkdir(parents=True, exist_ok=True)
        out_mp3 = out_root / f"{base}_cut.mp3"
        seg_pairs = [(float(s), float(e)) for s, e in self.segments]
        cut_path = cut_one_api(str(self.audio_path), seg_pairs, str(out_mp3), str(self.config_path), self.ffmpeg_path)

        label_rows = labels_from_segments(str(self.audio_path), seg_pairs, self.duration_sec)
        user_csv = Path("data/labels_user") / f"{base}.csv"
        save_labels_csv(label_rows, user_csv)
        save_labels_csv(label_rows, out_root / f"{base}.labels.csv")

        eval_dir = Path("data/labels_eval")
        eval_patterns = [str(eval_dir / "*.csv")] if eval_dir.exists() else [str(user_csv)]
        train_result = train_evaluate_and_update_model_api(
            labels_patterns=["data/labels_user/*.csv"],
            config_path=str(self.config_path),
            model_path="model/model.pkl",
            eval_patterns=eval_patterns,
            log_callback=lambda m: self.owner._enqueue_log("run", f"[train] {m}"),
        )

        eval_info = train_result.get("eval")
        old_eval = train_result.get("old_eval")
        delta = train_result.get("delta")
        msg = [f"カット保存: {cut_path}", f"教師データ保存: {user_csv}"]
        if train_result.get("backup_path"):
            msg.append(f"モデルバックアップ: {train_result['backup_path']}")
        if eval_info:
            msg.append(f"new precision={eval_info.precision:.4f} recall={eval_info.recall:.4f} f1={eval_info.f1:.4f}")
            if old_eval and delta:
                msg.append(f"差分 ΔP={delta['precision']:+.4f} ΔR={delta['recall']:+.4f} ΔF1={delta['f1']:+.4f}")
            elif not Path("data/labels_eval").exists():
                msg.append("※ data/labels_eval が未作成のため今回追加分で簡易評価しました。")
        self._save_session(out_root)
        messagebox.showinfo("完了", "\n".join(msg), parent=self.win)
        self._close()

    def _save_session(self, out_root: Path) -> None:
        session_dir = out_root / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "audio_path": str(self.audio_path),
            "segments": [{"start": round(s, 3), "end": round(e, 3)} for s, e in self.segments],
            "created_at": datetime.now().isoformat(),
            "detect_mode": self.mode,
        }
        (session_dir / f"{self.audio_path.stem}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _refresh_table(self) -> None:
        for iid in self.segment_table.get_children():
            self.segment_table.delete(iid)
        for i, (s, e) in enumerate(self.segments):
            self.segment_table.insert("", END, iid=str(i), values=(f"{s:.1f}", f"{e:.1f}", f"{(e-s):.1f}"))

    def _draw_timeline(self) -> None:
        self.timeline.delete("all")
        self.timeline.create_rectangle(10, 30, self.CANVAS_W - 10, 60, fill="#ddd", outline="#aaa")
        for idx, (s, e) in enumerate(self.segments):
            x1 = self._sec_to_x(s)
            x2 = self._sec_to_x(e)
            color = "#f39c12" if idx == self.selected_index else "#3498db"
            self.timeline.create_rectangle(x1, 30, x2, 60, fill=color, outline="")

    def _on_select_segment(self, _e) -> None:
        cur = self.segment_table.selection()
        if not cur:
            return
        self.selected_index = int(cur[0])
        s, e = self.segments[self.selected_index]
        self.start_var.set(f"{s:.1f}")
        self.end_var.set(f"{e:.1f}")
        self._draw_timeline()

    def _on_drag_start(self, event) -> None:
        self.drag_start_x = min(max(10, event.x), self.CANVAS_W - 10)

    def _on_drag_move(self, event) -> None:
        self.drag_current_x = min(max(10, event.x), self.CANVAS_W - 10)

    def _on_drag_release(self, event) -> None:
        self.drag_current_x = min(max(10, event.x), self.CANVAS_W - 10)
        if self.drag_start_x is None:
            return
        start = self._x_to_sec(min(self.drag_start_x, self.drag_current_x))
        end = self._x_to_sec(max(self.drag_start_x, self.drag_current_x))
        self.start_var.set(f"{start:.1f}")
        self.end_var.set(f"{end:.1f}")

    def _add_drag_selection(self) -> None:
        s, e = float(self.start_var.get()), float(self.end_var.get())
        if e <= s:
            return
        self.segments.append((round(s, 1), round(e, 1)))
        self.segments.sort(key=lambda x: x[0])
        self._refresh_table()
        self._draw_timeline()

    def _apply_edit(self) -> None:
        if self.selected_index is None:
            return
        s, e = float(self.start_var.get()), float(self.end_var.get())
        if e <= s:
            return
        self.segments[self.selected_index] = (round(s, 1), round(e, 1))
        self._refresh_table()
        self._draw_timeline()

    def _play_selected(self) -> None:
        if self.selected_index is None or not self.segments:
            return
        s, e = self.segments[self.selected_index]
        self.player.play_range(s, e)

    def _close(self) -> None:
        self.player.stop()
        self.win.grab_release()
        self.win.destroy()

    def _sec_to_x(self, sec: float) -> float:
        ratio = 0.0 if self.duration_sec <= 0 else sec / self.duration_sec
        return 10 + (self.CANVAS_W - 20) * ratio

    def _x_to_sec(self, x: float) -> float:
        ratio = (x - 10) / (self.CANVAS_W - 20)
        return max(0.0, min(self.duration_sec, ratio * self.duration_sec))
