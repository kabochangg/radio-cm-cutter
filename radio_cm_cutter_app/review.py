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
    TIMELINE_LEFT = 10
    TIMELINE_RIGHT = CANVAS_W - 10
    BAR_TOP = 30
    BAR_BOTTOM = 60
    HANDLE_HIT_PX = 8

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
        self.drag_mode: str | None = None
        self.drag_segment_index: int | None = None
        self.drag_shift_pressed = False
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
        Button(action, text="Start付近を再生(-2s〜+2s)", command=self._play_start_preview).pack(side=LEFT, padx=6)
        Button(action, text="End付近を再生(-2s〜+2s)", command=self._play_end_preview).pack(side=LEFT, padx=6)
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
        self._update_entry_from_selection()
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
        if self.selected_index is not None and 0 <= self.selected_index < len(self.segments):
            self.segment_table.selection_set(str(self.selected_index))
            self.segment_table.see(str(self.selected_index))

    def _draw_timeline(self) -> None:
        self.timeline.delete("all")
        self.timeline.create_rectangle(self.TIMELINE_LEFT, self.BAR_TOP, self.TIMELINE_RIGHT, self.BAR_BOTTOM, fill="#ddd", outline="#aaa")
        for idx, (s, e) in enumerate(self.segments):
            x1 = self._sec_to_x(s)
            x2 = self._sec_to_x(e)
            is_selected = idx == self.selected_index
            color = "#f39c12" if is_selected else "#3498db"
            outline = "#d35400" if is_selected else ""
            self.timeline.create_rectangle(x1, self.BAR_TOP, x2, self.BAR_BOTTOM, fill=color, outline=outline, width=2 if is_selected else 1)
            handle_color = "#e74c3c" if is_selected else "#1f618d"
            self.timeline.create_rectangle(x1 - 3, self.BAR_TOP, x1 + 3, self.BAR_BOTTOM, fill=handle_color, outline="")
            self.timeline.create_rectangle(x2 - 3, self.BAR_TOP, x2 + 3, self.BAR_BOTTOM, fill=handle_color, outline="")

        if self.drag_mode == "new" and self.drag_start_x is not None and self.drag_current_x is not None:
            x1 = min(self.drag_start_x, self.drag_current_x)
            x2 = max(self.drag_start_x, self.drag_current_x)
            self.timeline.create_rectangle(x1, self.BAR_TOP, x2, self.BAR_BOTTOM, fill="#9b59b6", stipple="gray50", outline="#8e44ad")

    def _on_select_segment(self, _e) -> None:
        cur = self.segment_table.selection()
        if not cur:
            return
        self._select_segment(int(cur[0]))

    def _on_drag_start(self, event) -> None:
        x = self._clamp_x(event.x)
        self.drag_start_x = x
        self.drag_current_x = x
        self.drag_shift_pressed = self._is_shift_pressed(event)
        hit = self._hit_test_segment(x, event.y)
        if hit is None:
            self.drag_mode = "new"
            self.drag_segment_index = None
            return

        idx, part = hit
        self.drag_segment_index = idx
        self._select_segment(idx)
        if part == "left":
            self.drag_mode = "move_start"
        elif part == "right":
            self.drag_mode = "move_end"
        else:
            self.drag_mode = "select"

    def _on_drag_move(self, event) -> None:
        if self.drag_mode is None:
            return
        self.drag_shift_pressed = self._is_shift_pressed(event)
        self.drag_current_x = self._clamp_x(event.x)

        if self.drag_mode == "new":
            self._draw_timeline()
            return
        if self.drag_mode not in {"move_start", "move_end"} or self.drag_segment_index is None:
            return

        idx = self.drag_segment_index
        s, e = self.segments[idx]
        dragged = self._snap_sec(self._x_to_sec(self.drag_current_x), self.drag_shift_pressed)
        min_len = self._min_len_sec()
        if self.drag_mode == "move_start":
            s = min(dragged, e - min_len)
        else:
            e = max(dragged, s + min_len)
        s, e = self._clamp_segment(s, e, min_len)
        self.segments[idx] = (s, e)
        self._refresh_table()
        self._update_entry_from_selection()
        self._draw_timeline()

    def _on_drag_release(self, event) -> None:
        self.drag_current_x = self._clamp_x(event.x)
        if self.drag_start_x is None or self.drag_mode is None:
            return

        if self.drag_mode == "new":
            start = self._snap_sec(self._x_to_sec(min(self.drag_start_x, self.drag_current_x)), self.drag_shift_pressed)
            end = self._snap_sec(self._x_to_sec(max(self.drag_start_x, self.drag_current_x)), self.drag_shift_pressed)
            start, end = self._clamp_segment(start, end, self._min_len_sec())
            self.start_var.set(f"{start:.1f}")
            self.end_var.set(f"{end:.1f}")
        elif self.drag_mode in {"move_start", "move_end"}:
            self._refresh_table()
            self._update_entry_from_selection()

        self.drag_mode = None
        self.drag_segment_index = None
        self.drag_start_x = None
        self.drag_current_x = None
        self._draw_timeline()

    def _add_drag_selection(self) -> None:
        s, e = self._safe_segment_from_entry()
        if s is None or e is None:
            return
        s, e = self._clamp_segment(s, e, self._min_len_sec())
        self.segments.append((s, e))
        self.segments.sort(key=lambda x: x[0])
        self.selected_index = self.segments.index((s, e))
        self._refresh_table()
        self._update_entry_from_selection()
        self._draw_timeline()

    def _apply_edit(self) -> None:
        if self.selected_index is None:
            return
        s, e = self._safe_segment_from_entry()
        if s is None or e is None:
            return
        self.segments[self.selected_index] = self._clamp_segment(s, e, self._min_len_sec())
        self._refresh_table()
        self._update_entry_from_selection()
        self._draw_timeline()

    def _play_selected(self) -> None:
        if self.selected_index is None or not self.segments:
            messagebox.showinfo("再生", "区間を選択してください", parent=self.win)
            return
        s, e = self.segments[self.selected_index]
        self.player.play_range(s, e)

    def _play_start_preview(self) -> None:
        if self.selected_index is None or not self.segments:
            messagebox.showinfo("再生", "区間を選択してください", parent=self.win)
            return
        s, _e = self.segments[self.selected_index]
        self.player.play_preview_around(s, radius=2.0)

    def _play_end_preview(self) -> None:
        if self.selected_index is None or not self.segments:
            messagebox.showinfo("再生", "区間を選択してください", parent=self.win)
            return
        _s, e = self.segments[self.selected_index]
        self.player.play_preview_around(e, radius=2.0)

    def _close(self) -> None:
        self.player.stop()
        self.win.grab_release()
        self.win.destroy()

    def _sec_to_x(self, sec: float) -> float:
        ratio = 0.0 if self.duration_sec <= 0 else sec / self.duration_sec
        return self.TIMELINE_LEFT + (self.CANVAS_W - 20) * ratio

    def _x_to_sec(self, x: float) -> float:
        ratio = (x - self.TIMELINE_LEFT) / (self.CANVAS_W - 20)
        return max(0.0, min(self.duration_sec, ratio * self.duration_sec))

    def _select_segment(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.segments):
            return
        self.selected_index = idx
        self._update_entry_from_selection()
        self._refresh_table()
        self._draw_timeline()

    def _update_entry_from_selection(self) -> None:
        if self.selected_index is None or self.selected_index >= len(self.segments):
            return
        s, e = self.segments[self.selected_index]
        self.start_var.set(f"{s:.1f}")
        self.end_var.set(f"{e:.1f}")

    def _clamp_x(self, x: float) -> float:
        return min(max(self.TIMELINE_LEFT, x), self.TIMELINE_RIGHT)

    def _hit_test_segment(self, x: float, y: float) -> tuple[int, str] | None:
        if y < self.BAR_TOP - self.HANDLE_HIT_PX or y > self.BAR_BOTTOM + self.HANDLE_HIT_PX:
            return None
        for idx in range(len(self.segments) - 1, -1, -1):
            s, e = self.segments[idx]
            x1 = self._sec_to_x(s)
            x2 = self._sec_to_x(e)
            if abs(x - x1) <= self.HANDLE_HIT_PX:
                return idx, "left"
            if abs(x - x2) <= self.HANDLE_HIT_PX:
                return idx, "right"
            if x1 <= x <= x2:
                return idx, "body"
        return None

    def _is_shift_pressed(self, event) -> bool:
        return bool(event.state & 0x0001)

    def _snap_sec(self, sec: float, shift_pressed: bool) -> float:
        step = 0.5 if shift_pressed else 0.1
        return round(sec / step) * step

    def _min_len_sec(self) -> float:
        try:
            return max(0.1, float(self.min_len_var.get()))
        except ValueError:
            return 0.1

    def _clamp_segment(self, start: float, end: float, min_len: float) -> tuple[float, float]:
        start = max(0.0, min(self.duration_sec, start))
        end = max(0.0, min(self.duration_sec, end))
        if end - start < min_len:
            if start + min_len <= self.duration_sec:
                end = start + min_len
            else:
                start = max(0.0, self.duration_sec - min_len)
                end = self.duration_sec
        start = max(0.0, min(self.duration_sec, start))
        end = max(0.0, min(self.duration_sec, end))
        snapped_start = self._snap_sec(start, False)
        snapped_end = self._snap_sec(end, False)
        if snapped_end - snapped_start < min_len:
            snapped_end = min(self.duration_sec, snapped_start + min_len)
            if snapped_end - snapped_start < min_len:
                snapped_start = max(0.0, snapped_end - min_len)
        return (round(snapped_start, 1), round(snapped_end, 1))

    def _safe_segment_from_entry(self) -> tuple[float | None, float | None]:
        try:
            return float(self.start_var.get()), float(self.end_var.get())
        except ValueError:
            messagebox.showwarning("入力エラー", "start/end は数値で入力してください", parent=self.win)
            return None, None
