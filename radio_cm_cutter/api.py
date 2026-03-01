"""Programmatic API for GUI and integrations."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import os
from datetime import datetime
import csv
import pickle
import shutil
from pathlib import Path
from typing import Callable

from .cli import (
    EvaluateResult,
    ProcessFolderResult,
    Segment,
    _detect_with_fallback,
    _process_folder_impl,
    _resolve_model_path,
    cmd_train,
    complement_segments,
    cut_mp3,
    decode_to_wav,
    evaluate_impl,
    ffprobe_duration,
    read_wav_pcm16,
)

LogCallback = Callable[[str], None]


class _LogWriter(io.TextIOBase):
    def __init__(self, fallback: io.TextIOBase, log_callback: LogCallback | None) -> None:
        self.fallback = fallback
        self.log_callback = log_callback
        self._buffer = ""

    def write(self, s: str) -> int:
        self.fallback.write(s)
        if self.log_callback is not None:
            self._buffer += s
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line:
                    self.log_callback(line)
        return len(s)

    def flush(self) -> None:
        self.fallback.flush()
        if self.log_callback is not None and self._buffer:
            self.log_callback(self._buffer)
            self._buffer = ""


def ensure_ffmpeg_on_path(ffmpeg_path: str | None = None) -> None:
    """Append user-selected FFmpeg location to PATH for this process."""
    if not ffmpeg_path:
        return

    p = Path(ffmpeg_path).expanduser()
    if p.is_file():
        p = p.parent

    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"ffmpeg path not found: {ffmpeg_path}")

    current = os.environ.get("PATH", "")
    entries = current.split(os.pathsep) if current else []
    p_str = str(p.resolve())
    if p_str not in entries:
        os.environ["PATH"] = p_str + (os.pathsep + current if current else "")


def default_output_dir() -> Path:
    return Path("cut_output") / datetime.now().strftime("%Y%m%d_%H%M%S")


def process_folder_api(
    folder: str,
    model_path: str | None = None,
    out_dir: str | None = None,
    config_path: str = "config.json",
    recursive: bool = True,
    overwrite: bool = False,
    keep_parts: bool = False,
    ffmpeg_path: str | None = None,
    detect_only: bool = False,
    log_callback: LogCallback | None = None,
) -> ProcessFolderResult:
    """Run folder processing via existing CLI logic and stream logs."""
    ensure_ffmpeg_on_path(ffmpeg_path)

    resolved_out = out_dir or str(default_output_dir())
    args = argparse.Namespace(
        config=config_path,
        folder=folder,
        out_dir=resolved_out,
        recursive=recursive,
        overwrite=overwrite,
        keep_parts=keep_parts,
        model=model_path,
        detect_only=detect_only,
    )

    out_writer = _LogWriter(sys.stdout, log_callback)
    err_writer = _LogWriter(sys.stderr, log_callback)
    with contextlib.redirect_stdout(out_writer), contextlib.redirect_stderr(err_writer):
        result = _process_folder_impl(args)
    out_writer.flush()
    err_writer.flush()
    return result


def train_api(
    labels: list[str],
    out_path: str = "model/model.pkl",
    config_path: str = "config.json",
    log_callback: LogCallback | None = None,
) -> Path:
    args = argparse.Namespace(config=config_path, labels=labels, out=out_path)
    out_writer = _LogWriter(sys.stdout, log_callback)
    err_writer = _LogWriter(sys.stderr, log_callback)
    with contextlib.redirect_stdout(out_writer), contextlib.redirect_stderr(err_writer):
        cmd_train(args)
    out_writer.flush()
    err_writer.flush()
    return Path(out_path)


def evaluate_api(
    labels: list[str],
    model_path: str | None = None,
    config_path: str = "config.json",
    log_callback: LogCallback | None = None,
) -> EvaluateResult:
    args = argparse.Namespace(config=config_path, labels=labels, model=model_path)
    out_writer = _LogWriter(sys.stdout, log_callback)
    err_writer = _LogWriter(sys.stderr, log_callback)
    with contextlib.redirect_stdout(out_writer), contextlib.redirect_stderr(err_writer):
        result = evaluate_impl(args)
    out_writer.flush()
    err_writer.flush()
    return result


def detect_one_api(
    audio_path: str,
    model_path: str | None = None,
    config_path: str = "config.json",
    ffmpeg_path: str | None = None,
    log_callback: LogCallback | None = None,
) -> tuple[list[tuple[float, float]], float, str]:
    """Detect CM segments from one audio file and return (segments, duration_sec, mode)."""
    ensure_ffmpeg_on_path(ffmpeg_path)

    input_path = Path(audio_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Audio not found: {input_path}")

    cfg_path = Path(config_path).expanduser().resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    model = _resolve_model_path(argparse.Namespace(model=model_path), cfg)

    cache_dir = Path.cwd() / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    wav_path = cache_dir / f"review_{input_path.stem}_{abs(hash(str(input_path))) % 1000000}_16k_mono.wav"
    decode_to_wav(input_path, wav_path, int(cfg["sample_rate"]))
    x = read_wav_pcm16(wav_path)
    sr = int(cfg["sample_rate"])
    duration = x.size / sr
    segments, mode, *_ = _detect_with_fallback(x, cfg, model)
    if log_callback:
        log_callback(f"検出モード: {mode} / 区間数: {len(segments)}")
    return [(float(s.start), float(s.end)) for s in segments], float(duration), mode


def cut_one_api(
    audio_path: str,
    segments: list[tuple[float, float]],
    out_path: str,
    config_path: str = "config.json",
    ffmpeg_path: str | None = None,
    keep_parts: bool = False,
) -> Path:
    """Cut one audio file with explicit CM segments and return output path."""
    ensure_ffmpeg_on_path(ffmpeg_path)

    cfg_path = Path(config_path).expanduser().resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    input_path = Path(audio_path).expanduser().resolve()
    output_path = Path(out_path).expanduser().resolve()

    if input_path == output_path:
        raise ValueError("Output path must differ from input path")

    cm_segments = [Segment(float(s), float(e), 0.0) for s, e in segments if float(e) > float(s)]
    duration = ffprobe_duration(input_path)
    keep = complement_segments(duration, cm_segments)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cut_mp3(
        input_path=input_path,
        keep_segments=keep,
        out_path=output_path,
        q=int(cfg.get("mp3_quality_q", 2)),
        keep_parts=keep_parts,
    )
    return output_path


def labels_from_segments(audio_path: str, segments: list[tuple[float, float]], duration_sec: float) -> list[dict[str, str]]:
    """Create label rows (cm/program) in the same format as data/labels/sample.csv or template.csv."""
    cm = [Segment(float(s), float(e), 0.0) for s, e in segments if float(e) > float(s)]
    program = complement_segments(float(duration_sec), cm)
    rows: list[dict[str, str]] = []
    for s in program:
        rows.append({"audio_path": audio_path, "start_sec": f"{s.start:.3f}", "end_sec": f"{s.end:.3f}", "label": "program", "note": ""})
    for s in cm:
        rows.append({"audio_path": audio_path, "start_sec": f"{s.start:.3f}", "end_sec": f"{s.end:.3f}", "label": "cm", "note": ""})
    rows.sort(key=lambda r: float(r["start_sec"]))
    return rows


def save_labels_csv(rows: list[dict[str, str]], out_csv: Path, template_csv: Path | None = None) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sample = template_csv or Path("data/labels/sample.csv")
    if not sample.exists():
        sample = Path("data/labels/template.csv")
    fieldnames = ["audio_path", "start_sec", "end_sec", "label", "note"]
    if sample.exists():
        with open(sample, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames = list(reader.fieldnames)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return out_csv


def train_evaluate_and_update_model_api(
    labels_patterns: list[str],
    config_path: str = "config.json",
    model_path: str = "model/model.pkl",
    eval_patterns: list[str] | None = None,
    log_callback: LogCallback | None = None,
) -> dict:
    model_p = Path(model_path)
    model_p.parent.mkdir(parents=True, exist_ok=True)
    backup_path: Path | None = None
    old_eval: EvaluateResult | None = None

    if model_p.exists():
        backup_path = model_p.with_name(f"model_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl")
        shutil.copy2(model_p, backup_path)
        if eval_patterns:
            old_eval = evaluate_api(eval_patterns, model_path=str(model_p), config_path=config_path, log_callback=log_callback)

    train_api(labels=labels_patterns, out_path=str(model_p), config_path=config_path, log_callback=log_callback)

    eval_result: EvaluateResult | None = None
    if eval_patterns:
        eval_result = evaluate_api(eval_patterns, model_path=str(model_p), config_path=config_path, log_callback=log_callback)

    delta = None
    if old_eval and eval_result:
        delta = {
            "precision": eval_result.precision - old_eval.precision,
            "recall": eval_result.recall - old_eval.recall,
            "f1": eval_result.f1 - old_eval.f1,
        }
    return {"backup_path": backup_path, "eval": eval_result, "old_eval": old_eval, "delta": delta}
