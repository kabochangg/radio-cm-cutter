"""Programmatic API for GUI and integrations."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

from .cli import ProcessFolderResult, _process_folder_impl

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
    )

    out_writer = _LogWriter(sys.stdout, log_callback)
    err_writer = _LogWriter(sys.stderr, log_callback)
    with contextlib.redirect_stdout(out_writer), contextlib.redirect_stderr(err_writer):
        result = _process_folder_impl(args)
    out_writer.flush()
    err_writer.flush()
    return result
