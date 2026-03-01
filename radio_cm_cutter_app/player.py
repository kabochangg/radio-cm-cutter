from __future__ import annotations

import subprocess
from pathlib import Path


class FFPlayPlayer:
    def __init__(self, audio_path: Path) -> None:
        self.audio_path = audio_path
        self.proc: subprocess.Popen | None = None

    def play_range(self, start: float, end: float) -> None:
        self.stop()
        dur = max(0.1, end - start)
        cmd = [
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{dur:.3f}",
            "-i",
            str(self.audio_path),
        ]
        self.proc = subprocess.Popen(cmd)

    def play_preview_around(self, sec: float, radius: float = 2.0) -> None:
        self.stop()
        start = max(0.0, sec - radius)
        dur = max(0.1, radius * 2.0)
        cmd = [
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{dur:.3f}",
            "-i",
            str(self.audio_path),
        ]
        self.proc = subprocess.Popen(cmd)

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
