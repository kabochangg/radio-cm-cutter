from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "radio-cm-cutter"

# ダウンロードURLはGUI実装に直書きせず、定数として分離する。
FFMPEG_DOWNLOAD_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
FFMPEG_ARCHIVE_NAME = "ffmpeg-release-essentials.zip"


def app_base_dir() -> Path:
    """Return a stable base dir for source execution and frozen EXE."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def bundled_resource_path(*parts: str) -> Path:
    """Resolve resource path even when bundled by PyInstaller."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidate = Path(meipass, *parts)
        if candidate.exists():
            return candidate
    return app_base_dir().joinpath(*parts)


def config_dir() -> Path:
    appdata = os.environ.get("APPDATA")
    base = Path(appdata) if appdata else (Path.home() / ".config")
    out = base / APP_NAME
    out.mkdir(parents=True, exist_ok=True)
    return out


def logs_dir() -> Path:
    local = os.environ.get("LOCALAPPDATA")
    if local:
        base = Path(local)
    else:
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else (Path.home() / ".local" / "state")
    out = base / APP_NAME / "logs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def ffmpeg_dir() -> Path:
    local = os.environ.get("LOCALAPPDATA")
    if local:
        base = Path(local)
    else:
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else (Path.home() / ".local" / "share")
    out = base / APP_NAME / "ffmpeg"
    out.mkdir(parents=True, exist_ok=True)
    return out
