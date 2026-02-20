from __future__ import annotations

import sys
from pathlib import Path


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

