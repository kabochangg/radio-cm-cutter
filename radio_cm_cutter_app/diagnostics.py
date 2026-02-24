from __future__ import annotations

import json
import os
import pickle
import platform
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from radio_cm_cutter_app import __version__ as APP_SEMVER
from radio_cm_cutter_app.support import app_base_dir, config_dir, logs_dir


def detect_ffmpeg(selected_ffmpeg: str | None = None) -> dict[str, str]:
    selected = (selected_ffmpeg or "").strip()
    detected = shutil.which("ffmpeg") or ""
    ffprobe = shutil.which("ffprobe") or ""
    selected_state = "未指定"
    selected_resolved = ""

    if selected:
        p = Path(selected).expanduser()
        if p.is_file():
            selected_resolved = str(p.resolve())
            selected_state = "ファイル指定"
        elif p.is_dir():
            selected_resolved = str(p.resolve())
            selected_state = "フォルダ指定"
        else:
            selected_resolved = str(p)
            selected_state = "存在しないパス"

    return {
        "selected_path": selected,
        "selected_path_state": selected_state,
        "selected_path_resolved": selected_resolved,
        "path_ffmpeg": detected,
        "path_ffprobe": ffprobe,
    }


def read_model_metadata(model_path: str | None) -> dict[str, str]:
    result = {
        "model_path": (model_path or "").strip(),
        "model_exists": "no",
        "model_type": "",
        "model_metadata": "",
    }
    p = Path(result["model_path"]).expanduser()
    if not result["model_path"]:
        return result
    if not p.exists() or not p.is_file():
        return result

    result["model_exists"] = "yes"
    try:
        with p.open("rb") as f:
            obj = pickle.load(f)
        result["model_type"] = type(obj).__name__
        if isinstance(obj, dict):
            meta_keys = [k for k in ("version", "created_at", "feature_names", "model_type", "notes") if k in obj]
            if meta_keys:
                meta = {k: obj[k] for k in meta_keys}
                result["model_metadata"] = json.dumps(meta, ensure_ascii=False)
    except Exception as exc:
        result["model_metadata"] = f"read_error: {exc}"
    return result


def collect_diagnostics(settings_path: Path, model_path: str | None, ffmpeg_path: str | None) -> dict[str, str]:
    py_ver = platform.python_version()
    try:
        sklearn_ver = version("scikit-learn")
    except PackageNotFoundError:
        sklearn_ver = "not installed"

    ffmpeg = detect_ffmpeg(ffmpeg_path)
    model = read_model_metadata(model_path)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "app_semver": APP_SEMVER,
        "python_version": py_ver,
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "base_dir": str(app_base_dir()),
        "settings_path": str(settings_path),
        "config_dir": str(config_dir()),
        "logs_dir": str(logs_dir()),
        "scikit_learn_version": sklearn_ver,
        **ffmpeg,
        **model,
    }


def to_text(data: dict[str, str]) -> str:
    lines = ["radio-cm-cutter diagnostics", ""]
    for k in sorted(data.keys()):
        lines.append(f"{k}={data[k]}")
    return "\n".join(lines) + "\n"


def export_diagnostic_zip(diag: dict[str, str], settings_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_zip = config_dir() / f"diagnostics_{stamp}.zip"

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "versions.txt").write_text(to_text(diag), encoding="utf-8")

        cfg_dir = root / "config"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        if settings_path.exists():
            shutil.copy2(settings_path, cfg_dir / settings_path.name)

        logs_src = logs_dir()
        logs_dst = root / "logs"
        logs_dst.mkdir(parents=True, exist_ok=True)
        if logs_src.exists():
            for p in logs_src.glob("*.log"):
                shutil.copy2(p, logs_dst / p.name)

        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in root.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=str(p.relative_to(root)))

    return out_zip
