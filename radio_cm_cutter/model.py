"""Model loading/training/evaluation helpers and metadata."""

from pathlib import Path
from typing import Any

from .cli import _load_model, cmd_evaluate, cmd_train, compute_frame_features

MODEL_VERSION = "1"


def load_model(path: Path) -> dict[str, Any]:
    return _load_model(path)


__all__ = [
    "MODEL_VERSION",
    "load_model",
    "compute_frame_features",
    "cmd_train",
    "cmd_evaluate",
]
