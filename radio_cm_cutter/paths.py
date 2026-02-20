"""Path utilities."""

from datetime import datetime
from pathlib import Path


def default_batch_output_dir(folder: Path) -> Path:
    return folder / "cut_output" / datetime.now().strftime("%Y%m%d_%H%M%S")
