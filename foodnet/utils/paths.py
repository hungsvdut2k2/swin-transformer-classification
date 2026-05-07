"""Environment detection + small path helpers."""
from __future__ import annotations
from pathlib import Path


def detect_env() -> str:
    """Return 'kaggle' if running on Kaggle, else 'local'."""
    if Path("/kaggle/input").exists():
        return "kaggle"
    return "local"


def ensure_dir(path: Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
