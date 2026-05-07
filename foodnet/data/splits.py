"""Loaders for split CSVs and class-name list."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def load_split(csv_path: Path) -> tuple[list[str], list[int]]:
    """Return (filepaths, labels) lists from a Phase 2/3 split CSV."""
    df = pd.read_csv(csv_path)
    return df["filepath"].astype(str).tolist(), df["label"].astype(int).tolist()


def load_class_names(classes_txt: Path) -> list[str]:
    """One class per line, in label-id order."""
    return [line.strip() for line in Path(classes_txt).read_text().splitlines() if line.strip()]
