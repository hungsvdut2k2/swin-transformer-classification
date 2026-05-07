"""SplitsDataset: csv-driven Food-101 dataset with PIL-error resample."""
from __future__ import annotations
import logging
import random
from pathlib import Path
from typing import Callable, ClassVar
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from foodnet.data.splits import load_split

log = logging.getLogger(__name__)


class SplitsDataset(Dataset):
    """Reads a Phase 2/3 split CSV. On a corrupt/missing file, logs once per
    path and resamples a random index from the same dataset."""

    _warned: ClassVar[set[str]] = set()

    def __init__(
        self,
        csv_path: Path,
        images_root: Path,
        transform: Callable | None = None,
    ) -> None:
        self.images_root = Path(images_root)
        self.transform = transform
        self.filepaths, self.labels = load_split(Path(csv_path))
        self._rng = random.Random(0)

    def __len__(self) -> int:
        return len(self.filepaths)

    def _open(self, idx: int) -> tuple[Image.Image, int]:
        rel = self.filepaths[idx]
        full = self.images_root / rel
        img = Image.open(full).convert("RGB")
        return img, self.labels[idx]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        try:
            img, label = self._open(idx)
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            rel = self.filepaths[idx]
            if rel not in self._warned:
                self._warned.add(rel)
                log.warning("SplitsDataset: skipping unreadable file %s (%s)", rel, type(e).__name__)
            return self.__getitem__(self._rng.randrange(len(self.filepaths)))
        if self.transform is not None:
            img = self.transform(img)
        return img, label
