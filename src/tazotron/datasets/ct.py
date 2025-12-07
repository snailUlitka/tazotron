"""Datasets for CT volumes stored on disk."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torchio as tio
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


class CTDataset(Dataset[tio.Subject]):
    """Loads CT volumes (and optional labelmaps) from a directory of .nii.gz files.

    Returns torchio.Subject objects compatible with downstream transforms such as necrosis injection or DRR rendering.
    """

    def __init__(  # noqa: PLR0913
        self,
        ct_dir: str | Path,
        *,
        label_dir: str | Path | None = None,
        transform: Callable[[tio.Subject], tio.Subject] | None = None,
        ct_key: str = "ct",
        label_key: str = "label",
        extensions: Iterable[str] = ("nii.gz",),
    ) -> None:
        self.ct_dir = Path(ct_dir)
        self.label_dir = Path(label_dir) if label_dir is not None else None
        self.transform = transform
        self.ct_key = ct_key
        self.label_key = label_key
        self.extensions = tuple(extensions)
        self.paths = self._collect_paths()
        if not self.paths:
            msg = f"No CT files found in {self.ct_dir}"
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.paths)

    def __getitem__(self, index: int) -> tio.Subject:
        """Load one sample as torchio.Subject."""
        ct_path = self.paths[index]
        subject = tio.Subject({self.ct_key: tio.ScalarImage(ct_path)})
        if self.label_dir is not None:
            label_path = self.label_dir / ct_path.name
            if not label_path.exists():
                msg = f"Label file not found for {ct_path.name} in {self.label_dir}"
                raise FileNotFoundError(msg)
            subject[self.label_key] = tio.LabelMap(label_path)
        if self.transform:
            subject = self.transform(subject)
        return subject

    def _collect_paths(self) -> list[Path]:
        return sorted(
            path
            for path in self.ct_dir.iterdir()
            if path.is_file() and any(str(path).endswith(ext) for ext in self.extensions)
        )
