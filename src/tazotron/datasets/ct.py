"""Datasets for CT volumes stored on disk."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torchio as tio
from diffdrr.data import read
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Callable


class CTDataset(Dataset[tio.Subject]):
    """Loads CT volumes (and labelmaps) from a directory of .nii.gz files.

    Returns torchio.Subject objects compatible with downstream transforms such as necrosis injection or DRR rendering.
    """

    def __init__(
        self,
        data_path: str | Path,
        *,
        ct_name: str | Path = "ct.nii.gz",
        label_femoral_head_left: str | Path = "label_femoral_head_left.nii.gz",
        label_femoral_head_right: str | Path = "label_femoral_head_right.nii.gz",
        transform: Callable[[tio.Subject], tio.Subject] | None = None,
    ) -> None:
        self.data_path = Path(data_path)

        if not self.data_path.is_dir():
            msg = f"Data path {self.data_path} is not a directory"
            raise ValueError(msg)

        self.ct_name = Path(ct_name)
        self.label_femoral_head_left = Path(label_femoral_head_left)
        self.label_femoral_head_right = Path(label_femoral_head_right)

        self.transform = transform

        self.paths = self._collect_paths()

        if not self.paths:
            msg = f"No CT files found in {self.data_path}"
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.paths)

    def __getitem__(self, index: int) -> tio.Subject:
        """Load one sample as torchio.Subject."""
        ct_path = self.paths[index]

        label_left_path = ct_path.parent / str(self.label_femoral_head_left)
        label_right_path = ct_path.parent / str(self.label_femoral_head_right)

        ct = tio.ScalarImage(ct_path)

        label_left = tio.LabelMap(label_left_path)
        label_right = tio.LabelMap(label_right_path)

        if label_left.data.shape != label_right.data.shape:
            msg = "Labelmap shape mismatch for femoral head labels."
            raise ValueError(msg)

        combined = torch.zeros_like(label_left.data)
        combined = torch.where(label_left.data > 0, torch.ones_like(combined), combined)
        combined = torch.where(label_right.data > 0, torch.full_like(combined, 2), combined)

        label = tio.LabelMap(tensor=combined, affine=label_left.affine)
        subject = read(volume=ct, labelmap=label, label=label)

        if self.transform:
            subject = self.transform(subject)

        return subject

    def _collect_paths(self) -> list[Path]:
        ct_paths: list[Path] = []

        for entry in sorted(self.data_path.iterdir()):
            if not entry.is_dir():
                continue

            ct_path = entry / str(self.ct_name)

            if ct_path.is_file():
                ct_paths.append(ct_path)

        return ct_paths
